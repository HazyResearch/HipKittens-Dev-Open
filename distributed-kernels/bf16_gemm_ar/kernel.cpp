#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include "../kittens_iris.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int DOT_SLICE = 32;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2; // 32

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (4)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;


struct micro_globals {
    gl<bf16, -1, -1, -1, -1> a_inbox, b, c;
    iris::iris_device_view iris_ctx;

    int M;
    int N;
    int K_local;

    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, NEW_COL_BLOCK_SIZE), ceil_div(M, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};


/***************************************************
* Iterate through tiles of A. 
* Load into shared mem.
* Store to destination rank global memory.
***************************************************/
template<int axis, ducks::gl::all GL>
__device__ inline static void kittens_allgather(
    const GL &dst, 
    int row, 
    int col, 
    const int M, 
    const int N, 
    const int K_local, 
    iris::iris_device_view& iris_ctx
) {
    using U = typename GL::dtype;
    int laneid = kittens::laneid();
    int cur_rank = iris_ctx.cur_rank();

    // Tile data structure.
    using RT = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
    using T = base_types::packing<typename RT::dtype>::unpacked_type;
    RT tile;
    constexpr int packing = base_types::packing<T>::num();
    const int row_offset = tile.base_tile_stride*(laneid/tile.base_tile_cols);
    const int col_offset = laneid%tile.base_tile_cols;

    // Global memory pointer.
    const coord<RT> idx = {0,0,0,0};
    U *dst_ptr = (U*)&dst[(idx.template unit_coord<axis, 3>())];
    U* dst_base = reinterpret_cast<U*>(dst_ptr);
    int base_offset = cur_rank * ( M * K_local );

    // Iterate through tiles on local rank.
    int num_tiles = K_local / BLOCK_SIZE;
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        for (int tile_m = 0; tile_m < M_BLOCK; tile_m++) {

            int tile_offset = base_offset + tile_m * K_local + tile_k*BLOCK_SIZE;

            // Load the tile from global memory.
            load<2>(tile, dst, {0, 0, 0, 0});

            // Send to all other ranks.
            for (int rank = 0; rank < iris_ctx.world_size(); rank++) {

                // I already have the data for this rank.
                if (rank == iris_ctx.cur_rank()) {
                    continue;
                }
                                
                #pragma unroll
                for(int i = 0; i < tile.height; i++) {
                    #pragma unroll
                    for(int j = 0; j < tile.width; j++) {
                        const int tile_col = j*tile.base_tile_cols + col_offset;
                        #pragma unroll
                        for(int k = 0; k < tile.base_tile_num_strides; k++) {
                            int tile_row = i*tile.base_tile_rows + row_offset + k*tile.base_tile_elements_per_stride_group;
                            #pragma unroll
                            for(int l = 0; l < tile.base_tile_stride / packing; l++) {
                                int idx = l + k * tile.base_tile_stride / packing;
                                
                                // Convert values
                                U val_x = base_types::convertor<U, T>::convert(tile.tiles[i][j].data[idx].x);
                                U val_y = base_types::convertor<U, T>::convert(tile.tiles[i][j].data[idx].y);
                                
    //                             // Use iris.store
    //                             int k_global = tile_k*BLOCK_SIZE + tile_col;
    //                             int dst_offset_0 = base_offset + ((row*BLOCK_SIZE) + tile_m*BLOCK_SIZE + tile_row + l*2 + 0) * K_local + k_global;
    //                             int dst_offset_1 = base_offset + ((row*BLOCK_SIZE) + tile_m*BLOCK_SIZE + tile_row + l*2 + 1) * K_local + k_global;
    //                             iris_ctx.store(&dst_ptr[dst_offset_0], val_x, rank);
    //                             iris_ctx.store(&dst_ptr[dst_offset_1], val_y, rank);
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ __launch_bounds__(NUM_THREADS, 2)
void micro_allgather_tk(micro_globals g) {
    // L2 cache and LLC cache swizzling.
    int row, col;
    {
        int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
        const int NUM_WGS  = gridDim.x * gridDim.y;
        const int WGM = 4;
        // Swizzle chiplet so that wgids are in the same XCD.
        wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
        // Swizzle for better L2 within the same XCD.
        const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE); // 7680 / 192 = 40
        const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE); // 7680 / 256 = 30
        const int num_wgid_in_group = WGM * num_pid_n;
        int group_id = wgid / num_wgid_in_group;
        int first_pid_m = group_id * WGM;
        int group_size_m = min(num_pid_m - first_pid_m, WGM);
        int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        int pid_n = (wgid % num_wgid_in_group) / group_size_m;
        // Assign the tile's row/column based on the pid_m and pid_n.
        row = pid_m * M_BLOCK; 
        col = pid_n * N_BLOCK; 
    }
    // All-gather the A tile.
    kittens_allgather<2>(g.a_inbox, row, col, g.M, g.N, g.K_local, g.iris_ctx);
}

void dispatch_micro(micro_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    // hipFuncSetAttribute((void*)micro_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipLaunchKernelGGL(micro_allgather_tk, g.grid(), g.block(), mem_size, g.stream, g);
    micro_allgather_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    // micro_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
  }

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_function<dispatch_micro>(m, "dispatch_micro", 
        &micro_globals::a_inbox, 
        &micro_globals::b, 
        &micro_globals::c,
        &micro_globals::iris_ctx,
        &micro_globals::M,
        &micro_globals::N,
        &micro_globals::K_local
    );
}



// __global__ __launch_bounds__(NUM_THREADS, 2)
// void micro_tk(micro_globals g) {

//     // Define tile data structures.
//     extern __shared__ alignment_dummy __shm[];
//     shared_allocator al((int*)&__shm[0]);
//     using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
//     using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
//     ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
//     ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
//     rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

//     // L2 cache and LLC cache swizzling.
//     {
//         int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
//         const int NUM_WGS  = gridDim.x * gridDim.y;
//         const int WGM = 4;
//         // Swizzle chiplet so that wgids are in the same XCD.
//         wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
//         // Swizzle for better L2 within the same XCD.
//         const int num_pid_m = ceil_div(g.M_local, NEW_ROW_BLOCK_SIZE); // 7680 / 192 = 40
//         const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE); // 7680 / 256 = 30
//         const int num_wgid_in_group = WGM * num_pid_n;
//         int group_id = wgid / num_wgid_in_group;
//         int first_pid_m = group_id * WGM;
//         int group_size_m = min(num_pid_m - first_pid_m, WGM);
//         int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
//         int pid_n = (wgid % num_wgid_in_group) / group_size_m;
//         // Assign the tile's row/column based on the pid_m and pid_n.
//         int row = pid_m * M_BLOCK; 
//         int col = pid_n * N_BLOCK; 
//     }

//     // Wave partitioning.
//     {
//         int warp_id = kittens::warpid();
//         int local_warp_id = warp_id % 4;
//         int warp_group_id = warp_id / 4;
//         bool is_producer = (warp_group_id == 0);
//         bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
//         int consumer_idx = is_consumer ? warp_group_id - 1 : 0;
//     }

//     // Pre-swizzle the addresses for I/O.
//     {
//         using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
//         constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
//         constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
//         constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
//         uint32_t swizzled_offsets_A[memcpy_per_tile];
//         uint32_t swizzled_offsets_B[memcpy_per_tile];
//         G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
//         G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);
//     }


//     for (int rank = 0; rank < g.iris_ctx.world_size(); rank++) {
    
//         int tic = 0;
//         int toc = 1;
//         if (is_producer) {
//             #pragma unroll
//             for (int m = 0; m < M_BLOCK; m++) {
//                 G::load<2, false>(As[tic][m][0], g.a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
//                 G::load<2, false>(As[tic][m][1], g.a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
//             }
//             #pragma unroll
//             for (int n = 0; n < N_BLOCK; n++) {
//                 G::load<2, false>(Bs[tic][n][0], g.b, {0, 0, col*2 + 2*n + 0, 0}, swizzled_offsets_B);
//                 G::load<2, false>(Bs[tic][n][1], g.b, {0, 0, col*2 + 2*n + 1, 0}, swizzled_offsets_B);
//             }
//             __builtin_amdgcn_s_waitcnt(0);
//         }
//         __syncthreads();


//         if (is_consumer) {
//             zero(C_accum[0][0]);
//             zero(C_accum[0][1]);
//             zero(C_accum[1][0]); 
//             zero(C_accum[1][1]);
//         }
//         int num_tiles = g.K / BLOCK_SIZE;
//         #pragma unroll
//         for (int tile = 0; tile < num_tiles-1; ++tile, tic ^= 1, toc ^= 1) {

//             if (is_producer) {
//                 #pragma unroll
//                 for (int m = 0; m < M_BLOCK; m++) {
//                     G::load<2, false>(As[toc][m][0], g.a, {0, 0, row*2 + 2*m + 0, tile + 1}, swizzled_offsets_A);
//                     G::load<2, false>(As[toc][m][1], g.a, {0, 0, row*2 + 2*m + 1, tile + 1}, swizzled_offsets_A);
//                 }
//                 #pragma unroll
//                 for (int n = 0; n < N_BLOCK; n++) {
//                     G::load<2, false>(Bs[toc][n][0], g.b, {0, 0, col*2 + 2*n + 0, tile + 1}, swizzled_offsets_B);
//                     G::load<2, false>(Bs[toc][n][1], g.b, {0, 0, col*2 + 2*n + 1, tile + 1}, swizzled_offsets_B);
//                 }
//                 __builtin_amdgcn_s_waitcnt(0);
//             } else if (is_consumer) {
//                 A_slice a0;
//                 B_slice b0, b1;

//                 auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
//                 load(b0, st_subtile_b);
//                 auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
//                 load(a0, st_subtile_a);
//                 asm volatile("s_waitcnt lgkmcnt(0)");
//                 __builtin_amdgcn_s_setprio(1);
//                 mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
//                 __builtin_amdgcn_s_setprio(0);

//                 st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
//                 load(b1, st_subtile_b);
//                 asm volatile("s_waitcnt lgkmcnt(0)");
//                 __builtin_amdgcn_s_setprio(1);
//                 mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
//                 __builtin_amdgcn_s_setprio(0);

//                 st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
//                 load(a0, st_subtile_a);
//                 asm volatile("s_waitcnt lgkmcnt(0)");
//                 __builtin_amdgcn_s_setprio(1);
//                 mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
//                 mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
//                 __builtin_amdgcn_s_setprio(0);
//             }
//             __builtin_amdgcn_sched_barrier(0);
//             __builtin_amdgcn_s_barrier();
            

//         }
//         if (is_consumer) { 
//             A_slice a0;
//             B_slice b0, b1;

//             auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0],{0, 0});
//             load(b0, st_subtile_b);
//             auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0],{0, 0});
//             load(a0, st_subtile_a);
//             asm volatile("s_waitcnt lgkmcnt(0)");
//             __builtin_amdgcn_s_setprio(1);
//             mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
//             __builtin_amdgcn_s_setprio(0);

//             st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1],{0, 0});
//             load(b1, st_subtile_b);
//             asm volatile("s_waitcnt lgkmcnt(0)");
//             __builtin_amdgcn_s_setprio(1);
//             mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
//             __builtin_amdgcn_s_setprio(0);

//             st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1],{0, 0});
//             load(a0, st_subtile_a);
//             asm volatile("s_waitcnt lgkmcnt(0)");
//             __builtin_amdgcn_s_setprio(1);
//             mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
//             mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
//             __builtin_amdgcn_s_setprio(0);
//         }
//         if (is_consumer) {
//             kittens_store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 0}, g.iris_ctx);
//             kittens_store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, (col + local_warp_id) * 2 + 1}, g.iris_ctx);
//             kittens_store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 0}, g.iris_ctx);
//             kittens_store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, (col + local_warp_id) * 2 + 1}, g.iris_ctx);
//         }

//     }
// }

