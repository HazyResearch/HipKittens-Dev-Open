#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
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

// All-Gather GEMM globals
// C[M, N] = A[M, K] @ B_full[N, K]^T
// B is column-sharded: each rank owns B_shard[N_local, K], N_local = N / world_size
// Each rank has full A, produces full C by iterating over remote B shards via IPC
struct ag_globals {
    gl<bf16, -1, -1, -1, -1> a;       // [M, K] - local, same on all ranks
    gl<bf16, -1, -1, -1, -1> b_shard; // [N_local, K] - local shard of B
    gl<bf16, -1, -1, -1, -1> c;       // [M, N] - full output, local
    iris::iris_device_view iris_ctx;

    int M;
    int N;
    int K;
    int N_local;
    int world_size;

    hipStream_t stream;
    // Grid tiles over (M, N_local) - one rank's B shard at a time
    dim3 grid()  { return dim3(ceil_div(N_local, NEW_COL_BLOCK_SIZE),
                              ceil_div(M, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_tk(ag_globals g, int source_rank, int col_offset_tiles) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    // Compute pointer translation for remote B access
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);
    uintptr_t remote_base = g.iris_ctx.get_heap_base(source_rank);
    intptr_t ptr_delta = (intptr_t)remote_base - (intptr_t)local_base;

    // Construct translated gl for remote B shard
    // The remote rank's b_shard has the same offset from its heap base as ours from ours
    bf16* local_b_ptr = g.b_shard.raw_ptr;
    bf16* remote_b_ptr = reinterpret_cast<bf16*>((uintptr_t)local_b_ptr + ptr_delta);
    // Create a gl pointing to the remote B shard
    gl<bf16, -1, -1, -1, -1> remote_b(remote_b_ptr, nullptr,
        g.b_shard.template shape<0>(),  // batch  (from 4D layout)
        g.b_shard.template shape<1>(),  // depth
        g.b_shard.template shape<2>(),  // rows = N_local
        g.b_shard.template shape<3>()); // cols = K

    // Original WGID
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int WGM = 4;
    // Swizzle chiplet so that wgids are in the same XCD
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N_local, NEW_COL_BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    // Assign the tile's row/column based on pid_m and pid_n
    int row = pid_m * M_BLOCK;
    int col = pid_n * N_BLOCK;  // col within N_local

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], remote_b, swizzled_offsets_B);

    int tic = 0;
    int toc = 1;
    // Producers load first K-tile of A (local) and B (remote via IPC)
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m][0], g.a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
            G::load<2, false>(As[tic][m][1], g.a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++) {
            G::load<2, false>(Bs[tic][n][0], remote_b, {0, 0, col*2 + 2*n + 0, 0}, swizzled_offsets_B);
            G::load<2, false>(Bs[tic][n][1], remote_b, {0, 0, col*2 + 2*n + 1, 0}, swizzled_offsets_B);
        }
        __builtin_amdgcn_s_waitcnt(0);
    }
    __syncthreads();

    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }
    int num_tiles = g.K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-1; ++tile, tic ^= 1, toc ^= 1) {

        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m][0], g.a, {0, 0, row*2 + 2*m + 0, tile + 1}, swizzled_offsets_A);
                G::load<2, false>(As[toc][m][1], g.a, {0, 0, row*2 + 2*m + 1, tile + 1}, swizzled_offsets_A);
            }
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[toc][n][0], remote_b, {0, 0, col*2 + 2*n + 0, tile + 1}, swizzled_offsets_B);
                G::load<2, false>(Bs[toc][n][1], remote_b, {0, 0, col*2 + 2*n + 1, tile + 1}, swizzled_offsets_B);
            }
            __builtin_amdgcn_s_waitcnt(0);
        } else if (is_consumer) {
            A_slice a0;
            B_slice b0, b1;

            auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
            load(b0, st_subtile_b);
            auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
            load(b1, st_subtile_b);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
            __builtin_amdgcn_s_setprio(0);

            st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
            load(a0, st_subtile_a);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
            mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
            __builtin_amdgcn_s_setprio(0);
        }
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Last K-tile
    if (is_consumer) {
        A_slice a0;
        B_slice b0, b1;

        auto st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
        load(b0, st_subtile_b);
        auto st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
        load(b1, st_subtile_b);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
        __builtin_amdgcn_s_setprio(0);

        st_subtile_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
        load(a0, st_subtile_a);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
        mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
        __builtin_amdgcn_s_setprio(0);
    }

    // Store to C — local write, no iris needed
    // Column index into full C = col_offset_tiles + col (within N_local partition)
    if (is_consumer) {
        int c_col = col_offset_tiles + col;
        store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 1});
        store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 1});
    }
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    int cur_rank = g.iris_ctx.cur_rank();
    // N_local in tile units for column offset into C
    // Each HALF_BLOCK_SIZE tile covers 32 rows of B (= 32 columns of C)
    // N_BLOCK tiles per workgroup, each tile is HALF_BLOCK_SIZE
    // col_offset_tiles = source_rank * (N_local / HALF_BLOCK_SIZE) / 2
    // But the gl coord system uses HALF_BLOCK_SIZE as the tile unit
    // In the existing kernel, col is in units of N_BLOCK (each N_BLOCK = 4 tiles of HALF_BLOCK_SIZE)
    // The gl indexing {0,0,row,col} where col is in HALF_BLOCK_SIZE-tile units

    for (int source_rank = 0; source_rank < g.world_size; source_rank++) {
        // Column offset in the same units as 'col' in the kernel (N_BLOCK-sized groups)
        // N_local elements -> N_local / HALF_BLOCK_SIZE half-block tiles -> / 2 for the [2] sub-tiling
        // But col is in N_BLOCK units, and the store uses col * 2 + local_warp_id * 2
        // Actually col is NOT in N_BLOCK units. col = pid_n * N_BLOCK, and store does col*2 + ...
        // The gl coord for columns: the 4th dim is in HALF_BLOCK_SIZE tile units
        // source_rank * N_local elements -> source_rank * N_local / HALF_BLOCK_SIZE tiles
        // In the store: c_col * 2 + local_warp_id * 2 + {0,1}
        // c_col = col_offset_tiles + col, where col = pid_n * N_BLOCK
        // So col_offset_tiles should be in the same units: N_BLOCK units
        // source_rank * (N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK
        // = source_rank * ceil_div(N_local, NEW_COL_BLOCK_SIZE) * N_BLOCK
        // Since N_local should be divisible: source_rank * (N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK
        int col_offset_tiles = source_rank * (g.N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK;
        ag_gemm_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g, source_rank, col_offset_tiles);
    }
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module — all-gather GEMM";
    py::bind_function<dispatch_ag_gemm>(m, "dispatch_ag_gemm",
        &ag_globals::a,
        &ag_globals::b_shard,
        &ag_globals::c,
        &ag_globals::iris_ctx,
        &ag_globals::M,
        &ag_globals::N,
        &ag_globals::K,
        &ag_globals::N_local,
        &ag_globals::world_size
    );
}
