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
    // Grid tiles over (M, N_local) — each workgroup handles one (M_tile, N_local_tile)
    // The rank loop is inside the kernel
    dim3 grid()  { return dim3(ceil_div(N_local, NEW_COL_BLOCK_SIZE),
                              ceil_div(M, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

// Inner GEMM: compute C_slice = A @ remote_b^T for one source rank's B shard
// Writes to the appropriate N_local-wide column slice of C
__device__ void ag_gemm_inner(
    const gl<bf16, -1, -1, -1, -1>& a,
    gl<bf16, -1, -1, -1, -1>& remote_b,
    const gl<bf16, -1, -1, -1, -1>& c,
    int M, int N_local, int K, int col_offset_tiles,
    // Shared memory (pre-allocated by caller)
    st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s> (&As)[2][M_BLOCK][2],
    st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s> (&Bs)[2][N_BLOCK][2],
    // Workgroup position
    int row, int col,
    // Warp info
    int local_warp_id, int warp_group_id, bool is_producer, bool is_consumer, int consumer_idx)
{
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], remote_b, swizzled_offsets_B);

    int tic = 0;
    int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m][0], a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
            G::load<2, false>(As[tic][m][1], a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
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
    int num_tiles = K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-1; ++tile, tic ^= 1, toc ^= 1) {
        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m][0], a, {0, 0, row*2 + 2*m + 0, tile + 1}, swizzled_offsets_A);
                G::load<2, false>(As[toc][m][1], a, {0, 0, row*2 + 2*m + 1, tile + 1}, swizzled_offsets_A);
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

    // Store to C — local write
    if (is_consumer) {
        int c_col = col_offset_tiles + col;
        store(c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 0});
        store(c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 1});
        store(c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 0});
        store(c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 1});
    }
    __syncthreads();
}

__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_tk(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    // Workgroup ID + swizzle (computed once, reused across rank iterations)
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N_local, NEW_COL_BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    int row = pid_m * M_BLOCK;
    int col = pid_n * N_BLOCK;

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    // Pointer translation setup
    int cur_rank = g.iris_ctx.cur_rank();
    int world_size = g.iris_ctx.world_size();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    // Iterate over all ranks' B shards inside the kernel
    int n_local_tiles_per_rank = (g.N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK;
    for (int source_rank = 0; source_rank < world_size; source_rank++) {
        // Translate b_shard pointer to point at source_rank's shard
        uintptr_t remote_base = g.iris_ctx.get_heap_base(source_rank);
        intptr_t ptr_delta = (intptr_t)remote_base - (intptr_t)local_base;

        // Copy gl on device, swap raw_ptr to translated address
        gl<bf16, -1, -1, -1, -1> remote_b = g.b_shard;
        remote_b.raw_ptr = reinterpret_cast<bf16*>((uintptr_t)g.b_shard.raw_ptr + ptr_delta);

        int col_offset_tiles = source_rank * n_local_tiles_per_rank;

        ag_gemm_inner(g.a, remote_b, g.c,
                      g.M, g.N_local, g.K, col_offset_tiles,
                      As, Bs,
                      row, col,
                      local_warp_id, warp_group_id, is_producer, is_consumer, consumer_idx);
    }
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    ag_gemm_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
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
