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

// Per-rank-iteration kernel args
struct ag_kernel_args {
    gl<bf16, -1, -1, -1, -1> a;        // [M, K] - local
    gl<bf16, -1, -1, -1, -1> remote_b; // [N_local, K] - translated to point at source_rank's shard
    gl<bf16, -1, -1, -1, -1> c;        // [M, N] - full output, local

    int M;
    int N_local;
    int K;
    int col_offset_tiles;  // offset into C's column dimension (in N_BLOCK units)
};

// Host-side globals for pybind11
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
    dim3 grid()  { return dim3(ceil_div(N_local, NEW_COL_BLOCK_SIZE),
                              ceil_div(M, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_tk(ag_kernel_args args) {

    // shared memory
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    // Original WGID
    int wgid = (blockIdx.y * gridDim.x) + blockIdx.x;
    const int NUM_WGS  = gridDim.x * gridDim.y;
    const int WGM = 4;
    // Swizzle chiplet so that wgids are in the same XCD
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    // Swizzle for better L2 within the same XCD
    const int num_pid_m = ceil_div(args.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(args.N_local, NEW_COL_BLOCK_SIZE);
    const int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
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
    G::prefill_swizzled_offsets(As[0][0][0], args.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], args.remote_b, swizzled_offsets_B);

    int tic = 0;
    int toc = 1;
    if (is_producer) {
        #pragma unroll
        for (int m = 0; m < M_BLOCK; m++) {
            G::load<2, false>(As[tic][m][0], args.a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
            G::load<2, false>(As[tic][m][1], args.a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
        }
        #pragma unroll
        for (int n = 0; n < N_BLOCK; n++) {
            G::load<2, false>(Bs[tic][n][0], args.remote_b, {0, 0, col*2 + 2*n + 0, 0}, swizzled_offsets_B);
            G::load<2, false>(Bs[tic][n][1], args.remote_b, {0, 0, col*2 + 2*n + 1, 0}, swizzled_offsets_B);
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
    int num_tiles = args.K / BLOCK_SIZE;
    #pragma unroll
    for (int tile = 0; tile < num_tiles-1; ++tile, tic ^= 1, toc ^= 1) {

        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[toc][m][0], args.a, {0, 0, row*2 + 2*m + 0, tile + 1}, swizzled_offsets_A);
                G::load<2, false>(As[toc][m][1], args.a, {0, 0, row*2 + 2*m + 1, tile + 1}, swizzled_offsets_A);
            }
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[toc][n][0], args.remote_b, {0, 0, col*2 + 2*n + 0, tile + 1}, swizzled_offsets_B);
                G::load<2, false>(Bs[toc][n][1], args.remote_b, {0, 0, col*2 + 2*n + 1, tile + 1}, swizzled_offsets_B);
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
    if (is_consumer) {
        int c_col = args.col_offset_tiles + col;
        store(args.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 0});
        store(args.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, c_col * 2 + local_warp_id * 2 + 1});
        store(args.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 0});
        store(args.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, c_col * 2 + local_warp_id * 2 + 1});
    }
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_tk, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    for (int source_rank = 0; source_rank < g.world_size; source_rank++) {
        // Translate local b_shard pointer to point at source_rank's b_shard
        uintptr_t remote_base = g.iris_ctx.get_heap_base(source_rank);
        intptr_t ptr_delta = (intptr_t)remote_base - (intptr_t)local_base;
        bf16* remote_b_ptr = reinterpret_cast<bf16*>((uintptr_t)g.b_shard.raw_ptr + ptr_delta);

        // Construct gl for remote B shard on host (same shape as local b_shard)
        gl<bf16, -1, -1, -1, -1> remote_b(remote_b_ptr,
            g.b_shard.batch(), g.b_shard.depth(),
            g.b_shard.rows(), g.b_shard.cols());

        // Column offset: source_rank * (N_local / BLOCK_SIZE) in N_BLOCK units
        int col_offset_tiles = source_rank * (g.N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK;

        ag_kernel_args args;
        args.a = g.a;
        args.remote_b = remote_b;
        args.c = g.c;
        args.M = g.M;
        args.N_local = g.N_local;
        args.K = g.K;
        args.col_offset_tiles = col_offset_tiles;

        ag_gemm_tk<<<g.grid(), g.block(), mem_size, g.stream>>>(args);
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
