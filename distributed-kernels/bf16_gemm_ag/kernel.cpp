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

// Batch 2 ranks per K-pass to halve A re-reads
constexpr int RANK_BATCH = 2;

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
void ag_gemm_tk(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    // 2 sets of accumulators for RANK_BATCH=2
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C0_accum[2][2]; // rank batch 0
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C1_accum[2][2]; // rank batch 1

    // Workgroup ID + swizzle
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
    int n_local_tiles_per_rank = (g.N_local / NEW_COL_BLOCK_SIZE) * N_BLOCK;
    int num_tiles = g.K / BLOCK_SIZE;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], g.a, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b_shard, swizzled_offsets_B);

    // Process ranks in batches of RANK_BATCH=2
    // For each batch: iterate K tiles, load A once, load B from 2 remote ranks
    for (int rank_base = 0; rank_base < world_size; rank_base += RANK_BATCH) {
        int r0 = rank_base;
        int r1 = rank_base + 1;
        // Handle odd world_size: if r1 >= world_size, we only process r0
        bool has_r1 = (r1 < world_size);

        // Translate pointers for both ranks
        uintptr_t remote_base0 = g.iris_ctx.get_heap_base(r0);
        intptr_t delta0 = (intptr_t)remote_base0 - (intptr_t)local_base;
        gl<bf16, -1, -1, -1, -1> remote_b0 = g.b_shard;
        remote_b0.raw_ptr = reinterpret_cast<bf16*>((uintptr_t)g.b_shard.raw_ptr + delta0);

        gl<bf16, -1, -1, -1, -1> remote_b1 = g.b_shard;
        if (has_r1) {
            uintptr_t remote_base1 = g.iris_ctx.get_heap_base(r1);
            intptr_t delta1 = (intptr_t)remote_base1 - (intptr_t)local_base;
            remote_b1.raw_ptr = reinterpret_cast<bf16*>((uintptr_t)g.b_shard.raw_ptr + delta1);
        }

        // Zero accumulators
        if (is_consumer) {
            zero(C0_accum[0][0]); zero(C0_accum[0][1]);
            zero(C0_accum[1][0]); zero(C0_accum[1][1]);
            zero(C1_accum[0][0]); zero(C1_accum[0][1]);
            zero(C1_accum[1][0]); zero(C1_accum[1][1]);
        }

        // ── K-tile loop: load A once, compute against both B shards ──

        int tic = 0, toc = 1;

        // Prefetch first A tile
        if (is_producer) {
            #pragma unroll
            for (int m = 0; m < M_BLOCK; m++) {
                G::load<2, false>(As[tic][m][0], g.a, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
                G::load<2, false>(As[tic][m][1], g.a, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
            }
            // Load B from rank r0
            #pragma unroll
            for (int n = 0; n < N_BLOCK; n++) {
                G::load<2, false>(Bs[tic][n][0], remote_b0, {0, 0, col*2 + 2*n + 0, 0}, swizzled_offsets_B);
                G::load<2, false>(Bs[tic][n][1], remote_b0, {0, 0, col*2 + 2*n + 1, 0}, swizzled_offsets_B);
            }
            __builtin_amdgcn_s_waitcnt(0);
        }
        __syncthreads();

        for (int tile = 0; tile < num_tiles; tile++, tic ^= 1, toc ^= 1) {
            // ── Phase 1: Compute C0 += A * B_r0^T while prefetching B_r1 ──

            if (is_producer && has_r1) {
                // Load B from rank r1 into toc buffer
                #pragma unroll
                for (int n = 0; n < N_BLOCK; n++) {
                    G::load<2, false>(Bs[toc][n][0], remote_b1, {0, 0, col*2 + 2*n + 0, tile}, swizzled_offsets_B);
                    G::load<2, false>(Bs[toc][n][1], remote_b1, {0, 0, col*2 + 2*n + 1, tile}, swizzled_offsets_B);
                }
                __builtin_amdgcn_s_waitcnt(0);
            } else if (is_consumer) {
                // Compute C0 += A_tic * B_r0_tic
                A_slice a0;
                B_slice b0, b1;

                auto st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
                load(b0, st_b);
                auto st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C0_accum[0][0], a0, b0, C0_accum[0][0]);
                __builtin_amdgcn_s_setprio(0);

                st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
                load(b1, st_b);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C0_accum[0][1], a0, b1, C0_accum[0][1]);
                __builtin_amdgcn_s_setprio(0);

                st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C0_accum[1][0], a0, b0, C0_accum[1][0]);
                mma_ABt(C0_accum[1][1], a0, b1, C0_accum[1][1]);
                __builtin_amdgcn_s_setprio(0);
            }
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();

            if (!has_r1) continue;

            // ── Phase 2: Compute C1 += A * B_r1^T while prefetching next A + B_r0 ──

            if (is_producer) {
                if (tile + 1 < num_tiles) {
                    // Load next A tile
                    #pragma unroll
                    for (int m = 0; m < M_BLOCK; m++) {
                        G::load<2, false>(As[tic][m][0], g.a, {0, 0, row*2 + 2*m + 0, tile + 1}, swizzled_offsets_A);
                        G::load<2, false>(As[tic][m][1], g.a, {0, 0, row*2 + 2*m + 1, tile + 1}, swizzled_offsets_A);
                    }
                    // Load B from rank r0 for next tile
                    #pragma unroll
                    for (int n = 0; n < N_BLOCK; n++) {
                        G::load<2, false>(Bs[tic][n][0], remote_b0, {0, 0, col*2 + 2*n + 0, tile + 1}, swizzled_offsets_B);
                        G::load<2, false>(Bs[tic][n][1], remote_b0, {0, 0, col*2 + 2*n + 1, tile + 1}, swizzled_offsets_B);
                    }
                }
                __builtin_amdgcn_s_waitcnt(0);
            } else if (is_consumer) {
                // Compute C1 += A_toc * B_r1_toc (A is still in tic from phase 1, B_r1 is in toc)
                A_slice a0;
                B_slice b0, b1;

                // A is still valid in As[tic] from phase 1 load
                auto st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[toc][local_warp_id][0], {0, 0});
                load(b0, st_b);
                auto st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C1_accum[0][0], a0, b0, C1_accum[0][0]);
                __builtin_amdgcn_s_setprio(0);

                st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[toc][local_warp_id][1], {0, 0});
                load(b1, st_b);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C1_accum[0][1], a0, b1, C1_accum[0][1]);
                __builtin_amdgcn_s_setprio(0);

                st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C1_accum[1][0], a0, b0, C1_accum[1][0]);
                mma_ABt(C1_accum[1][1], a0, b1, C1_accum[1][1]);
                __builtin_amdgcn_s_setprio(0);
            }
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_barrier();
        }

        // Store C0 (rank r0's contribution)
        if (is_consumer) {
            int c_col0 = r0 * n_local_tiles_per_rank + col;
            store(g.c, C0_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, c_col0 * 2 + local_warp_id * 2 + 0});
            store(g.c, C0_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, c_col0 * 2 + local_warp_id * 2 + 1});
            store(g.c, C0_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, c_col0 * 2 + local_warp_id * 2 + 0});
            store(g.c, C0_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, c_col0 * 2 + local_warp_id * 2 + 1});
        }

        // Store C1 (rank r1's contribution)
        if (is_consumer && has_r1) {
            int c_col1 = r1 * n_local_tiles_per_rank + col;
            store(g.c, C1_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, c_col1 * 2 + local_warp_id * 2 + 0});
            store(g.c, C1_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, c_col1 * 2 + local_warp_id * 2 + 1});
            store(g.c, C1_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, c_col1 * 2 + local_warp_id * 2 + 0});
            store(g.c, C1_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, c_col1 * 2 + local_warp_id * 2 + 1});
        }
        __syncthreads();
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
