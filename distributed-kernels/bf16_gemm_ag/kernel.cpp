#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
#include <hip/hip_cooperative_groups.h>
namespace cg = cooperative_groups;
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2; // 32

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;

#define NUM_PRODUCER_WORKERS (8)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

// Persistent All-Gather GEMM with In-Kernel Cooperative Staging
//
// Single kernel launch. For each source rank r:
//   1. ALL workgroups cooperatively copy A_shard[r] from remote XGMI → local staging
//   2. Grid-wide sync (cooperative_groups)
//   3. Grid-stride GEMM: each workgroup processes output tiles, reading A from local staging
//   4. Grid-wide sync before next rank
//
// Launched with hipLaunchCooperativeKernel (occupancy-limited grid).
struct ag_globals {
    gl<bf16, -1, -1, -1, -1> a_shard;   // [M, K_local] on iris heap
    gl<bf16, -1, -1, -1, -1> a_staging; // [M, K_local] local HBM staging buffer
    gl<bf16, -1, -1, -1, -1> b;         // [N, K] local
    gl<bf16, -1, -1, -1, -1> c;         // [M, N] local output
    iris::iris_device_view iris_ctx;

    int M;
    int N;
    int K;
    int K_local;
    int world_size;

    hipStream_t stream;
    dim3 grid()  { return dim3(ceil_div(N, NEW_COL_BLOCK_SIZE),
                              ceil_div(M, NEW_ROW_BLOCK_SIZE)); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_persistent(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    // Grid info
    int wg_id = blockIdx.x;  // 1D grid for cooperative launch
    int num_wgs = gridDim.x;

    // Warp roles
    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    // Output tile dimensions
    int num_tiles_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    int num_tiles_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    int total_output_tiles = num_tiles_m * num_tiles_n;
    int K_local_tiles = g.K_local / BLOCK_SIZE;

    // Pointer translation setup
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    // Prefill swizzled offsets (same for all tiles/ranks)
    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile + 1]; // +1 for leftover
    uint32_t swizzled_offsets_B[memcpy_per_tile + 1];
    G::prefill_swizzled_offsets(As[0][0][0], g.a_staging, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    // Cooperative copy setup
    int global_tid = wg_id * blockDim.x + threadIdx.x;
    int total_threads = num_wgs * blockDim.x;
    int total_elements = g.M * g.K_local;
    int num_vec = total_elements / 8; // 8 bf16 per int4

    cg::grid_group grid = cg::this_grid();

    // ═══════════════════════════════════════════════════════════
    // Main loop: iterate over source ranks
    // ═══════════════════════════════════════════════════════════
    for (int r = 0; r < g.world_size; r++) {

        // ─── Phase 1: Cooperative copy of rank r's A shard → local staging ───
        uintptr_t base_r = g.iris_ctx.get_heap_base(r);
        intptr_t delta = (intptr_t)base_r - (intptr_t)local_base;
        bf16* remote_a = reinterpret_cast<bf16*>(
            (uintptr_t)g.a_shard.raw_ptr + delta);

        int4* dst4 = reinterpret_cast<int4*>(g.a_staging.raw_ptr);
        const int4* src4 = reinterpret_cast<const int4*>(remote_a);
        for (int i = global_tid; i < num_vec; i += total_threads) {
            dst4[i] = src4[i];
        }

        grid.sync();

        // ─── Phase 2: Grid-stride GEMM over output tiles ───
        int k_offset = r * K_local_tiles;

        for (int tile_id = wg_id; tile_id < total_output_tiles; tile_id += num_wgs) {
            // Map tile_id to (pid_m, pid_n) with WGM grouping for M-locality
            const int WGM = 4;
            int num_wgid_in_group = WGM * num_tiles_n;
            int group_id = tile_id / num_wgid_in_group;
            int first_pid_m = group_id * WGM;
            int group_size_m = min(num_tiles_m - first_pid_m, WGM);
            int pid_m = first_pid_m + ((tile_id % num_wgid_in_group) % group_size_m);
            int pid_n = (tile_id % num_wgid_in_group) / group_size_m;
            int row = pid_m * M_BLOCK;
            int col = pid_n * N_BLOCK;

            // Initialize accumulators
            if (is_consumer) {
                if (r == 0) {
                    zero(C_accum[0][0]); zero(C_accum[0][1]);
                    zero(C_accum[1][0]); zero(C_accum[1][1]);
                } else {
                    // Load existing C (bf16→float32 automatic)
                    load(C_accum[0][0], g.c, {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
                    load(C_accum[0][1], g.c, {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
                    load(C_accum[1][0], g.c, {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
                    load(C_accum[1][1], g.c, {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
                }
            }

            // Prefetch tile 0: A from LOCAL staging, B at rank's K offset
            if (is_producer) {
                #pragma unroll
                for (int m = 0; m < M_BLOCK; m++) {
                    G::load<2, false>(As[0][m][0], g.a_staging, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
                    G::load<2, false>(As[0][m][1], g.a_staging, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
                }
                #pragma unroll
                for (int n = 0; n < N_BLOCK; n++) {
                    G::load<2, false>(Bs[0][n][0], g.b, {0, 0, col*2 + 2*n + 0, k_offset + 0}, swizzled_offsets_B);
                    G::load<2, false>(Bs[0][n][1], g.b, {0, 0, col*2 + 2*n + 1, k_offset + 0}, swizzled_offsets_B);
                }
                __builtin_amdgcn_s_waitcnt(0);
            }
            __syncthreads();

            // K-tile loop
            int tic = 0, toc = 1;
            for (int tile = 0; tile < K_local_tiles - 1; tile++, tic ^= 1, toc ^= 1) {
                int next_k = tile + 1;

                if (is_producer) {
                    #pragma unroll
                    for (int m = 0; m < M_BLOCK; m++) {
                        G::load<2, false>(As[toc][m][0], g.a_staging, {0, 0, row*2 + 2*m + 0, next_k}, swizzled_offsets_A);
                        G::load<2, false>(As[toc][m][1], g.a_staging, {0, 0, row*2 + 2*m + 1, next_k}, swizzled_offsets_A);
                    }
                    #pragma unroll
                    for (int n = 0; n < N_BLOCK; n++) {
                        G::load<2, false>(Bs[toc][n][0], g.b, {0, 0, col*2 + 2*n + 0, k_offset + next_k}, swizzled_offsets_B);
                        G::load<2, false>(Bs[toc][n][1], g.b, {0, 0, col*2 + 2*n + 1, k_offset + next_k}, swizzled_offsets_B);
                    }
                    __builtin_amdgcn_s_waitcnt(0);
                } else if (is_consumer) {
                    A_slice a0;
                    B_slice b0, b1;

                    auto st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
                    load(b0, st_b);
                    auto st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
                    load(a0, st_a);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    __builtin_amdgcn_s_setprio(1);
                    mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
                    __builtin_amdgcn_s_setprio(0);

                    st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
                    load(b1, st_b);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    __builtin_amdgcn_s_setprio(1);
                    mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
                    __builtin_amdgcn_s_setprio(0);

                    st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
                    load(a0, st_a);
                    asm volatile("s_waitcnt lgkmcnt(0)");
                    __builtin_amdgcn_s_setprio(1);
                    mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
                    mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
                    __builtin_amdgcn_s_setprio(0);
                }
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
            }

            // Last tile — no prefetch needed
            if (is_consumer) {
                A_slice a0;
                B_slice b0, b1;

                auto st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][0], {0, 0});
                load(b0, st_b);
                auto st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][0], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum[0][0], a0, b0, C_accum[0][0]);
                __builtin_amdgcn_s_setprio(0);

                st_b = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(Bs[tic][local_warp_id][1], {0, 0});
                load(b1, st_b);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum[0][1], a0, b1, C_accum[0][1]);
                __builtin_amdgcn_s_setprio(0);

                st_a = subtile_inplace<HALF_BLOCK_SIZE, BLOCK_SIZE>(As[tic][consumer_idx][1], {0, 0});
                load(a0, st_a);
                asm volatile("s_waitcnt lgkmcnt(0)");
                __builtin_amdgcn_s_setprio(1);
                mma_ABt(C_accum[1][0], a0, b0, C_accum[1][0]);
                mma_ABt(C_accum[1][1], a0, b1, C_accum[1][1]);
                __builtin_amdgcn_s_setprio(0);
            }

            // Store C
            if (is_consumer) {
                store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
                store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
                store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
                store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
            }
        } // end grid-stride over output tiles

        grid.sync(); // Ensure all tiles done before next rank's copy

    } // end rank loop
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_persistent,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Query occupancy for cooperative launch
    int max_blocks_per_sm;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        (void*)ag_gemm_persistent,
        NUM_THREADS,
        mem_size);

    int num_sms;
    hipDeviceGetAttribute(&num_sms, hipDeviceAttributeMultiprocessorCount, 0);
    int max_grid = max_blocks_per_sm * num_sms;

    int desired_grid = g.grid().x * g.grid().y; // 1920
    int actual_grid = min(desired_grid, max_grid);

    // Launch cooperative kernel (1D grid)
    void* args[] = {&g};
    hipLaunchCooperativeKernel(
        (void*)ag_gemm_persistent,
        dim3(actual_grid),
        dim3(NUM_THREADS),
        args,
        mem_size,
        g.stream);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module — persistent all-gather GEMM with in-kernel staging";
    py::bind_function<dispatch_ag_gemm>(m, "dispatch_ag_gemm",
        &ag_globals::a_shard,
        &ag_globals::a_staging,
        &ag_globals::b,
        &ag_globals::c,
        &ag_globals::iris_ctx,
        &ag_globals::M,
        &ag_globals::N,
        &ag_globals::K,
        &ag_globals::K_local,
        &ag_globals::world_size
    );
}
