#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <iris/iris.hpp>
using namespace kittens;

constexpr int BLOCK_SIZE = 64;
constexpr int M_BLOCK = 2;
constexpr int N_BLOCK = 4;
constexpr int HALF_BLOCK_SIZE = BLOCK_SIZE / 2; // 32

constexpr int NEW_ROW_BLOCK_SIZE = BLOCK_SIZE * M_BLOCK;   // 128
constexpr int NEW_COL_BLOCK_SIZE = BLOCK_SIZE * N_BLOCK;    // 256

#define NUM_PRODUCER_WORKERS (8)
#define NUM_CONSUMER_WORKERS (M_BLOCK * 4)
#define NUM_THREADS ((NUM_PRODUCER_WORKERS + NUM_CONSUMER_WORKERS) * kittens::WARP_THREADS)
#define NUM_PRODUCER_THREADS (NUM_PRODUCER_WORKERS * kittens::WARP_THREADS)

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

// Persistent fused AG-GEMM
// ALL blocks do data movement first, then transition to GEMM via work queue.
// Phase 1: every block cooperatively copies remote A → a_local (max bandwidth)
// Phase 2: blocks grab GEMM tiles from atomic counter, compute, store, repeat
// Accumulators in registers per tile — zero once, accumulate all ranks, store once.

struct ag_globals {
    gl<bf16, -1, -1, -1, -1> a_shard;   // [M, K_local] on iris heap
    gl<bf16, -1, -1, -1, -1> a_local;   // [world_size * M, K_local] local HBM
    gl<bf16, -1, -1, -1, -1> b;         // [N, K] local
    gl<bf16, -1, -1, -1, -1> c;         // [M, N] local output
    iris::iris_device_view iris_ctx;

    int M;
    int N;
    int K;
    int K_local;
    int world_size;
    uintptr_t counters_ptr;      // int[world_size] — per-rank copy-done counters
    uintptr_t work_counter_ptr;  // int — atomic GEMM tile counter
    int num_output_tiles;        // total GEMM tiles

    hipStream_t stream;

    __host__ __device__ int* counters() { return reinterpret_cast<int*>(counters_ptr); }
    __host__ __device__ int* work_counter() { return reinterpret_cast<int*>(work_counter_ptr); }

    // No grid()/block() — dispatch computes persistent grid size
    size_t dynamic_shared_memory() { return 98304 + 16; } // +16 for scratch
};

// Spin on a single counter until >= target
__device__ __forceinline__ void spin_until_ready(volatile int* counter, int target) {
    while (*counter < target) {}
    __threadfence();
}

// ============================================================================
// Persistent fused kernel: all blocks copy, then all blocks compute
// ============================================================================
__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_persistent(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    // Scratch space after tile arrays for broadcasting tile_id
    int* sh_tile_id = reinterpret_cast<int*>(reinterpret_cast<char*>(&__shm[0]) + 98304);

    int total_blocks = gridDim.x;
    int* counters = g.counters();
    int* work_ctr = g.work_counter();

    // ════════════════════════════════════════════════════════
    // Phase 1: ALL blocks cooperatively copy remote A → a_local
    // Maximum XGMI bandwidth — every CU pulling data
    // ════════════════════════════════════════════════════════
    {
        int shard_elements = g.M * g.K_local;
        int cur_rank = g.iris_ctx.cur_rank();
        uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

        int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x;
        int global_stride = total_blocks * NUM_THREADS;

        for (int r = 0; r < g.world_size; r++) {
            uintptr_t base_r = g.iris_ctx.get_heap_base(r);
            intptr_t delta = (intptr_t)base_r - (intptr_t)local_base;
            const bf16* remote_a = reinterpret_cast<const bf16*>(
                (uintptr_t)g.a_shard.raw_ptr + delta);
            bf16* dst = g.a_local.raw_ptr + r * shard_elements;

            const int4* src4 = reinterpret_cast<const int4*>(remote_a);
            int4* dst4 = reinterpret_cast<int4*>(dst);
            int num_vec = shard_elements / 8;

            for (int i = global_tid; i < num_vec; i += global_stride) {
                dst4[i] = src4[i];
            }

            __threadfence();

            if (threadIdx.x == 0) {
                atomicAdd(&counters[r], 1);
            }
        }
    }

    // ════════════════════════════════════════════════════════
    // Phase 2: Persistent GEMM — grab tiles from work queue
    // ════════════════════════════════════════════════════════

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    int K_local_tiles = g.K_local / BLOCK_SIZE;
    int shard_elements = g.M * g.K_local;

    auto a_gl = g.a_local;

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a_gl, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    volatile int* vol_counters = counters;

    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int WGM = 4;

    // ── Persistent tile loop ──
    while (true) {
        // Thread 0 grabs next tile, broadcasts via shared memory
        if (threadIdx.x == 0) {
            *sh_tile_id = atomicAdd(work_ctr, 1);
        }
        __syncthreads();
        int tile_id = *sh_tile_id;
        __syncthreads();  // prevent sh_tile_id overwrite race on next iteration

        if (tile_id >= g.num_output_tiles) break;

        // Map tile_id → (row, col) with swizzle
        int wgid = tile_id;
        const int NUM_WGS = g.num_output_tiles;
        wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
        const int num_wgid_in_group = WGM * num_pid_n;
        int group_id = wgid / num_wgid_in_group;
        int first_pid_m = group_id * WGM;
        int group_size_m = min(num_pid_m - first_pid_m, WGM);
        int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
        int pid_n = (wgid % num_wgid_in_group) / group_size_m;
        int row = pid_m * M_BLOCK;
        int col = pid_n * N_BLOCK;

        // Zero accumulators for this tile
        rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];
        if (is_consumer) {
            zero(C_accum[0][0]);
            zero(C_accum[0][1]);
            zero(C_accum[1][0]);
            zero(C_accum[1][1]);
        }

        // Process all ranks for this output tile
        for (int rank = 0; rank < g.world_size; rank++) {
            // Spin until this rank's data is fully copied
            if (threadIdx.x == 0) {
                spin_until_ready(&vol_counters[rank], total_blocks);
            }
            __syncthreads();

            // Point a_gl to this rank's section
            a_gl.raw_ptr = g.a_local.raw_ptr + rank * shard_elements;

            int tic = 0, toc = 1;

            // Prefetch first tile
            if (is_producer) {
                int global_k0 = rank * K_local_tiles;
                #pragma unroll
                for (int m = 0; m < M_BLOCK; m++) {
                    G::load<2, false>(As[0][m][0], a_gl, {0, 0, row*2 + 2*m + 0, 0}, swizzled_offsets_A);
                    G::load<2, false>(As[0][m][1], a_gl, {0, 0, row*2 + 2*m + 1, 0}, swizzled_offsets_A);
                }
                #pragma unroll
                for (int n = 0; n < N_BLOCK; n++) {
                    G::load<2, false>(Bs[0][n][0], g.b, {0, 0, col*2 + 2*n + 0, global_k0}, swizzled_offsets_B);
                    G::load<2, false>(Bs[0][n][1], g.b, {0, 0, col*2 + 2*n + 1, global_k0}, swizzled_offsets_B);
                }
                __builtin_amdgcn_s_waitcnt(0);
            }
            __syncthreads();

            // K-tile loop
            for (int tile = 0; tile < K_local_tiles - 1; tile++, tic ^= 1, toc ^= 1) {
                int next_local_k = tile + 1;
                int next_global_k = rank * K_local_tiles + next_local_k;

                if (is_producer) {
                    #pragma unroll
                    for (int m = 0; m < M_BLOCK; m++) {
                        G::load<2, false>(As[toc][m][0], a_gl, {0, 0, row*2 + 2*m + 0, next_local_k}, swizzled_offsets_A);
                        G::load<2, false>(As[toc][m][1], a_gl, {0, 0, row*2 + 2*m + 1, next_local_k}, swizzled_offsets_A);
                    }
                    #pragma unroll
                    for (int n = 0; n < N_BLOCK; n++) {
                        G::load<2, false>(Bs[toc][n][0], g.b, {0, 0, col*2 + 2*n + 0, next_global_k}, swizzled_offsets_B);
                        G::load<2, false>(Bs[toc][n][1], g.b, {0, 0, col*2 + 2*n + 1, next_global_k}, swizzled_offsets_B);
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

            // Last tile of this rank
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
        }

        // Store C for this tile — single write
        if (is_consumer) {
            store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
            store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
            store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
            store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
        }
    }
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_persistent,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Query max occupancy to determine persistent grid size
    int max_blocks_per_cu;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_cu, (void*)ag_gemm_persistent, NUM_THREADS, mem_size);

    int device_id;
    hipGetDevice(&device_id);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);
    int num_cus = props.multiProcessorCount;

    int total_blocks = max_blocks_per_cu * num_cus;

    // Zero counters + work counter
    hipMemsetAsync(g.counters(), 0, g.world_size * sizeof(int), g.stream);
    hipMemsetAsync(g.work_counter(), 0, sizeof(int), g.stream);

    // Single persistent launch — all blocks copy, then all blocks compute
    ag_gemm_persistent<<<total_blocks, NUM_THREADS, mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module — persistent fused all-gather GEMM";
    py::bind_function<dispatch_ag_gemm>(m, "dispatch_ag_gemm",
        &ag_globals::a_shard,
        &ag_globals::a_local,
        &ag_globals::b,
        &ag_globals::c,
        &ag_globals::iris_ctx,
        &ag_globals::M,
        &ag_globals::N,
        &ag_globals::K,
        &ag_globals::K_local,
        &ag_globals::world_size,
        &ag_globals::counters_ptr,
        &ag_globals::work_counter_ptr,
        &ag_globals::num_output_tiles
    );
}
