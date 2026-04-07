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

#define PREFETCH_THREADS 256

using G = kittens::group<NUM_PRODUCER_WORKERS>;
using A_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;
using B_slice = rt_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, row_l, rt_16x32_s>;

// Fused All-Gather GEMM with row-block-level signaling
// Prefetchers copy A row-block by row-block and signal per (rank, row_block).
// GEMM blocks only spin on their own row_block — start computing as soon as
// their 128 rows are ready, even while other row_blocks are still in flight.

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
    int num_prefetch_blocks;
    uintptr_t counters_ptr;   // int[world_size * num_row_blocks]

    hipStream_t stream;

    __host__ __device__ int* counters() { return reinterpret_cast<int*>(counters_ptr); }
    __host__ __device__ int num_row_blocks() { return ceil_div(M, NEW_ROW_BLOCK_SIZE); }

    __host__ __device__ int num_gemm_blocks() {
        return ceil_div(N, NEW_COL_BLOCK_SIZE) * ceil_div(M, NEW_ROW_BLOCK_SIZE);
    }
    dim3 grid()  { return dim3(num_prefetch_blocks + num_gemm_blocks()); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

// Spin-wait on a single counter until >= target
__device__ __forceinline__ void spin_until_ready(volatile int* counter, int target) {
    while (*counter < target) {
        // spin — volatile ensures re-read
    }
    __threadfence();  // ensure subsequent reads see data written before the counter
}

// ============================================================================
// Fused kernel: prefetcher + GEMM blocks, row-block-level signaling
// ============================================================================
__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_fused(ag_globals g) {
    int block_id = blockIdx.x;

    // ════════════════════════════════════════════════════════
    // Prefetcher blocks
    // ════════════════════════════════════════════════════════
    if (block_id < g.num_prefetch_blocks) {
        if (threadIdx.x >= PREFETCH_THREADS) return;

        int prefetch_id = block_id;
        int cur_rank = g.iris_ctx.cur_rank();
        uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);
        int shard_elements = g.M * g.K_local;
        int num_rb = g.num_row_blocks();

        // Row-block dimensions
        int rb_rows = NEW_ROW_BLOCK_SIZE;  // 128 rows per row-block
        int rb_elements = rb_rows * g.K_local;  // elements per row-block

        int global_tid = prefetch_id * PREFETCH_THREADS + threadIdx.x;
        int global_stride = g.num_prefetch_blocks * PREFETCH_THREADS;

        for (int r = 0; r < g.world_size; r++) {
            uintptr_t base_r = g.iris_ctx.get_heap_base(r);
            intptr_t delta = (intptr_t)base_r - (intptr_t)local_base;
            const bf16* remote_a = reinterpret_cast<const bf16*>(
                (uintptr_t)g.a_shard.raw_ptr + delta);
            bf16* dst_base = g.a_local.raw_ptr + r * shard_elements;

            for (int rb = 0; rb < num_rb; rb++) {
                int row_start = rb * rb_rows;
                int actual_rows = min(rb_rows, g.M - row_start);
                int actual_elements = actual_rows * g.K_local;

                // Source and dest for this row-block
                const int4* src4 = reinterpret_cast<const int4*>(remote_a + row_start * g.K_local);
                int4* dst4 = reinterpret_cast<int4*>(dst_base + row_start * g.K_local);
                int num_vec = actual_elements / 8;

                for (int i = global_tid; i < num_vec; i += global_stride) {
                    dst4[i] = src4[i];
                }

                __threadfence();

                if (threadIdx.x == 0) {
                    atomicAdd(&g.counters()[r * num_rb + rb], 1);
                }
            }
        }
        return;
    }

    // ════════════════════════════════════════════════════════
    // GEMM blocks
    // ════════════════════════════════════════════════════════
    int gemm_block_id = block_id - g.num_prefetch_blocks;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];

    // Workgroup ID mapping + swizzle
    int wgid = gemm_block_id;
    const int NUM_WGS = g.num_gemm_blocks();
    const int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM*WGM);
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
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

    int K_local_tiles = g.K_local / BLOCK_SIZE;
    int shard_elements = g.M * g.K_local;
    int num_rb = g.num_row_blocks();

    auto a_gl = g.a_local;  // working copy — raw_ptr updated per rank

    // Prefill swizzled offsets
    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a_gl, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    // Zero accumulators — stay in registers for ALL K tiles across ALL ranks
    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }

    volatile int* vol_counters = g.counters();

    // ── Process each rank ──
    for (int rank = 0; rank < g.world_size; rank++) {
        // Spin only on THIS block's row_block for this rank
        if (threadIdx.x == 0) {
            spin_until_ready(&vol_counters[rank * num_rb + pid_m], g.num_prefetch_blocks);
        }
        __syncthreads();

        // Point a_gl to this rank's section
        a_gl.raw_ptr = g.a_local.raw_ptr + rank * shard_elements;

        int tic = 0, toc = 1;

        // Prefetch first tile of this rank
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

        // K-tile loop within this rank
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

    // Store C — single write, accumulated across all ranks
    if (is_consumer) {
        store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
        store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
    }
}

void dispatch_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_fused,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Zero counters: world_size * num_row_blocks
    int num_counters = g.world_size * g.num_row_blocks();
    hipMemsetAsync(g.counters(), 0, num_counters * sizeof(int), g.stream);

    // Single fused launch
    ag_gemm_fused<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module — fused all-gather GEMM with row-block signaling";
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
        &ag_globals::num_prefetch_blocks,
        &ag_globals::counters_ptr
    );
}
