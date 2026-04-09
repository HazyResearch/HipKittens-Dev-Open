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

// Two-kernel AG-GEMM:
// 1. ag_copy_kernel: all CUs copy remote A shards → local HBM (max XGMI bandwidth)
// 2. ag_gemm_kernel: standard grid GEMM on pre-staged data, accumulates all ranks
//    Single accumulator per output tile → one C store instead of load-accumulate-store per rank

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
    uintptr_t counters_ptr;      // unused (kept for API compat)
    uintptr_t work_counter_ptr;  // unused (kept for API compat)
    int num_output_tiles;        // total GEMM tiles

    hipStream_t stream;

    __host__ __device__ int* counters() { return reinterpret_cast<int*>(counters_ptr); }
    __host__ __device__ int* work_counter() { return reinterpret_cast<int*>(work_counter_ptr); }

    dim3 grid() { return dim3(num_output_tiles); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 98304; }
};

// ============================================================================
// Kernel 1: Copy all remote A shards → a_local using ALL CUs
// ============================================================================
#define COPY_THREADS 1024

__global__ __launch_bounds__(COPY_THREADS, 1)
void ag_copy_kernel(const bf16* __restrict__ a_shard_ptr,
                    bf16* __restrict__ a_local_ptr,
                    iris::iris_device_view iris_ctx,
                    int shard_elements, int world_size) {
    int cur_rank = iris_ctx.cur_rank();
    uintptr_t local_base = iris_ctx.get_heap_base(cur_rank);

    int global_tid = blockIdx.x * COPY_THREADS + threadIdx.x;
    int global_stride = gridDim.x * COPY_THREADS;

    // Ring-ordered iteration: step 0 = local, step s = (cur+s)%W
    // At each step all GPUs read from unique sources → no XGMI contention
    for (int step = 0; step < world_size; step++) {
        int src_rank = (cur_rank + step) % world_size;
        uintptr_t base_r = iris_ctx.get_heap_base(src_rank);
        intptr_t delta = (intptr_t)base_r - (intptr_t)local_base;
        const int4* src4 = reinterpret_cast<const int4*>(
            (uintptr_t)a_shard_ptr + delta);
        int4* dst4 = reinterpret_cast<int4*>(a_local_ptr + src_rank * shard_elements);
        int num_vec = shard_elements / 8;

        for (int i = global_tid; i < num_vec; i += global_stride) {
            dst4[i] = src4[i];
        }
    }
}

// ============================================================================
// Kernel 2: Standard grid GEMM on pre-staged data
// Each block handles one output tile, accumulates across all ranks
// ============================================================================
__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_kernel(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    int K_local_tiles = g.K_local / BLOCK_SIZE;
    int shard_elements = g.M * g.K_local;

    auto a_gl = g.a_local;

    // Map block → output tile with swizzle
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int WGM = 4;

    int wgid = blockIdx.x;
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

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a_gl, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    // Zero accumulators — single set for entire computation
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];
    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }

    // Process all ranks for this output tile
    for (int rank = 0; rank < g.world_size; rank++) {
        // Point a_gl to this rank's section (data already in a_local)
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

    // Store C — single write for this tile (accumulated across all ranks)
    if (is_consumer) {
        store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
        store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
    }
}

// ============================================================================
// Kernel 3: Fused AG-GEMM — ring-ordered iris reads directly in GEMM producers
// No copy kernel needed. Producers read A shards from remote ranks via XGMI
// using iris pointer translation. Ring ordering prevents link contention.
// ============================================================================
__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_fused_gemm_kernel(ag_globals g) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    int K_local_tiles = g.K_local / BLOCK_SIZE;

    // Iris pointer translation setup
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    // a_gl will be repointed to each remote rank's A shard
    auto a_gl = g.a_shard;

    // Map block → output tile with swizzle
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int WGM = 4;

    int wgid = blockIdx.x;
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

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a_gl, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    // Zero accumulators — single set for entire computation
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];
    if (is_consumer) {
        zero(C_accum[0][0]);
        zero(C_accum[0][1]);
        zero(C_accum[1][0]);
        zero(C_accum[1][1]);
    }

    // Ring-ordered iteration with block stagger:
    // Different blocks start reading from different ranks to spread XGMI load.
    // Block b starts at rank (cur_rank + b) % W, then proceeds in ring order.
    // This ensures at any step, blocks are spread across all source ranks.
    int block_offset = blockIdx.x % g.world_size;
    for (int step = 0; step < g.world_size; step++) {
        int src_rank = (cur_rank + step + block_offset) % g.world_size;

        // Translate pointer to src_rank's A shard on iris symmetric heap
        uintptr_t src_base = g.iris_ctx.get_heap_base(src_rank);
        intptr_t delta = (intptr_t)src_base - (intptr_t)local_base;
        a_gl.raw_ptr = (bf16*)((uintptr_t)g.a_shard.raw_ptr + delta);

        int tic = 0, toc = 1;

        // Prefetch first tile
        if (is_producer) {
            int global_k0 = src_rank * K_local_tiles;
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
            int next_global_k = src_rank * K_local_tiles + next_local_k;

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

    // Store C — single write for this tile (accumulated across all ranks)
    if (is_consumer) {
        store(g.c, C_accum[0][0], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[0][1], {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
        store(g.c, C_accum[1][0], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
        store(g.c, C_accum[1][1], {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
    }
}

void dispatch_ag_gemm(ag_globals g) {
    int shard_elements = g.M * g.K_local;

    // ── Phase 1: Copy kernel (lean, no register spills) ──
    {
        int device_id;
        hipGetDevice(&device_id);
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, device_id);
        int num_cus = props.multiProcessorCount;

        int copy_blocks = num_cus;
        ag_copy_kernel<<<copy_blocks, COPY_THREADS, 0, g.stream>>>(
            g.a_shard.raw_ptr, g.a_local.raw_ptr,
            g.iris_ctx, shard_elements, g.world_size);
    }

    // ── Phase 2: Standard grid GEMM (stream ordering ensures copy done) ──
    {
        const unsigned long mem_size = g.dynamic_shared_memory();
        hipFuncSetAttribute((void*)ag_gemm_kernel,
                            hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        static bool printed = false;
        if (!printed) {
            fprintf(stderr, "[ag_gemm] output_tiles=%d\n", g.num_output_tiles);
            printed = true;
        }

        ag_gemm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
    }
}

// Copy only — for timing the copy phase in isolation
void dispatch_copy_only(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int device_id;
    hipGetDevice(&device_id);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);
    int copy_blocks = props.multiProcessorCount;
    ag_copy_kernel<<<copy_blocks, COPY_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size);
}

// Copy using hipMemcpyAsync (DMA engine) — ring-ordered
void dispatch_copy_memcpy(ag_globals g) {
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);
    int shard_elements = g.M * g.K_local;
    size_t shard_bytes = shard_elements * sizeof(bf16);

    for (int step = 0; step < g.world_size; step++) {
        int src_rank = (cur_rank + step) % g.world_size;
        uintptr_t src_base = g.iris_ctx.get_heap_base(src_rank);
        intptr_t delta = (intptr_t)src_base - (intptr_t)local_base;
        const void* src = (const void*)((uintptr_t)g.a_shard.raw_ptr + delta);
        void* dst = (void*)(g.a_local.raw_ptr + src_rank * shard_elements);
        hipMemcpyAsync(dst, src, shard_bytes, hipMemcpyDeviceToDevice, g.stream);
    }
}

// GEMM only — assumes data already in a_local
void dispatch_gemm_only(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_kernel,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    ag_gemm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Fused AG-GEMM — single kernel, no copy needed
// Producers read remote A shards directly via iris XGMI with ring ordering
void dispatch_fused_ag_gemm(ag_globals g) {
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_fused_gemm_kernel,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[fused_ag_gemm] output_tiles=%d, ring-ordered iris reads\n",
                g.num_output_tiles);
        printed = true;
    }

    ag_fused_gemm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// ============================================================================
// Pipelined AG-GEMM: copy rank r on stream_copy, GEMM rank r on stream_gemm
// Events synchronize: GEMM for rank r waits for copy of rank r to complete
// Copy of rank r+1 runs concurrently with GEMM of rank r
// ============================================================================

// Copy kernel for a SINGLE rank (one step of the ring)
__global__ __launch_bounds__(COPY_THREADS, 1)
void ag_copy_one_rank_kernel(const bf16* __restrict__ a_shard_ptr,
                              bf16* __restrict__ a_local_ptr,
                              intptr_t delta,
                              int shard_elements, int dst_rank) {
    int global_tid = blockIdx.x * COPY_THREADS + threadIdx.x;
    int global_stride = gridDim.x * COPY_THREADS;

    const int4* src4 = reinterpret_cast<const int4*>(
        (uintptr_t)a_shard_ptr + delta);
    int4* dst4 = reinterpret_cast<int4*>(a_local_ptr + dst_rank * shard_elements);
    int num_vec = shard_elements / 8;

    for (int i = global_tid; i < num_vec; i += global_stride) {
        dst4[i] = src4[i];
    }
}

// GEMM kernel for a SINGLE rank's K-slice (accumulates into C)
// rank_step: which rank we're processing (0 = first, zero accumulators)
// src_rank: which rank's data in a_local to read from
__global__ __launch_bounds__(NUM_THREADS, 2)
void ag_gemm_one_rank_kernel(ag_globals g, int src_rank, int rank_step) {

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using ST_A = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    using ST_B = st_bf<HALF_BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>;
    ST_A (&As)[2][M_BLOCK][2] = al.allocate<ST_A, 2, M_BLOCK, 2>();
    ST_B (&Bs)[2][N_BLOCK][2] = al.allocate<ST_B, 2, N_BLOCK, 2>();

    int warp_id = kittens::warpid();
    int local_warp_id = warp_id % 4;
    int warp_group_id = warp_id / 4;
    bool is_producer = (warp_group_id == 0);
    bool is_consumer = (warp_group_id > 0 && warp_group_id <= M_BLOCK);
    int consumer_idx = is_consumer ? warp_group_id - 1 : 0;

    int K_local_tiles = g.K_local / BLOCK_SIZE;
    int shard_elements = g.M * g.K_local;

    // Point to this rank's section in a_local
    auto a_gl = g.a_local;
    a_gl.raw_ptr = g.a_local.raw_ptr + src_rank * shard_elements;

    // Map block → output tile with swizzle
    const int num_pid_m = ceil_div(g.M, NEW_ROW_BLOCK_SIZE);
    const int num_pid_n = ceil_div(g.N, NEW_COL_BLOCK_SIZE);
    const int WGM = 4;

    int wgid = blockIdx.x;
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

    using T = typename st_bf<BLOCK_SIZE, BLOCK_SIZE, st_16x32_s>::dtype;
    constexpr int bytes_per_thread = st_16x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_PRODUCER_THREADS;
    constexpr int memcpy_per_tile = BLOCK_SIZE * BLOCK_SIZE * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_A[memcpy_per_tile];
    uint32_t swizzled_offsets_B[memcpy_per_tile];
    G::prefill_swizzled_offsets(As[0][0][0], a_gl, swizzled_offsets_A);
    G::prefill_swizzled_offsets(Bs[0][0][0], g.b, swizzled_offsets_B);

    // Accumulators: load from C if continuing, else zero
    rt_fl<HALF_BLOCK_SIZE, HALF_BLOCK_SIZE, col_l, rt_16x16_s> C_accum[2][2];
    if (is_consumer) {
        if (rank_step == 0) {
            zero(C_accum[0][0]);
            zero(C_accum[0][1]);
            zero(C_accum[1][0]);
            zero(C_accum[1][1]);
        } else {
            // Load current C accumulation
            load(C_accum[0][0], g.c, {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 0});
            load(C_accum[0][1], g.c, {0, 0, (row + consumer_idx) * 2 + 0, col * 2 + local_warp_id * 2 + 1});
            load(C_accum[1][0], g.c, {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 0});
            load(C_accum[1][1], g.c, {0, 0, (row + consumer_idx) * 2 + 1, col * 2 + local_warp_id * 2 + 1});
        }
    }

    int tic = 0, toc = 1;
    int global_k0 = src_rank * K_local_tiles;

    // Prefetch first tile
    if (is_producer) {
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

    // K-tile loop for this single rank
    for (int tile = 0; tile < K_local_tiles - 1; tile++, tic ^= 1, toc ^= 1) {
        int next_local_k = tile + 1;
        int next_global_k = src_rank * K_local_tiles + next_local_k;

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

    // Last tile
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
}

// Pipelined dispatch: overlap copy of rank r+1 with GEMM of rank r
void dispatch_pipelined_ag_gemm(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    int device_id;
    hipGetDevice(&device_id);
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);
    int copy_blocks = props.multiProcessorCount;

    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_one_rank_kernel,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    // Create two streams + events for pipelining
    hipStream_t stream_copy, stream_gemm;
    hipStreamCreate(&stream_copy);
    hipStreamCreate(&stream_gemm);

    hipEvent_t copy_done[8]; // max 8 ranks
    for (int i = 0; i < g.world_size; i++) {
        hipEventCreate(&copy_done[i]);
    }

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[pipelined_ag_gemm] output_tiles=%d, %d ranks, 2-stream pipeline\n",
                g.num_output_tiles, g.world_size);
        printed = true;
    }

    for (int step = 0; step < g.world_size; step++) {
        int src_rank = (cur_rank + step) % g.world_size;
        uintptr_t src_base = g.iris_ctx.get_heap_base(src_rank);
        intptr_t delta = (intptr_t)src_base - (intptr_t)local_base;

        // Launch copy for this rank on stream_copy
        ag_copy_one_rank_kernel<<<copy_blocks, COPY_THREADS, 0, stream_copy>>>(
            g.a_shard.raw_ptr, g.a_local.raw_ptr, delta, shard_elements, src_rank);
        hipEventRecord(copy_done[step], stream_copy);

        // GEMM stream waits for this rank's copy to finish
        hipStreamWaitEvent(stream_gemm, copy_done[step], 0);

        // Launch GEMM for this rank on stream_gemm
        ag_gemm_one_rank_kernel<<<g.grid(), g.block(), mem_size, stream_gemm>>>(
            g, src_rank, step);
    }

    // Record completion on the original stream
    hipEvent_t all_done;
    hipEventCreate(&all_done);
    hipEventRecord(all_done, stream_gemm);
    hipStreamWaitEvent(g.stream, all_done, 0);

    // Cleanup (deferred to avoid blocking)
    hipEventDestroy(all_done);
    for (int i = 0; i < g.world_size; i++) {
        hipEventDestroy(copy_done[i]);
    }
    hipStreamDestroy(stream_copy);
    hipStreamDestroy(stream_gemm);
}

// ============================================================================
// Parallel Direct All-Gather: read from ALL remote ranks simultaneously
//
// Instead of ring (1 link at a time), uses the fully-connected XGMI mesh:
// each block handles a chunk of ONE source rank's shard, and all ranks'
// blocks run concurrently → all 7 XGMI links saturated simultaneously.
//
// Grid: (world_size * blocks_per_rank) blocks
// Each block: reads from one remote rank's a_shard, writes to local a_local
// ============================================================================

#define DIRECT_AG_THREADS 512

// Pull version: each GPU reads from all remote shards simultaneously
__global__ __launch_bounds__(DIRECT_AG_THREADS, 1)
void ag_direct_pull_kernel(const bf16* __restrict__ a_shard_ptr,
                           bf16* __restrict__ a_local_ptr,
                           iris::iris_device_view iris_ctx,
                           int shard_elements, int world_size,
                           int blocks_per_rank) {
    int rank_idx = blockIdx.x / blocks_per_rank;  // which source rank
    int block_in_rank = blockIdx.x % blocks_per_rank;  // which chunk within that rank
    if (rank_idx >= world_size) return;

    int cur_rank = iris_ctx.cur_rank();
    uintptr_t local_base = iris_ctx.get_heap_base(cur_rank);
    int src_rank = (cur_rank + rank_idx) % world_size;  // ring-ordered to reduce contention

    uintptr_t src_base = iris_ctx.get_heap_base(src_rank);
    intptr_t delta = (intptr_t)src_base - (intptr_t)local_base;

    const int4* src4 = reinterpret_cast<const int4*>((uintptr_t)a_shard_ptr + delta);
    int4* dst4 = reinterpret_cast<int4*>(a_local_ptr + src_rank * shard_elements);
    int num_vec = shard_elements * (int)sizeof(bf16) / (int)sizeof(int4);

    int tid = block_in_rank * DIRECT_AG_THREADS + threadIdx.x;
    int stride = blocks_per_rank * DIRECT_AG_THREADS;

    for (int i = tid; i < num_vec; i += stride) {
        dst4[i] = src4[i];
    }
}

// Push from regular CUDA memory: reads cached shard, writes to remote iris heap
__global__ __launch_bounds__(DIRECT_AG_THREADS, 1)
void ag_direct_push_cached_kernel(const bf16* __restrict__ a_shard_cached,
                                  bf16* __restrict__ a_local_ptr,
                                  iris::iris_device_view iris_ctx,
                                  int shard_elements, int world_size,
                                  int blocks_per_rank) {
    int rank_idx = blockIdx.x / blocks_per_rank;
    int block_in_rank = blockIdx.x % blocks_per_rank;
    if (rank_idx >= world_size) return;

    int cur_rank = iris_ctx.cur_rank();
    uintptr_t local_base = iris_ctx.get_heap_base(cur_rank);
    int num_vec = shard_elements * (int)sizeof(bf16) / (int)sizeof(int4);
    const int4* src4 = reinterpret_cast<const int4*>(a_shard_cached);

    int tid = block_in_rank * DIRECT_AG_THREADS + threadIdx.x;
    int stride = blocks_per_rank * DIRECT_AG_THREADS;

    if (rank_idx == 0) {
        int4* dst4 = reinterpret_cast<int4*>(a_local_ptr + cur_rank * shard_elements);
        for (int i = tid; i < num_vec; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        int dst_rank = (cur_rank + rank_idx) % world_size;
        uintptr_t dst_base = iris_ctx.get_heap_base(dst_rank);
        intptr_t delta = (intptr_t)dst_base - (intptr_t)local_base;
        bf16* remote_a_local = (bf16*)((uintptr_t)a_local_ptr + delta);
        int4* dst4 = reinterpret_cast<int4*>(remote_a_local + cur_rank * shard_elements);
        for (int i = tid; i < num_vec; i += stride) {
            dst4[i] = src4[i];
        }
    }
}

// Push version: each GPU writes its own shard to all remote ranks simultaneously
__global__ __launch_bounds__(DIRECT_AG_THREADS, 1)
void ag_direct_push_kernel(const bf16* __restrict__ a_shard_ptr,
                           bf16* __restrict__ a_local_ptr,
                           iris::iris_device_view iris_ctx,
                           int shard_elements, int world_size,
                           int blocks_per_rank) {
    int rank_idx = blockIdx.x / blocks_per_rank;  // which dest rank
    int block_in_rank = blockIdx.x % blocks_per_rank;
    if (rank_idx >= world_size) return;

    int cur_rank = iris_ctx.cur_rank();
    uintptr_t local_base = iris_ctx.get_heap_base(cur_rank);

    // Read from own shard (local HBM), write to dest's a_local
    const int4* src4 = reinterpret_cast<const int4*>(a_shard_ptr);
    int num_vec = shard_elements * (int)sizeof(bf16) / (int)sizeof(int4);

    if (rank_idx == 0) {
        // Local: write to own a_local slot
        int4* dst4 = reinterpret_cast<int4*>(a_local_ptr + cur_rank * shard_elements);
        int tid = block_in_rank * DIRECT_AG_THREADS + threadIdx.x;
        int stride = blocks_per_rank * DIRECT_AG_THREADS;
        for (int i = tid; i < num_vec; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        // Remote: write to dest rank's a_local via XGMI
        int dst_rank = (cur_rank + rank_idx) % world_size;
        uintptr_t dst_base = iris_ctx.get_heap_base(dst_rank);
        intptr_t delta = (intptr_t)dst_base - (intptr_t)local_base;
        int4* dst4 = reinterpret_cast<int4*>(
            (uintptr_t)a_local_ptr + delta + cur_rank * shard_elements * (int)sizeof(bf16));
        // dst4 points to dst_rank's a_local[cur_rank * shard_elements]
        // Recalculate properly: a_local_ptr + delta is remote a_local base
        bf16* remote_a_local = (bf16*)((uintptr_t)a_local_ptr + delta);
        dst4 = reinterpret_cast<int4*>(remote_a_local + cur_rank * shard_elements);

        int tid = block_in_rank * DIRECT_AG_THREADS + threadIdx.x;
        int stride = blocks_per_rank * DIRECT_AG_THREADS;
        for (int i = tid; i < num_vec; i += stride) {
            dst4[i] = src4[i];
        }
    }
}

void dispatch_direct_pull_ag(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 32;  // 32 blocks per source rank, 8 ranks = 256 blocks total
    int total_blocks = g.world_size * blocks_per_rank;

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[direct_pull_ag] shard=%d elements, %d blocks/rank, %d total blocks\n",
                shard_elements, blocks_per_rank, total_blocks);
        printed = true;
    }

    ag_direct_pull_kernel<<<total_blocks, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}

void dispatch_direct_push_ag(ag_globals g) {
    int shard_elements = g.M * g.K_local;

    // Use num_output_tiles to pass blocks_per_rank from Python (default 32)
    // Actually just sweep here
    int blocks_per_rank = 32;
    int total_blocks = g.world_size * blocks_per_rank;

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[direct_push_ag] shard=%d elements, %d blocks/rank, %d total blocks\n",
                shard_elements, blocks_per_rank, total_blocks);
        printed = true;
    }

    ag_direct_push_kernel<<<total_blocks, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}

// Push from regular CUDA memory (tests if fine-grained reads are the bottleneck)
// Allocates a cached copy of a_shard on regular CUDA memory (hipMalloc), copies to it,
// then pushes from there. work_counter_ptr is repurposed as the cached buffer pointer.
void dispatch_direct_push_cached_ag(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 64;
    int total_blocks = g.world_size * blocks_per_rank;

    // Use work_counter_ptr as cached shard buffer (allocated in Python as regular CUDA tensor)
    bf16* cached_shard = reinterpret_cast<bf16*>(g.work_counter_ptr);

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[direct_push_cached_ag] shard=%d elements, reading from regular CUDA memory\n",
                shard_elements);
        printed = true;
    }

    ag_direct_push_cached_kernel<<<total_blocks, DIRECT_AG_THREADS, 0, g.stream>>>(
        cached_shard, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}

// hipMemcpyAsync-based AG: use runtime memcpy instead of kernel writes
void dispatch_memcpy_push_ag(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int shard_bytes = shard_elements * sizeof(bf16);
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[memcpy_push_ag] shard=%d bytes, using hipMemcpyAsync to all peers\n",
                shard_bytes);
        printed = true;
    }

    for (int r = 0; r < g.world_size; r++) {
        bf16* dst;
        if (r == cur_rank) {
            dst = g.a_local.raw_ptr + cur_rank * shard_elements;
        } else {
            intptr_t delta = (intptr_t)g.iris_ctx.get_heap_base(r) - (intptr_t)local_base;
            bf16* remote_a_local = (bf16*)((uintptr_t)g.a_local.raw_ptr + delta);
            dst = remote_a_local + cur_rank * shard_elements;
        }
        hipMemcpyAsync(dst, g.a_shard.raw_ptr, shard_bytes,
                       hipMemcpyDeviceToDevice, g.stream);
    }
}

// Variants with different blocks_per_rank
void dispatch_direct_push_ag_8(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 8;
    ag_direct_push_kernel<<<g.world_size * blocks_per_rank, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}
void dispatch_direct_push_ag_16(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 16;
    ag_direct_push_kernel<<<g.world_size * blocks_per_rank, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}
void dispatch_direct_push_ag_64(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 64;
    ag_direct_push_kernel<<<g.world_size * blocks_per_rank, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}
void dispatch_direct_push_ag_128(ag_globals g) {
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 128;
    ag_direct_push_kernel<<<g.world_size * blocks_per_rank, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);
}

// ============================================================================
// RCCL-exact Ring All-Gather
//
// Matches RCCL's architecture:
// - 16 channels (blocks), 512 threads (8 warps of 64) per block
// - Thread role specialization: thread 0 = wait/poll, last thread = post/signal
//   Middle threads = data copy workers
// - skip_fence path: s_waitcnt vmcnt(0) + intra-block barrier (no threadfence)
// - Counter: __atomic_store_n / __atomic_load_n with RELAXED on gfx9
// - DirectWrite: sender writes directly into receiver's output buffer
// - 16-byte (int4) vector copies
// ============================================================================

#define AG_NTHREADS 512
#define AG_NCHANNELS 16

// Counter helpers (RCCL's exact pattern on gfx9)
__device__ __forceinline__ void ag_st_flag(uint64_t* ptr, uint64_t val) {
    __atomic_store_n(ptr, val, __ATOMIC_RELAXED);
}

__device__ __forceinline__ uint64_t ag_ld_flag(uint64_t* ptr) {
    return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}

// RCCL skip_fence: signal fence + s_waitcnt + block barrier
// This is cheaper than __threadfence() which does cache invalidation
__device__ __forceinline__ void ag_skip_fence(int nthreads) {
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    asm volatile("s_waitcnt lgkmcnt(0) vmcnt(0)" ::: "memory");
    __syncthreads();  // intra-block barrier (RCCL uses a custom one, __syncthreads is equivalent)
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

__global__ __launch_bounds__(AG_NTHREADS, 1)
void ag_push_ring_kernel(bf16* __restrict__ a_shard_ptr,
                         bf16* __restrict__ a_local_ptr,
                         iris::iris_device_view iris_ctx,
                         int shard_elements, int world_size,
                         uint64_t* __restrict__ counters,
                         int num_channels) {
    int channel = blockIdx.x;
    if (channel >= num_channels) return;

    int cur_rank = iris_ctx.cur_rank();
    uintptr_t local_base = iris_ctx.get_heap_base(cur_rank);

    int next_rank = (cur_rank + 1) % world_size;
    int prev_rank = (cur_rank + world_size - 1) % world_size;

    intptr_t push_delta = (intptr_t)iris_ctx.get_heap_base(next_rank) - (intptr_t)local_base;
    intptr_t prev_delta = (intptr_t)iris_ctx.get_heap_base(prev_rank) - (intptr_t)local_base;

    bf16* next_a_local = (bf16*)((uintptr_t)a_local_ptr + push_delta);

    // Counter layout: counters[channel * world_size + rank]
    // my_counter lives on THIS rank's heap (sender writes here)
    // prev_counter is on PREV rank's heap (we read it remotely via XGMI)
    uint64_t* my_counter = &counters[channel * world_size + cur_rank];
    uint64_t* prev_counter = (uint64_t*)(
        (uintptr_t)&counters[channel * world_size + prev_rank] + prev_delta);

    // This channel's slice of the shard
    int ch_elems = shard_elements / num_channels;
    int ch_off_bytes = channel * ch_elems * (int)sizeof(bf16);
    int num_vec = ch_elems * (int)sizeof(bf16) / (int)sizeof(int4);

    // Thread roles (RCCL pattern):
    // Thread 0: polls prev rank's counter (RoleWaitRecv)
    // Last thread: signals completion (RolePostSend)
    // All threads: data copy workers (all participate in copy)

    for (int step = 0; step < world_size - 1; step++) {
        int src_rank = (cur_rank - step + world_size) % world_size;
        int shard_off = src_rank * shard_elements * (int)sizeof(bf16);

        // === WAIT PHASE ===
        // Step 0: no wait needed (sending own data)
        // Step 1+: wait for prev rank to have completed step-1
        if (step > 0) {
            // Only thread 0 polls (like RCCL's RoleWaitRecv)
            if (threadIdx.x == 0) {
                uint64_t cache = ag_ld_flag(prev_counter);
                while (cache < (uint64_t)step) {
                    __builtin_amdgcn_s_sleep(1);
                    cache = ag_ld_flag(prev_counter);
                }
            }
            __syncthreads();  // all threads wait for thread 0
        }

        // === COPY PHASE ===
        if (step == 0) {
            // directCopySend: own shard → local a_local + next rank's a_local
            const int4* src = reinterpret_cast<const int4*>((char*)a_shard_ptr + ch_off_bytes);
            int4* dst_local = reinterpret_cast<int4*>((char*)a_local_ptr + shard_off + ch_off_bytes);
            int4* dst_next  = reinterpret_cast<int4*>((char*)next_a_local + shard_off + ch_off_bytes);

            for (int i = threadIdx.x; i < num_vec; i += AG_NTHREADS) {
                int4 v = src[i];
                dst_local[i] = v;
                dst_next[i] = v;
            }
        } else if (step < world_size - 2) {
            // directRecvCopyDirectSend: local a_local → next rank's a_local
            const int4* src = reinterpret_cast<const int4*>((char*)a_local_ptr + shard_off + ch_off_bytes);
            int4* dst_next  = reinterpret_cast<int4*>((char*)next_a_local + shard_off + ch_off_bytes);
            for (int i = threadIdx.x; i < num_vec; i += AG_NTHREADS) {
                dst_next[i] = src[i];
            }
        }
        // Last step: directRecv — data is already in a_local from prev push

        // === POST PHASE (skip_fence + signal) ===
        ag_skip_fence(AG_NTHREADS);

        // Only the designated "post" thread signals
        if (threadIdx.x == AG_NTHREADS - 1) {
            ag_st_flag(my_counter, (uint64_t)(step + 1));
        }
    }
}

// ── Single-step ring copy kernel (for host-side ring dispatch) ──
// Copies one shard slice from local → next rank's a_local
// No sync — host manages step ordering via stream + barriers
__global__ __launch_bounds__(AG_NTHREADS, 1)
void ag_ring_step_kernel(bf16* __restrict__ src_ptr,
                         bf16* __restrict__ dst_ptr,
                         int num_elements) {
    int global_tid = blockIdx.x * AG_NTHREADS + threadIdx.x;
    int global_stride = gridDim.x * AG_NTHREADS;
    int num_vec = num_elements * (int)sizeof(bf16) / (int)sizeof(int4);
    const int4* src4 = reinterpret_cast<const int4*>(src_ptr);
    int4* dst4 = reinterpret_cast<int4*>(dst_ptr);

    for (int i = global_tid; i < num_vec; i += global_stride) {
        dst4[i] = src4[i];
    }
}

void dispatch_push_ring_ag(ag_globals g) {
    int shard_elements = g.M * g.K_local;

    uint64_t* sync_counters = reinterpret_cast<uint64_t*>(g.counters_ptr);

    int num_channels = AG_NCHANNELS;

    static bool printed = false;
    if (!printed) {
        fprintf(stderr, "[push_ring_ag] shard=%d elements, %d channels x %d threads, RCCL-style ring\n",
                shard_elements, num_channels, AG_NTHREADS);
        printed = true;
    }

    ag_push_ring_kernel<<<num_channels, AG_NTHREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size,
        sync_counters, num_channels);
}

// Per-step push ring dispatch. Call from Python with iris.barrier() between steps.
// step: 0..world_size-2
// Step 0: push own shard to local a_local + next rank's a_local
// Step 1..W-3: forward shard from local a_local to next rank's a_local
// Step W-2: no push needed (prev rank pushed directly into our a_local)
// NOTE: step is passed via num_output_tiles (overloaded) to reuse BIND_AG_GLOBALS
void dispatch_push_ring_step(ag_globals g) {
    int step = g.num_output_tiles;  // overloaded: Python passes step here
    int shard_elements = g.M * g.K_local;
    int cur_rank = g.iris_ctx.cur_rank();
    uintptr_t local_base = g.iris_ctx.get_heap_base(cur_rank);
    int next_rank = (cur_rank + 1) % g.world_size;
    intptr_t push_delta = (intptr_t)g.iris_ctx.get_heap_base(next_rank) - (intptr_t)local_base;
    bf16* next_a_local = (bf16*)((uintptr_t)g.a_local.raw_ptr + push_delta);

    int num_blocks = 256;  // Use all CUs

    int src_rank = (cur_rank - step + g.world_size) % g.world_size;

    if (step == 0) {
        // Copy own shard to local a_local slot
        ag_ring_step_kernel<<<num_blocks, AG_NTHREADS, 0, g.stream>>>(
            g.a_shard.raw_ptr,
            g.a_local.raw_ptr + src_rank * shard_elements,
            shard_elements);
        // Push own shard to next rank's a_local slot
        ag_ring_step_kernel<<<num_blocks, AG_NTHREADS, 0, g.stream>>>(
            g.a_shard.raw_ptr,
            next_a_local + src_rank * shard_elements,
            shard_elements);
    } else if (step < g.world_size - 2) {
        // Forward: read from local a_local (where prev rank pushed), push to next
        ag_ring_step_kernel<<<num_blocks, AG_NTHREADS, 0, g.stream>>>(
            g.a_local.raw_ptr + src_rank * shard_elements,
            next_a_local + src_rank * shard_elements,
            shard_elements);
    }
    // Last step (W-2): data already in our a_local from prev rank's push, no kernel needed
}

// Push ring AG + standard TK GEMM (sequential, for comparison with RCCL+TK)
void dispatch_push_ring_ag_gemm(ag_globals g) {
    dispatch_push_ring_ag(g);

    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_kernel,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    ag_gemm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

// Direct push AG (best iris AG) + TK GEMM
void dispatch_direct_push_ag_gemm(ag_globals g) {
    // AG phase: direct push from cached shard (work_counter_ptr = cached shard ptr)
    // For simplicity, just use the iris heap shard here (a_shard)
    int shard_elements = g.M * g.K_local;
    int blocks_per_rank = 64;
    int total_blocks = g.world_size * blocks_per_rank;

    ag_direct_push_kernel<<<total_blocks, DIRECT_AG_THREADS, 0, g.stream>>>(
        g.a_shard.raw_ptr, g.a_local.raw_ptr,
        g.iris_ctx, shard_elements, g.world_size, blocks_per_rank);

    // GEMM phase
    const unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)ag_gemm_kernel,
                        hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    ag_gemm_kernel<<<g.grid(), g.block(), mem_size, g.stream>>>(g);
}

#define BIND_AG_GLOBALS \
    &ag_globals::a_shard, \
    &ag_globals::a_local, \
    &ag_globals::b, \
    &ag_globals::c, \
    &ag_globals::iris_ctx, \
    &ag_globals::M, \
    &ag_globals::N, \
    &ag_globals::K, \
    &ag_globals::K_local, \
    &ag_globals::world_size, \
    &ag_globals::counters_ptr, \
    &ag_globals::work_counter_ptr, \
    &ag_globals::num_output_tiles

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module — all-gather GEMM (staged + fused)";
    py::bind_function<dispatch_ag_gemm>(m, "dispatch_ag_gemm", BIND_AG_GLOBALS);
    py::bind_function<dispatch_copy_only>(m, "dispatch_copy_only", BIND_AG_GLOBALS);
    py::bind_function<dispatch_gemm_only>(m, "dispatch_gemm_only", BIND_AG_GLOBALS);
    py::bind_function<dispatch_fused_ag_gemm>(m, "dispatch_fused_ag_gemm", BIND_AG_GLOBALS);
    py::bind_function<dispatch_pipelined_ag_gemm>(m, "dispatch_pipelined_ag_gemm", BIND_AG_GLOBALS);
    py::bind_function<dispatch_copy_memcpy>(m, "dispatch_copy_memcpy", BIND_AG_GLOBALS);
    py::bind_function<dispatch_push_ring_ag>(m, "dispatch_push_ring_ag", BIND_AG_GLOBALS);
    py::bind_function<dispatch_push_ring_ag_gemm>(m, "dispatch_push_ring_ag_gemm", BIND_AG_GLOBALS);

    // Host-side ring: per-step dispatch with iris.barrier() between steps
    // Python passes 'step' as the num_output_tiles argument (overloaded)
    py::bind_function<dispatch_push_ring_step>(m, "dispatch_push_ring_step", BIND_AG_GLOBALS);

    // Direct parallel AG (fully-connected mesh, all links simultaneously)
    py::bind_function<dispatch_direct_pull_ag>(m, "dispatch_direct_pull_ag", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag>(m, "dispatch_direct_push_ag", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_cached_ag>(m, "dispatch_direct_push_cached_ag", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag_gemm>(m, "dispatch_direct_push_ag_gemm", BIND_AG_GLOBALS);
    py::bind_function<dispatch_memcpy_push_ag>(m, "dispatch_memcpy_push_ag", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag_8>(m, "dispatch_direct_push_ag_8", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag_16>(m, "dispatch_direct_push_ag_16", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag_64>(m, "dispatch_direct_push_ag_64", BIND_AG_GLOBALS);
    py::bind_function<dispatch_direct_push_ag_128>(m, "dispatch_direct_push_ag_128", BIND_AG_GLOBALS);
}
