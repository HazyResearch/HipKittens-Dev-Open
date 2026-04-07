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
}
