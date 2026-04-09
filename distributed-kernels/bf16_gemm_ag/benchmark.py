"""Benchmark: AG-GEMM phases, RCCL+TK fusion, and torch baseline"""
import torch
import ctypes
import time
import sys; sys.path.insert(0, "..")
import iris_py
import tk_kernel
import os

torch.manual_seed(0)

def make_iris_tensor(iris, shape, dtype="bfloat16"):
    dtype_map = {
        "bfloat16": (torch.bfloat16, torch.uint16),
        "float16": (torch.float16, None),
    }
    torch_dtype, view_via = dtype_map[dtype]
    iris_tensor = iris.empty(shape, dtype=dtype)
    ptr = iris_tensor.data_ptr()

    if view_via is not None:
        class W:
            def __init__(self):
                self.__cuda_array_interface__ = {
                    'shape': tuple(shape), 'typestr': '<u2',
                    'data': (ptr, False), 'version': 3, 'strides': None,
                }
        t = torch.as_tensor(W(), device=f'cuda:{iris.rank()}').view(torch_dtype)
    else:
        class W:
            def __init__(self):
                self.__cuda_array_interface__ = {
                    'shape': tuple(shape), 'typestr': '<f2',
                    'data': (ptr, False), 'version': 3, 'strides': None,
                }
        t = torch.as_tensor(W(), device=f'cuda:{iris.rank()}')
    t._iris_tensor = iris_tensor
    return t


iris = iris_py.Iris(heap_size_mb=1024, verbose=False)
rank = iris.rank()
world_size = iris.world_size()
torch.cuda.set_device(rank)

import torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

WARMUP = 10
ITERS = 50

configs = [
    (7680,  8192, 8192),
]

if rank == 0:
    print("="*90)
    print(f"AG-GEMM Fusion Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}, World size: {world_size}")
    print(f"Warmup: {WARMUP}, Measured: {ITERS}")
    print("="*90)

scale = 10.0

for M, K, N in configs:
    K_local = K // world_size
    flops = 2.0 * M * N * K
    num_output_tiles = (M // 128) * (N // 256)

    A_shard_iris = make_iris_tensor(iris, [M, K_local], dtype="bfloat16")
    # a_local on iris heap so push ring AG can write to remote ranks' a_local
    A_local = make_iris_tensor(iris, [world_size * M, K_local], dtype="bfloat16")
    counters = torch.zeros(world_size, dtype=torch.int32, device='cuda')
    work_counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    # Sync counters for push ring AG on iris heap (generous size for multi-channel)
    sync_counters = make_iris_tensor(iris, [4096], dtype="bfloat16")
    sync_counters_ptr = sync_counters.data_ptr()
    B = torch.empty(N, K, dtype=torch.bfloat16, device='cuda')
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    torch.manual_seed(rank)
    A_shard_iris.copy_(torch.randn(M, K_local, dtype=torch.bfloat16, device='cuda') / scale)
    torch.manual_seed(42)
    B.copy_(torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / scale)
    C.zero_()

    A_shard_torch = A_shard_iris.clone()  # regular CUDA tensor for RCCL

    iris.barrier()
    dist.barrier()

    iris_device_ctx = iris.get_device_view()
    counters_ptr = counters.data_ptr()
    work_ptr = work_counter.data_ptr()

    def call_gemm_only():
        tk_kernel.dispatch_gemm_only(A_shard_iris, A_local, B, C,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)

    def call_full():
        tk_kernel.dispatch_ag_gemm(A_shard_iris, A_local, B, C,
                                   iris_device_ctx, M, N, K, K_local, world_size,
                                   counters_ptr, work_ptr, num_output_tiles)

    def call_fused():
        tk_kernel.dispatch_fused_ag_gemm(A_shard_iris, A_local, B, C,
                                          iris_device_ctx, M, N, K, K_local, world_size,
                                          counters_ptr, work_ptr, num_output_tiles)

    def call_pipelined():
        tk_kernel.dispatch_pipelined_ag_gemm(A_shard_iris, A_local, B, C,
                                              iris_device_ctx, M, N, K, K_local, world_size,
                                              counters_ptr, work_ptr, num_output_tiles)

    def call_direct_pull_ag():
        tk_kernel.dispatch_direct_pull_ag(A_shard_iris, A_local, B, C,
                                           iris_device_ctx, M, N, K, K_local, world_size,
                                           counters_ptr, work_ptr, num_output_tiles)

    def call_direct_push_ag():
        tk_kernel.dispatch_direct_push_ag(A_shard_iris, A_local, B, C,
                                            iris_device_ctx, M, N, K, K_local, world_size,
                                            counters_ptr, work_ptr, num_output_tiles)

    # Cached push: read from regular CUDA memory (not fine-grained iris heap)
    A_shard_cached = A_shard_iris.clone()  # regular CUDA tensor
    A_shard_cached.copy_(A_shard_iris)
    cached_ptr = A_shard_cached.data_ptr()
    def call_direct_push_cached_ag():
        tk_kernel.dispatch_direct_push_cached_ag(A_shard_iris, A_local, B, C,
                                                   iris_device_ctx, M, N, K, K_local, world_size,
                                                   counters_ptr, cached_ptr, num_output_tiles)

    def call_push_ring_ag():
        sync_counters.zero_()
        tk_kernel.dispatch_push_ring_ag(A_shard_iris, A_local, B, C,
                                         iris_device_ctx, M, N, K, K_local, world_size,
                                         sync_counters_ptr, work_ptr, num_output_tiles)

    def call_host_ring_ag():
        for step in range(world_size - 1):
            # step is passed as num_output_tiles (overloaded)
            tk_kernel.dispatch_push_ring_step(A_shard_iris, A_local, B, C,
                                               iris_device_ctx, M, N, K, K_local, world_size,
                                               sync_counters_ptr, work_ptr, step)
            torch.cuda.synchronize()
            iris.barrier()

    def call_push_ring_ag_gemm():
        sync_counters.zero_()
        tk_kernel.dispatch_push_ring_ag_gemm(A_shard_iris, A_local, B, C,
                                              iris_device_ctx, M, N, K, K_local, world_size,
                                              sync_counters_ptr, work_ptr, num_output_tiles)

    def call_direct_push_ag_gemm():
        tk_kernel.dispatch_direct_push_ag_gemm(A_shard_iris, A_local, B, C,
                                                iris_device_ctx, M, N, K, K_local, world_size,
                                                counters_ptr, work_ptr, num_output_tiles)

    def call_rccl_ag():
        dist.all_gather_into_tensor(A_local, A_shard_torch)

    def call_rccl_ag_tk_gemm():
        dist.all_gather_into_tensor(A_local, A_shard_torch)
        tk_kernel.dispatch_gemm_only(A_shard_iris, A_local, B, C,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)

    def call_torch_ag_matmul():
        dist.all_gather_into_tensor(A_local, A_shard_torch)
        A_full = A_local.view(world_size, M, K_local).permute(1, 0, 2).reshape(M, K)
        return torch.matmul(A_full, B.t())

    def time_fn(fn, name, warmup=WARMUP, iters=ITERS):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        dist.barrier() if 'rccl' in name or 'torch' in name else iris.barrier()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        dist.barrier() if 'rccl' in name or 'torch' in name else iris.barrier()
        return s.elapsed_time(e) / iters

    # ── Time each phase ──
    copy_iris_ms = time_fn(lambda: tk_kernel.dispatch_copy_only(
        A_shard_iris, A_local, B, C, iris_device_ctx, M, N, K, K_local, world_size,
        counters_ptr, work_ptr, num_output_tiles), "iris_copy")

    copy_memcpy_ms = time_fn(lambda: tk_kernel.dispatch_copy_memcpy(
        A_shard_iris, A_local, B, C, iris_device_ctx, M, N, K, K_local, world_size,
        counters_ptr, work_ptr, num_output_tiles), "iris_memcpy_copy")

    # Stage data for GEMM-only timing
    tk_kernel.dispatch_copy_only(A_shard_iris, A_local, B, C,
                                 iris_device_ctx, M, N, K, K_local, world_size,
                                 counters_ptr, work_ptr, num_output_tiles)
    torch.cuda.synchronize()

    gemm_ms = time_fn(call_gemm_only, "gemm_only")
    full_iris_ms = time_fn(call_full, "iris_full")
    fused_iris_ms = time_fn(call_fused, "iris_fused")
    pipelined_iris_ms = time_fn(call_pipelined, "iris_pipelined")

    # Host-side ring AG (iris.barrier() between steps — no in-kernel sync)
    host_ring_ag_ms = time_fn(call_host_ring_ag, "iris_host_ring_ag")

    # In-kernel push ring AG
    push_ring_ag_ms = time_fn(call_push_ring_ag, "iris_push_ring_ag")

    # Direct parallel AG (all XGMI links simultaneously)
    direct_pull_ms = time_fn(call_direct_pull_ag, "iris_direct_pull")
    direct_push_ms = time_fn(call_direct_push_ag, "iris_direct_push")
    direct_push_cached_ms = time_fn(call_direct_push_cached_ag, "iris_direct_push_cached")

    # hipMemcpyAsync-based push
    def call_memcpy_push_ag():
        tk_kernel.dispatch_memcpy_push_ag(A_shard_iris, A_local, B, C,
                                           iris_device_ctx, M, N, K, K_local, world_size,
                                           counters_ptr, work_ptr, num_output_tiles)
    memcpy_push_ms = time_fn(call_memcpy_push_ag, "iris_memcpy_push")

    # Sweep blocks_per_rank for push
    def call_push_8():
        tk_kernel.dispatch_direct_push_ag_8(A_shard_iris, A_local, B, C,
                                             iris_device_ctx, M, N, K, K_local, world_size,
                                             counters_ptr, work_ptr, num_output_tiles)
    def call_push_16():
        tk_kernel.dispatch_direct_push_ag_16(A_shard_iris, A_local, B, C,
                                              iris_device_ctx, M, N, K, K_local, world_size,
                                              counters_ptr, work_ptr, num_output_tiles)
    def call_push_64():
        tk_kernel.dispatch_direct_push_ag_64(A_shard_iris, A_local, B, C,
                                              iris_device_ctx, M, N, K, K_local, world_size,
                                              counters_ptr, work_ptr, num_output_tiles)
    def call_push_128():
        tk_kernel.dispatch_direct_push_ag_128(A_shard_iris, A_local, B, C,
                                               iris_device_ctx, M, N, K, K_local, world_size,
                                               counters_ptr, work_ptr, num_output_tiles)
    push_8_ms = time_fn(call_push_8, "iris_push_8")
    push_16_ms = time_fn(call_push_16, "iris_push_16")
    push_64_ms = time_fn(call_push_64, "iris_push_64")
    push_128_ms = time_fn(call_push_128, "iris_push_128")

    # Push ring AG + TK GEMM
    push_ring_ag_gemm_ms = time_fn(call_push_ring_ag_gemm, "iris_push_ring_ag_gemm")

    # Direct push AG + TK GEMM (best combo)
    direct_push_ag_gemm_ms = time_fn(call_direct_push_ag_gemm, "iris_direct_push_gemm")

    # RCCL-based
    rccl_ag_ms = time_fn(call_rccl_ag, "rccl_ag")

    # Stage via RCCL for gemm-only
    dist.all_gather_into_tensor(A_local, A_shard_torch)
    torch.cuda.synchronize()

    gemm_after_rccl_ms = time_fn(call_gemm_only, "gemm_only2")

    # RCCL AG + TK GEMM (the fusion!)
    rccl_tk_ms = time_fn(call_rccl_ag_tk_gemm, "rccl_tk")

    # Torch baseline: RCCL AG + rocBLAS matmul
    # Use simple all_gather + matmul like before
    A_shards_list = [torch.empty_like(A_shard_torch) for _ in range(world_size)]
    def call_torch_baseline():
        dist.all_gather(A_shards_list, A_shard_torch)
        A_full = torch.cat(A_shards_list, dim=1)
        torch.matmul(A_full, B.t())
    torch_ms = time_fn(call_torch_baseline, "torch_baseline")

    # rocBLAS matmul only (full K, no comm)
    A_full_local = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / scale
    def call_rocblas():
        torch.matmul(A_full_local, B.t())
    rocblas_ms = time_fn(call_rocblas, "rocblas")

    if rank == 0:
        print(f"\n  Shape: {M} x {K} x {N}  (K_local={K_local})")
        print(f"  Copy: {world_size}×{M}×{K_local}×2 = {world_size*M*K_local*2/1e6:.1f} MB")
        print()
        print(f"  {'Approach':<35s}  {'Time (ms)':>10s}  {'TFLOPS':>8s}")
        print(f"  {'-'*58}")
        total_bytes = world_size*M*K_local*2
        copy_bw = total_bytes/1e9/(copy_iris_ms*1e-3)
        print(f"  {'Iris copy (kernel, ring)':<35s}  {copy_iris_ms:10.3f}  {copy_bw:7.0f} GB/s")
        memcpy_bw = total_bytes/1e9/(copy_memcpy_ms*1e-3)
        print(f"  {'Iris copy (hipMemcpy, ring)':<35s}  {copy_memcpy_ms:10.3f}  {memcpy_bw:7.0f} GB/s")
        rccl_bw = total_bytes/1e9/(rccl_ag_ms*1e-3)
        host_ring_bw = total_bytes/1e9/(host_ring_ag_ms*1e-3)
        push_bw = total_bytes/1e9/(push_ring_ag_ms*1e-3)
        print(f"  {'Iris host ring AG (barrier)':<35s}  {host_ring_ag_ms:10.3f}  {host_ring_bw:7.0f} GB/s")
        print(f"  {'Iris push ring AG (in-kernel)':<35s}  {push_ring_ag_ms:10.3f}  {push_bw:7.0f} GB/s")
        direct_pull_bw = total_bytes/1e9/(direct_pull_ms*1e-3)
        direct_push_bw = total_bytes/1e9/(direct_push_ms*1e-3)
        print(f"  {'Iris direct pull AG (parallel)':<35s}  {direct_pull_ms:10.3f}  {direct_pull_bw:7.0f} GB/s")
        print(f"  {'Iris direct push AG (32 blk/r)':<35s}  {direct_push_ms:10.3f}  {direct_push_bw:7.0f} GB/s")
        cached_bw = total_bytes/1e9/(direct_push_cached_ms*1e-3)
        print(f"  {'Iris push cached (64 blk/r)':<35s}  {direct_push_cached_ms:10.3f}  {cached_bw:7.0f} GB/s")
        memcpy_push_bw = total_bytes/1e9/(memcpy_push_ms*1e-3)
        print(f"  {'Iris hipMemcpyAsync push':<35s}  {memcpy_push_ms:10.3f}  {memcpy_push_bw:7.0f} GB/s")
        for bpr, ms in [(8, push_8_ms), (16, push_16_ms), (64, push_64_ms), (128, push_128_ms)]:
            bw = total_bytes/1e9/(ms*1e-3)
            print(f"  {'Iris direct push AG (%d blk/r)' % bpr:<35s}  {ms:10.3f}  {bw:7.0f} GB/s")
        print(f"  {'RCCL all_gather_into_tensor':<35s}  {rccl_ag_ms:10.3f}  {rccl_bw:7.0f} GB/s")
        print(f"  {'TK GEMM only':<35s}  {gemm_ms:10.3f}  {flops/(gemm_ms*1e-3)/1e12:8.1f}")
        print(f"  {'rocBLAS matmul only':<35s}  {rocblas_ms:10.3f}  {flops/(rocblas_ms*1e-3)/1e12:8.1f}")
        print(f"  {'-'*58}")
        print(f"  {'Iris copy + TK GEMM (staged)':<35s}  {full_iris_ms:10.3f}  {flops/(full_iris_ms*1e-3)/1e12:8.1f}")
        print(f"  {'Iris fused AG-GEMM (ring)':<35s}  {fused_iris_ms:10.3f}  {flops/(fused_iris_ms*1e-3)/1e12:8.1f}")
        print(f"  {'Iris pipelined (copy||GEMM)':<35s}  {pipelined_iris_ms:10.3f}  {flops/(pipelined_iris_ms*1e-3)/1e12:8.1f}")
        print(f"  {'Iris host ring AG + TK GEMM':<35s}  {host_ring_ag_ms + gemm_ms:10.3f}  {flops/((host_ring_ag_ms + gemm_ms)*1e-3)/1e12:8.1f}")
        print(f"  {'Iris direct push AG + TK GEMM':<35s}  {direct_push_ag_gemm_ms:10.3f}  {flops/(direct_push_ag_gemm_ms*1e-3)/1e12:8.1f}")
        print(f"  {'RCCL AG + TK GEMM':<35s}  {rccl_tk_ms:10.3f}  {flops/(rccl_tk_ms*1e-3)/1e12:8.1f}")
        print(f"  {'torch AG + rocBLAS (baseline)':<35s}  {torch_ms:10.3f}  {flops/(torch_ms*1e-3)/1e12:8.1f}")
        print(f"  {'-'*58}")
        best_iris = min(fused_iris_ms, pipelined_iris_ms, full_iris_ms)
        best_name = "fused" if best_iris == fused_iris_ms else ("pipelined" if best_iris == pipelined_iris_ms else "staged")
        print(f"\n  Best iris ({best_name}) vs torch: {torch_ms/best_iris:.2f}x")
        print(f"  Best iris ({best_name}) vs RCCL+TK: {rccl_tk_ms/best_iris:.2f}x")
        print(f"  Pipelined vs staged: {full_iris_ms/pipelined_iris_ms:.2f}x")

    del A_shard_iris, A_local, counters, work_counter, sync_counters, B, C, iris_device_ctx
    del A_shard_torch, A_shards_list, A_full_local
    import gc; gc.collect()
    torch.cuda.synchronize()
    iris.barrier()
    dist.barrier()

if rank == 0:
    print("="*90)

dist.destroy_process_group()
iris.barrier()
del iris
import gc; gc.collect()
torch.cuda.synchronize()
from mpi4py import MPI
MPI.Finalize()
os._exit(0)
