"""Benchmark: AG-GEMM phases (copy vs GEMM) and comparison vs torch"""
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
    print(f"AG-GEMM Phase Breakdown + Comparison")
    print(f"Device: {torch.cuda.get_device_name()}, World size: {world_size}")
    print(f"Warmup: {WARMUP}, Measured: {ITERS}")
    print("="*90)

scale = 10.0

for M, K, N in configs:
    K_local = K // world_size
    flops = 2.0 * M * N * K
    num_output_tiles = (M // 128) * (N // 256)

    A_shard_iris = make_iris_tensor(iris, [M, K_local], dtype="bfloat16")
    A_local = torch.empty(world_size * M, K_local, dtype=torch.bfloat16, device='cuda')
    counters = torch.zeros(world_size, dtype=torch.int32, device='cuda')
    work_counter = torch.zeros(1, dtype=torch.int32, device='cuda')
    B_iris = torch.empty(N, K, dtype=torch.bfloat16, device='cuda')
    C_iris = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    torch.manual_seed(rank)
    A_shard_iris.copy_(torch.randn(M, K_local, dtype=torch.bfloat16, device='cuda') / scale)
    torch.manual_seed(42)
    B_iris.copy_(torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / scale)
    C_iris.zero_()

    A_shard_torch = A_shard_iris.clone()
    B_torch = B_iris.clone()
    C_torch = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    iris.barrier()
    dist.barrier()

    iris_device_ctx = iris.get_device_view()
    counters_ptr = counters.data_ptr()
    work_ptr = work_counter.data_ptr()

    # ── Time copy only ──
    for _ in range(WARMUP):
        tk_kernel.dispatch_copy_only(A_shard_iris, A_local, B_iris, C_iris,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)
    torch.cuda.synchronize()
    iris.barrier()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        tk_kernel.dispatch_copy_only(A_shard_iris, A_local, B_iris, C_iris,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)
    e.record()
    torch.cuda.synchronize()
    iris.barrier()
    copy_ms = s.elapsed_time(e) / ITERS

    # ── Time GEMM only (data already staged) ──
    # Stage data once
    tk_kernel.dispatch_copy_only(A_shard_iris, A_local, B_iris, C_iris,
                                 iris_device_ctx, M, N, K, K_local, world_size,
                                 counters_ptr, work_ptr, num_output_tiles)
    torch.cuda.synchronize()
    iris.barrier()

    for _ in range(WARMUP):
        tk_kernel.dispatch_gemm_only(A_shard_iris, A_local, B_iris, C_iris,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)
    torch.cuda.synchronize()

    s2 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(ITERS):
        tk_kernel.dispatch_gemm_only(A_shard_iris, A_local, B_iris, C_iris,
                                     iris_device_ctx, M, N, K, K_local, world_size,
                                     counters_ptr, work_ptr, num_output_tiles)
    e2.record()
    torch.cuda.synchronize()
    gemm_ms = s2.elapsed_time(e2) / ITERS

    # ── Time full (copy + GEMM) ──
    for _ in range(WARMUP):
        tk_kernel.dispatch_ag_gemm(A_shard_iris, A_local, B_iris, C_iris,
                                   iris_device_ctx, M, N, K, K_local, world_size,
                                   counters_ptr, work_ptr, num_output_tiles)
    torch.cuda.synchronize()
    iris.barrier()

    s3 = torch.cuda.Event(enable_timing=True)
    e3 = torch.cuda.Event(enable_timing=True)
    s3.record()
    for _ in range(ITERS):
        tk_kernel.dispatch_ag_gemm(A_shard_iris, A_local, B_iris, C_iris,
                                   iris_device_ctx, M, N, K, K_local, world_size,
                                   counters_ptr, work_ptr, num_output_tiles)
    e3.record()
    torch.cuda.synchronize()
    iris.barrier()
    full_ms = s3.elapsed_time(e3) / ITERS

    # ── Torch baseline: AG + matmul ──
    A_shards_list = [torch.empty_like(A_shard_torch) for _ in range(world_size)]

    for _ in range(WARMUP):
        dist.all_gather(A_shards_list, A_shard_torch)
        A_full_torch = torch.cat(A_shards_list, dim=1)
        C_torch = torch.matmul(A_full_torch, B_torch.t())
    torch.cuda.synchronize()
    dist.barrier()

    s4 = torch.cuda.Event(enable_timing=True)
    e4 = torch.cuda.Event(enable_timing=True)
    s4.record()
    for _ in range(ITERS):
        dist.all_gather(A_shards_list, A_shard_torch)
        A_full_torch = torch.cat(A_shards_list, dim=1)
        C_torch = torch.matmul(A_full_torch, B_torch.t())
    e4.record()
    torch.cuda.synchronize()
    dist.barrier()
    torch_ms = s4.elapsed_time(e4) / ITERS

    # ── Torch matmul only (no AG) — how fast is rocBLAS on full K? ──
    A_full_local = torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / scale
    for _ in range(WARMUP):
        C_torch = torch.matmul(A_full_local, B_torch.t())
    torch.cuda.synchronize()

    s5 = torch.cuda.Event(enable_timing=True)
    e5 = torch.cuda.Event(enable_timing=True)
    s5.record()
    for _ in range(ITERS):
        C_torch = torch.matmul(A_full_local, B_torch.t())
    e5.record()
    torch.cuda.synchronize()
    rocblas_ms = s5.elapsed_time(e5) / ITERS

    gemm_tflops = flops / (gemm_ms * 1e-3) / 1e12
    full_tflops = flops / (full_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12
    rocblas_tflops = flops / (rocblas_ms * 1e-3) / 1e12
    copy_bytes = world_size * M * K_local * 2  # bf16 = 2 bytes
    copy_bw = copy_bytes / (copy_ms * 1e-3) / 1e9  # GB/s

    if rank == 0:
        print(f"\n  Shape: {M} x {K} x {N}  (K_local={K_local})")
        print(f"  Copy bytes: {copy_bytes / 1e6:.1f} MB ({world_size} shards × {M}×{K_local}×2)")
        print()
        print(f"  {'Phase':<25s}  {'Time (ms)':>10s}  {'TFLOPS':>8s}  {'BW (GB/s)':>10s}")
        print(f"  {'-'*60}")
        print(f"  {'Copy (iris XGMI)':<25s}  {copy_ms:10.3f}  {'':>8s}  {copy_bw:10.1f}")
        print(f"  {'GEMM only (TK)':<25s}  {gemm_ms:10.3f}  {gemm_tflops:8.1f}  {'':>10s}")
        print(f"  {'Full (copy+GEMM)':<25s}  {full_ms:10.3f}  {full_tflops:8.1f}  {'':>10s}")
        print(f"  {'-'*60}")
        print(f"  {'torch AG+matmul':<25s}  {torch_ms:10.3f}  {torch_tflops:8.1f}  {'':>10s}")
        print(f"  {'rocBLAS matmul only':<25s}  {rocblas_ms:10.3f}  {rocblas_tflops:8.1f}  {'':>10s}")
        print()
        print(f"  GEMM efficiency: TK={gemm_tflops:.0f} vs rocBLAS={rocblas_tflops:.0f} TFLOPS "
              f"({gemm_tflops/rocblas_tflops*100:.0f}%)")
        print(f"  Comm overhead: {copy_ms:.3f}ms ({copy_ms/full_ms*100:.0f}% of full)")

    del A_shard_iris, A_local, counters, work_counter, B_iris, C_iris, iris_device_ctx
    del A_shard_torch, B_torch, C_torch, A_shards_list, A_full_local
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
