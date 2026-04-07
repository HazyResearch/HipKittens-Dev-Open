"""Benchmark: fused AG-GEMM kernel vs torch.distributed.all_gather + torch.matmul"""
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


# Setup
iris = iris_py.Iris(heap_size_mb=1024, verbose=False)
rank = iris.rank()
world_size = iris.world_size()
torch.cuda.set_device(rank)

# Also init torch.distributed for the baseline
import torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

WARMUP = 10
ITERS = 50
NUM_PREFETCH_BLOCKS = 32

configs = [
    # (M,    K,    N)
    (7680,  8192, 8192),
]

if rank == 0:
    print("="*80)
    print(f"Fused AG-GEMM Benchmark: HipKittens (prefetch+spin) vs torch.distributed.all_gather + matmul")
    print(f"Device: {torch.cuda.get_device_name()}, World size: {world_size}")
    print(f"Prefetcher blocks: {NUM_PREFETCH_BLOCKS}")
    print(f"Warmup: {WARMUP}, Measured: {ITERS}")
    print("="*80)
    print(f"{'M':>6s} x {'K':>5s} x {'N':>5s}  {'K_local':>7s}  "
          f"{'TK (ms)':>9s}  {'Torch (ms)':>10s}  {'Speedup':>7s}  {'TK TFLOPS':>10s}  {'Torch TFLOPS':>13s}")
    print("-"*80)

scale = 10.0

for M, K, N in configs:
    K_local = K // world_size
    flops = 2.0 * M * N * K  # total FLOPS for GEMM

    # ── Allocate iris tensors ──
    A_shard_iris = make_iris_tensor(iris, [M, K_local], dtype="bfloat16")
    A_local = torch.empty(world_size * M, K_local, dtype=torch.bfloat16, device='cuda')
    ROWS_PER_CHUNK = 8
    num_row_blocks = M // 128
    num_chunks = (num_row_blocks + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
    counters = torch.zeros(world_size * num_chunks, dtype=torch.int32, device='cuda')
    B_iris = torch.empty(N, K, dtype=torch.bfloat16, device='cuda')
    C_iris = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    torch.manual_seed(rank)
    A_shard_iris.copy_(torch.randn(M, K_local, dtype=torch.bfloat16, device='cuda') / scale)
    torch.manual_seed(42)
    B_iris.copy_(torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / scale)
    C_iris.zero_()

    # ── Allocate torch tensors (regular CUDA memory) ──
    A_shard_torch = A_shard_iris.clone()
    B_torch = B_iris.clone()
    C_torch = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

    iris.barrier()
    dist.barrier()

    counters_ptr = counters.data_ptr()

    # ── Benchmark: HipKittens fused AG-GEMM ──
    iris_device_ctx = iris.get_device_view()

    for _ in range(WARMUP):
        tk_kernel.dispatch_ag_gemm(A_shard_iris, A_local, B_iris, C_iris,
                                   iris_device_ctx, M, N, K, K_local, world_size,
                                   NUM_PREFETCH_BLOCKS, counters_ptr)
    torch.cuda.synchronize()
    iris.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        tk_kernel.dispatch_ag_gemm(A_shard_iris, A_local, B_iris, C_iris,
                                   iris_device_ctx, M, N, K, K_local, world_size,
                                   NUM_PREFETCH_BLOCKS, counters_ptr)
    end.record()
    torch.cuda.synchronize()
    iris.barrier()
    tk_ms = start.elapsed_time(end) / ITERS

    # ── Benchmark: torch.distributed.all_gather + matmul ──
    A_shards_list = [torch.empty_like(A_shard_torch) for _ in range(world_size)]

    for _ in range(WARMUP):
        dist.all_gather(A_shards_list, A_shard_torch)
        A_full_torch = torch.cat(A_shards_list, dim=1)
        C_torch = torch.matmul(A_full_torch, B_torch.t())
    torch.cuda.synchronize()
    dist.barrier()

    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True)
    start2.record()
    for _ in range(ITERS):
        dist.all_gather(A_shards_list, A_shard_torch)
        A_full_torch = torch.cat(A_shards_list, dim=1)
        C_torch = torch.matmul(A_full_torch, B_torch.t())
    end2.record()
    torch.cuda.synchronize()
    dist.barrier()
    torch_ms = start2.elapsed_time(end2) / ITERS

    speedup = torch_ms / tk_ms
    tk_tflops = flops / (tk_ms * 1e-3) / 1e12
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

    if rank == 0:
        print(f"{M:6d} x {K:5d} x {N:5d}  {K_local:7d}  "
              f"{tk_ms:9.3f}  {torch_ms:10.3f}  {speedup:6.2f}x  {tk_tflops:10.2f}  {torch_tflops:13.2f}")

    # Cleanup per-config
    del A_shard_iris, A_local, counters, B_iris, C_iris, iris_device_ctx
    del A_shard_torch, B_torch, C_torch, A_shards_list
    import gc; gc.collect()
    torch.cuda.synchronize()
    iris.barrier()
    dist.barrier()

if rank == 0:
    print("="*80)

# Cleanup
dist.destroy_process_group()
iris.barrier()
del iris
import gc; gc.collect()
torch.cuda.synchronize()
from mpi4py import MPI
MPI.Finalize()
os._exit(0)
