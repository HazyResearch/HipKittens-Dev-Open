import torch
import ctypes
import sys; sys.path.insert(0, "..")
import iris_py
import tk_kernel

torch.manual_seed(0)

def make_iris_tensor(iris, shape, dtype="bfloat16"):
    """Create a PyTorch tensor backed by iris fine-grained memory."""
    dtype_map = {
        "bfloat16": (torch.bfloat16, torch.uint16),
        "bf16": (torch.bfloat16, torch.uint16),
        "float32": (torch.float32, None),
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
        typestr_map = {torch.float32: '<f4', torch.float16: '<f2'}
        class W:
            def __init__(self):
                self.__cuda_array_interface__ = {
                    'shape': tuple(shape), 'typestr': typestr_map[torch_dtype],
                    'data': (ptr, False), 'version': 3, 'strides': None,
                }
        t = torch.as_tensor(W(), device=f'cuda:{iris.rank()}')
    t._iris_tensor = iris_tensor
    return t


# Setup
M = 192 * 40  # 7680
K = 8192
N = 8192
scale = 10.0

iris = iris_py.Iris(heap_size_mb=1024, verbose=False)
rank = iris.rank()
world_size = iris.world_size()
torch.cuda.set_device(rank)

K_local = K // world_size

if rank == 0:
    print("="*60)
    print(f"All-Gather GEMM (K-sharded, staged): C[{M},{N}] = sum_r(A_shard_r[{M},{K_local}] @ B[{N},{K}]^T)")
    print(f"A K-sharded: A_shard[{M},{K_local}] per rank, {world_size} ranks")
    print(f"Using local staging buffer for XGMI→HBM copy")
    print("="*60)

# Allocate A_shard on iris heap (remote-accessible)
if rank == 0:
    print("\n[Allocating Tensors]")
A_shard = make_iris_tensor(iris, [M, K_local], dtype="bfloat16")

# Double-buffered staging: local HBM (not on iris heap), same shape as A_shard
A_staging = torch.empty(M, K_local, dtype=torch.bfloat16, device='cuda')
A_staging2 = torch.empty(M, K_local, dtype=torch.bfloat16, device='cuda')

# B and C are local — regular CUDA memory
B = torch.empty(N, K, dtype=torch.bfloat16, device='cuda')
C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')

if rank == 0:
    print(f"  A_shard: {A_shard.shape} (iris heap)")
    print(f"  A_staging: {A_staging.shape} x2 (double-buffered local HBM)")
    print(f"  B: {B.shape} (local), C: {C.shape} (local)")

# Verify iris backing for A_shard
iris_ok = (A_shard.data_ptr() == A_shard._iris_tensor.data_ptr())
print(f"Rank {rank}: iris-backed={iris_ok}, A_shard=0x{A_shard.data_ptr():x}")

# Initialize — each rank gets different A_shard, same B
torch.manual_seed(rank)
A_shard.copy_(torch.randn(M, K_local, dtype=torch.bfloat16, device='cuda') / scale)
torch.manual_seed(42)
B.copy_(torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / scale)
C.zero_()
iris.barrier()

# Compute reference: reconstruct full A, matmul
if rank == 0:
    print("\n[Computing Reference (all-gather A + matmul)]")
A_full_list = []
for r in range(world_size):
    torch.manual_seed(r)
    A_full_list.append(torch.randn(M, K_local, dtype=torch.bfloat16, device='cuda') / scale)
A_full = torch.cat(A_full_list, dim=1)  # [M, K]
C_ref = torch.matmul(A_full, B.t())     # [M, N]
if rank == 0:
    print(f"  A_full: {A_full.shape}, C_ref: {C_ref.shape}")

# Run AG-GEMM kernel (staged)
if rank == 0:
    print("\n[Running AG-GEMM Kernel (staged, K-sharded)]")
iris_device_ctx = iris.get_device_view()
iris.barrier()

tk_kernel.dispatch_ag_gemm(A_shard, A_staging, A_staging2, B, C, iris_device_ctx,
                           M, N, K, K_local, world_size)
torch.cuda.synchronize()
iris.barrier()

# Validate
if rank == 0:
    print("\n[Validating Results]")
diff = (C.float() - C_ref.float()).abs()
max_error = diff.max().item()
mean_error = diff.mean().item()
status = "PASSED" if max_error < 0.5 else "FAILED"
print(f"Rank {rank}: max_error={max_error:.4f}, mean_error={mean_error:.6f}, {status}")

if rank == 0:
    print("\n" + "="*60)
    print(f"Result: {status}")
    print("="*60)

# Cleanup
import gc
from mpi4py import MPI
del A_shard, A_staging, A_staging2, B, C, A_full, C_ref, A_full_list
gc.collect()
torch.cuda.synchronize()
iris.barrier()
del iris_device_ctx
del iris
gc.collect()
torch.cuda.synchronize()
MPI.Finalize()
import os
os._exit(0)
