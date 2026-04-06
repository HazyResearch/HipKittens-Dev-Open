import torch
import ctypes
import sys; sys.path.insert(0, "..")
import iris_py
import tk_kernel

torch.manual_seed(0)

def make_iris_tensor(iris, shape, dtype="bfloat16"):
    """Create a PyTorch tensor backed by iris fine-grained memory.

    Allocates on the iris symmetric heap and creates a zero-copy
    torch tensor via torch.from_blob (ROCm/HIP treats iris memory
    as a valid device pointer).
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[dtype]

    iris_tensor = iris.empty(shape, dtype=dtype)
    ptr = iris_tensor.data_ptr()
    numel = 1
    for s in shape:
        numel *= s

    # Create a torch tensor that directly wraps the iris device pointer
    # torch.from_blob works with device pointers on ROCm
    t = torch.from_blob(ctypes.c_void_p(ptr), shape, dtype=torch_dtype, device='cuda')
    # Prevent iris_tensor from being garbage collected
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

N_local = N // world_size

if rank == 0:
    print("="*60)
    print(f"All-Gather GEMM: C[{M},{N}] = A[{M},{K}] @ B_full[{N},{K}]^T")
    print(f"B column-sharded: B_shard[{N_local},{K}] per rank, {world_size} ranks")
    print("="*60)

# Allocate tensors on iris heap
if rank == 0:
    print("\n[Allocating Iris Tensors]")
A = make_iris_tensor(iris, [M, K], dtype="bfloat16")
B_shard = make_iris_tensor(iris, [N_local, K], dtype="bfloat16")
C = make_iris_tensor(iris, [M, N], dtype="bfloat16")
if rank == 0:
    print(f"  A: {A.shape}, B_shard: {B_shard.shape}, C: {C.shape}")

# Verify iris backing
iris_ok = (A.data_ptr() == A._iris_tensor.data_ptr() and
           B_shard.data_ptr() == B_shard._iris_tensor.data_ptr() and
           C.data_ptr() == C._iris_tensor.data_ptr())
print(f"Rank {rank}: iris-backed={iris_ok}, A=0x{A.data_ptr():x}, B=0x{B_shard.data_ptr():x}, C=0x{C.data_ptr():x}")

# Initialize
torch.manual_seed(42)
A.copy_(torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / scale)
torch.manual_seed(rank)
B_shard.copy_(torch.randn(N_local, K, dtype=torch.bfloat16, device='cuda') / scale)
C.zero_()
iris.barrier()

# Compute reference
if rank == 0:
    print("\n[Computing Reference (all-gather + matmul)]")
B_full_list = []
for r in range(world_size):
    torch.manual_seed(r)
    B_full_list.append(torch.randn(N_local, K, dtype=torch.bfloat16, device='cuda') / scale)
B_full = torch.cat(B_full_list, dim=0)  # [N, K]
C_ref = torch.matmul(A, B_full.t())     # [M, N]
if rank == 0:
    print(f"  B_full: {B_full.shape}, C_ref: {C_ref.shape}")

# Run AG-GEMM kernel
if rank == 0:
    print("\n[Running AG-GEMM Kernel]")
iris_device_ctx = iris.get_device_view()
iris.barrier()

tk_kernel.dispatch_ag_gemm(A, B_shard, C, iris_device_ctx, M, N, K, N_local, world_size)
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
del A, B_shard, C, B_full, C_ref, B_full_list
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
