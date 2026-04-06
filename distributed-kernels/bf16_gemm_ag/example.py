import torch
import sys; sys.path.insert(0, "..")
import iris_py
import tk_kernel

torch.manual_seed(0)

# Helper function to create iris tensor and wrap as PyTorch tensor
def make_iris_tensor(iris, shape, dtype="bfloat16"):
    """Create iris tensor and wrap as PyTorch tensor (HACK using __cuda_array_interface__)"""
    import ctypes

    iris_tensor = iris.empty(shape, dtype=dtype)

    # Map dtype to torch dtype and typestr
    dtype_map = {
        "bfloat16": (torch.bfloat16, "<u2"),  # bfloat16 as uint16
        "bf16": (torch.bfloat16, "<u2"),
        "float32": (torch.float32, "<f4"),
        "float": (torch.float32, "<f4"),
        "float16": (torch.float16, "<f2"),
        "half": (torch.float16, "<f2"),
    }
    torch_dtype, typestr = dtype_map[dtype]

    # HACK: Use __cuda_array_interface__ to create zero-copy tensor
    class CudaArrayWrapper:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {
                'shape': tuple(shape),
                'typestr': typestr,
                'data': (ptr, False),  # (ptr, read_only)
                'version': 3,
                'strides': None,  # C-contiguous
            }
            self._iris_tensor = iris_tensor  # Keep alive

    wrapper = CudaArrayWrapper(iris_tensor.data_ptr(), shape, typestr)

    try:
        torch_tensor = torch.as_tensor(wrapper, device='cuda').to(torch_dtype)
    except:
        try:
            torch_tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            torch_tensor._iris_tensor = iris_tensor
        except:
            torch_tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            torch_tensor._iris_tensor = iris_tensor

    return torch_tensor

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
A = make_iris_tensor(iris, [M, K], dtype="bfloat16")           # same on all ranks
B_shard = make_iris_tensor(iris, [N_local, K], dtype="bfloat16") # different per rank
C = make_iris_tensor(iris, [M, N], dtype="bfloat16")            # full output
if rank == 0:
    print(f"  A: {A.shape}, B_shard: {B_shard.shape}, C: {C.shape}")

# Initialize
# A is the same on all ranks (same seed)
torch.manual_seed(42)
A.copy_(torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / scale)
# B_shard is different per rank (seed = rank)
torch.manual_seed(rank)
B_shard.copy_(torch.randn(N_local, K, dtype=torch.bfloat16, device='cuda') / scale)
C.zero_()
iris.barrier()

# Compute reference: all-gather B shards, then matmul
if rank == 0:
    print("\n[Computing Reference (all-gather + matmul)]")

# Gather all B shards to form B_full
B_full_list = [torch.empty(N_local, K, dtype=torch.bfloat16, device='cuda') for _ in range(world_size)]
for r in range(world_size):
    torch.manual_seed(r)
    B_full_list[r] = torch.randn(N_local, K, dtype=torch.bfloat16, device='cuda') / scale
B_full = torch.cat(B_full_list, dim=0)  # [N, K]
C_ref = torch.matmul(A, B_full.t())     # [M, N]

if rank == 0:
    print(f"  B_full: {B_full.shape}, C_ref: {C_ref.shape}")
    print(f"  C_ref stats: mean={C_ref.float().mean():.4f}, std={C_ref.float().std():.4f}")

# Run AG-GEMM kernel
if rank == 0:
    print("\n[Running AG-GEMM Kernel]")
iris_device_ctx = iris.get_device_view()

# Debug: verify iris backing and print pointer info
a_iris_ptr = A._iris_tensor.data_ptr() if hasattr(A, '_iris_tensor') else 0
b_iris_ptr = B_shard._iris_tensor.data_ptr() if hasattr(B_shard, '_iris_tensor') else 0
c_iris_ptr = C._iris_tensor.data_ptr() if hasattr(C, '_iris_tensor') else 0
print(f"Rank {rank}: A.data_ptr()=0x{A.data_ptr():x} (iris: 0x{a_iris_ptr:x}), "
      f"B_shard.data_ptr()=0x{B_shard.data_ptr():x} (iris: 0x{b_iris_ptr:x}), "
      f"C.data_ptr()=0x{C.data_ptr():x} (iris: 0x{c_iris_ptr:x})")
# Check if torch tensor is backed by iris (same pointer)
a_ok = A.data_ptr() == a_iris_ptr if a_iris_ptr else "no iris"
b_ok = B_shard.data_ptr() == b_iris_ptr if b_iris_ptr else "no iris"
c_ok = C.data_ptr() == c_iris_ptr if c_iris_ptr else "no iris"
print(f"Rank {rank}: iris-backed: A={a_ok}, B_shard={b_ok}, C={c_ok}")
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
