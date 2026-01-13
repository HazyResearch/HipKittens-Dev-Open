import torch
import sys; sys.path.insert(0, "..")
import iris_py
import tk_kernel


torch.manual_seed(0)

# Helper function unchanged
def make_iris_tensor(iris, shape, dtype="bfloat16"):
    import ctypes
    iris_tensor = iris.empty(shape, dtype=dtype)

    dtype_map = {
        "bfloat16": (torch.bfloat16, "<u2"),
        "bf16": (torch.bfloat16, "<u2"),
        "float32": (torch.float32, "<f4"),
        "float": (torch.float32, "<f4"),
        "float16": (torch.float16, "<f2"),
        "half": (torch.float16, "<f2"),
    }
    torch_dtype, typestr = dtype_map[dtype]

    class CudaArrayWrapper:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {
                'shape': tuple(shape),
                'typestr': typestr,
                'data': (ptr, False),
                'version': 3,
                'strides': None,
            }
            self._iris_tensor = iris_tensor

    wrapper = CudaArrayWrapper(iris_tensor.data_ptr(), shape, typestr)

    try:
        torch_tensor = torch.as_tensor(wrapper, device='cuda').to(torch_dtype)
    except:
        torch_tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
        torch_tensor._iris_tensor = iris_tensor

    return torch_tensor

# Setup
M = 192 * 40
K = 8192
N = 8192
scale = 10.0

iris = iris_py.Iris(heap_size_mb=512, verbose=False)
rank = iris.rank()
world_size = iris.world_size()
torch.cuda.set_device(rank)

K_local = K // world_size

if rank == 0:
    print("="*50)
    print(f"All-Gather Matmul: {M}x{K} @ {K}x{N}")
    print("="*50)

# Allocate
if rank == 0:
    print("\n[Allocating Iris Tensors]")
A = make_iris_tensor(iris, [world_size, M, K_local])
B = make_iris_tensor(iris, [N, K])
C = make_iris_tensor(iris, [M, N])

if rank == 0:
    print(f"✓ A_local: {A.shape}")
    print(f"✓ B_local: {B.shape}")
    print(f"✓ C_local: {C.shape}")

# Init
if rank == 0:
    print("\n[Initializing In-Place]")
torch.manual_seed(1234 + rank)
A.zero_()
A[rank, :, :].copy_(torch.randn(M, K_local, device='cuda', dtype=torch.bfloat16) / scale) 
torch.manual_seed(0)
B.copy_(torch.randn(N, K, device='cuda', dtype=torch.bfloat16) / scale)

C.zero_()
iris.barrier()

# Run kernel
if rank == 0:
    print("\n[Running HipKittens Kernel]")

iris_device_ctx = iris.get_device_view()
tk_kernel.dispatch_micro(
    A, B, C,
    iris_device_ctx,
    M, N, K_local
)

if rank == 0: breakpoint()

torch.cuda.synchronize()
iris.barrier()

# Reference (uses gathered A_full written by kernel)
if rank == 0:
    print("\n[Computing Reference]")

import torch.distributed as dist
import os
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
tmp = torch.empty((world_size * M, K_local), device='cuda', dtype=torch.bfloat16)
dist.all_gather_into_tensor(tmp, A[rank, :, :])
A_ref = tmp.view(world_size, M, K_local)

if rank == 0:
    breakpoint()
C_ref = 0
for s in range(world_size):
    B_shard_t = B[:, s*K_local:(s+1)*K_local].t()   
    C_ref += A_ref[s].float() @ B_shard_t.float()
dist.destroy_process_group()

# Validate
if rank == 0:
    print("\n[Validating Results]")
diff = (C.float() - C_ref.float()).abs()
max_error = diff.max().item()
status = "✓ PASSED" if max_error < 0.1 else "✗ FAILED"
print(f"Rank {rank}: Max error: {max_error:.6f}, {status}")

if rank == 0:
    print("\n" + "="*50)
    print("Done!")
    print("="*50)

# Cleanup
import gc
from mpi4py import MPI
del A, B, C, C_ref
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
