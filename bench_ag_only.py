"""Benchmark RCCL all_gather in isolation for AG+GEMM comparison."""
import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", 1)))

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl")

    # Shape: A_shard is [7680, 1024] bf16 per rank
    # Full gathered A is [7680, 8192] (gathering along dim=1, K dimension)
    M, K_shard = 7680, 1024
    dtype = torch.bfloat16

    A_shard = torch.randn(M, K_shard, dtype=dtype, device=f"cuda:{rank}")

    # Pre-allocate output buffers
    ag_output_list = [torch.empty(M, K_shard, dtype=dtype, device=f"cuda:{rank}") for _ in range(world_size)]

    warmup = 10
    measured = 50

    # --- Benchmark 1: all_gather only ---
    for _ in range(warmup):
        dist.all_gather(ag_output_list, A_shard)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(measured):
        dist.all_gather(ag_output_list, A_shard)
    end_event.record()
    torch.cuda.synchronize()

    ag_only_ms = start_event.elapsed_time(end_event) / measured

    # --- Benchmark 2: all_gather + cat ---
    for _ in range(warmup):
        dist.all_gather(ag_output_list, A_shard)
        A_full = torch.cat(ag_output_list, dim=1)
    torch.cuda.synchronize()

    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)

    start_event2.record()
    for _ in range(measured):
        dist.all_gather(ag_output_list, A_shard)
        A_full = torch.cat(ag_output_list, dim=1)
    end_event2.record()
    torch.cuda.synchronize()

    ag_cat_ms = start_event2.elapsed_time(end_event2) / measured

    # --- Benchmark 3: all_gather_into_tensor (contiguous, no cat needed) ---
    A_full_buf = torch.empty(M, K_shard * world_size, dtype=dtype, device=f"cuda:{rank}")

    for _ in range(warmup):
        dist.all_gather_into_tensor(A_full_buf, A_shard)
    torch.cuda.synchronize()

    start_event3 = torch.cuda.Event(enable_timing=True)
    end_event3 = torch.cuda.Event(enable_timing=True)

    start_event3.record()
    for _ in range(measured):
        dist.all_gather_into_tensor(A_full_buf, A_shard)
    end_event3.record()
    torch.cuda.synchronize()

    ag_into_tensor_ms = start_event3.elapsed_time(end_event3) / measured

    if rank == 0:
        # Data volume: each rank sends M * K_shard * 2 bytes, received by (world_size-1) peers
        # Standard AG bandwidth: data_size * (world_size - 1) / world_size per rank
        shard_bytes = M * K_shard * 2  # bf16 = 2 bytes
        total_data = shard_bytes * world_size  # total gathered data
        # Bus bandwidth formula for all-gather: (N-1)/N * total_data / time
        bus_bw_factor = (world_size - 1) / world_size

        def bw_gbps(ms):
            return (total_data * bus_bw_factor) / (ms / 1000) / 1e9

        print(f"=== RCCL All-Gather Benchmark ===")
        print(f"Shape: A_shard=[{M}, {K_shard}] bf16, {world_size} ranks")
        print(f"Gathered shape: [{M}, {K_shard * world_size}]")
        print(f"Shard size: {shard_bytes / 1024:.1f} KB, Total: {total_data / 1024 / 1024:.1f} MB")
        print(f"Warmup: {warmup}, Measured: {measured}")
        print(f"")
        print(f"all_gather (list):         {ag_only_ms:.4f} ms  ({bw_gbps(ag_only_ms):.1f} GB/s bus BW)")
        print(f"all_gather + cat:          {ag_cat_ms:.4f} ms  ({bw_gbps(ag_cat_ms):.1f} GB/s bus BW)")
        print(f"all_gather_into_tensor:    {ag_into_tensor_ms:.4f} ms  ({bw_gbps(ag_into_tensor_ms):.1f} GB/s bus BW)")
        print(f"")
        print(f"cat overhead:              {ag_cat_ms - ag_only_ms:.4f} ms")
        print(f"")
        print(f"For reference: AG+matmul total = 1.783 ms")
        print(f"Remaining for matmul (if using into_tensor): {1.783 - ag_into_tensor_ms:.4f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
