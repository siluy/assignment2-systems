import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import os
import numpy as np
from table import generate_md_table

def setup(rank, world_size, backend):
    """初始化进程组"""
    # 尝试手动设置RANK环境变量
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    if backend == "nccl":
        # 让我们坚持使用 rank，因为这才是 PyTorch re-index 后的正确设备号
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def worker_fn(rank, world_size, backend, size_mb, results_dict):
    setup(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    tensor_size_bytes = size_mb * 1024 * 1024
    num_elements = tensor_size_bytes // 4
    data = torch.randn(num_elements, device=device, dtype=torch.float32)
    warmup_steps = 5
    for _ in range(warmup_steps):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    measure_steps = 20
    timings = []
    for _ in range(measure_steps):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = timeit.default_timer()
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time)
    avg_latency_ms = np.mean(timings) * 1000
    if rank == 0:
        results_dict[(backend, world_size, size_mb)] = avg_latency_ms
    cleanup()

def main():
    backends = ["gloo", "nccl"]
    world_sizes = [2, 4, 6]
    tensor_sizes_mb = [1, 10, 100, 1000]
    # Manager Dict 在多进程中共享结果
    manager = mp.Manager()
    results_dict = manager.dict()

    for backend in backends:
        for ws in world_sizes:
            if backend == "nccl" and ws > torch.cuda.device_count():
                print(f"Skipping NCCL test for world_size={ws} (requires {ws} GPUs, have {torch.cuda.device_count()})")
                continue
            for size_mb in tensor_sizes_mb:
                print(f"Running: backend={backend}, world_size={ws}, size_mb={size_mb}")
                mp.spawn(worker_fn,
                         args=(ws, backend, size_mb, results_dict),
                         nprocs=ws,
                         join=True)
    table_data = []
    for (backend, ws, size_mb), latency in sorted(results_dict.items()):
        table_data.append([backend, ws, size_mb, f"{latency:.4f}"])
    table_columns = ["Backend", "World Size", "Size(MB)", "Avg Latency(ms)"]
    table = generate_md_table(table_data, table_columns)
    print(table)

if __name__ == "__main__":
    main()