import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import timeit
import numpy as np
from typing import Any, List
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.model import AdamW

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def naive_ddp_worker(rank, world_size, model_config, num_steps=5):
    setup(rank, world_size)
    # 1. initialize model and optimizer
    torch.manual_seed(42)
    model = Transformer(**model_config).to(rank)
    optimizer = AdamW(model.parameters())
    # 2. rank0 的权重广播给全体进程
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    for step in range(num_steps):
        # 3. 创建数据并分片
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, 128), device=rank)
        local_batch = full_batch[rank * 4 : (rank + 1) * 4]
        # 4. 本地计算
        optimizer.zero_grad()
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()
        # 5. 梯度同步，逐参数做 all-reduce
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.gard.data /= world_size
        # 6. 模型更新
        optimizer.step()

# 最基础的ddp实现，存在逐参数通信过多的问题
def naive_ddp_benchmark(rank, world_size, model_config, num_steps=10):
    setup(rank, world_size)
    torch.manual_seed(42)
    model = Transformer(**model_config).to(rank)
    optimizer = AdamW(model.parameters())
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    total_tims = []
    comm_times = []
    # warmup
    for _ in range(5):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, 512), device=rank)
        local_batch = full_batch[rank * 4 : (rank + 1) * 4]
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        optimizer.step()
    # measure
    for _ in range(num_steps):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, 512), device=rank)
        local_batch = full_batch[rank * 4 : (rank + 1) * 4]
        
        torch.cuda.synchronize()
        step_start_time = timeit.default_timer()

        optimizer.zero_grad()
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        comm_start_time = timeit.default_timer()

        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        
        torch.cuda.synchronize()
        comm_end_time = timeit.default_timer()
        comm_times.append(comm_end_time - comm_start_time)
        
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= world_size
        optimizer.step()

        torch.cuda.synchronize()
        step_end_time = timeit.default_timer()
        total_tims.append(step_end_time - step_start_time)
    
    if rank == 0:
        avg_total_time = np.mean(total_tims) * 1000
        avg_comm_time = np.mean(comm_times) * 1000
        comm_proportion = (avg_comm_time / avg_total_time) * 100

        print(f"World Size: {world_size}")
        print(f"Average total time per step: {avg_total_time:.2f} ms")
        print(f"Average communication time per step: {avg_comm_time:.2f} ms")
        print(f"Proportion of time spent on communication: {comm_proportion:.2f}%")

    cleanup()

# 优化通信，不再循环 all-reduce 每个参数的梯度，将所有参数的梯度统一压入一个一维张量，作一次 all-reduce，再还原
def flap_ddp_benchmark(rank, world_size, model_config, num_steps=10):
    setup(rank, world_size)
    torch.manual_seed(42)
    model = Transformer(**model_config).to(rank)
    optimizer = AdamW(model.parameters())
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    total_tims = []
    comm_times = []
    # warmup
    for _ in range(5):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, 512), device=rank)
        local_batch = full_batch[rank * 4 : (rank + 1) * 4]
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        optimizer.step()
    # measure
    for _ in range(num_steps):
        full_batch = torch.randint(0, model_config['vocab_size'], (world_size * 4, 512), device=rank)
        local_batch = full_batch[rank * 4 : (rank + 1) * 4]
        
        torch.cuda.synchronize()
        step_start_time = timeit.default_timer()

        optimizer.zero_grad()
        outputs = model(local_batch)
        loss = outputs.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        comm_start_time = timeit.default_timer()

        # core improvement:
        # 1. 将所有梯度收集到列表
        grads = [p.grad.data for p in model.parameters() if p.grad is not None]
        # 2. 将列表 flatten 为一维
        flat_grad = torch._utils._flatten_dense_tensors(grads)
        # 3. 对一维大张量作一次 all-reduce
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        # 4. avg
        flat_grad /= world_size
        # 5. 还原
        new_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
        # 6. 将新梯度复制回模型的 .grad
        for old_g, new_g in zip(grads, new_grads):
            old_g.copy_(new_g)
        
        torch.cuda.synchronize()
        comm_end_time = timeit.default_timer()
        comm_times.append(comm_end_time - comm_start_time)
        
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data /= world_size
        optimizer.step()

        torch.cuda.synchronize()
        step_end_time = timeit.default_timer()
        total_tims.append(step_end_time - step_start_time)
    
    if rank == 0:
        avg_total_time = np.mean(total_tims) * 1000
        avg_comm_time = np.mean(comm_times) * 1000
        comm_proportion = (avg_comm_time / avg_total_time) * 100

        print(f"World Size: {world_size}")
        print(f"Average total time per step: {avg_total_time:.2f} ms")
        print(f"Average communication time per step: {avg_comm_time:.2f} ms")
        print(f"Proportion of time spent on communication: {comm_proportion:.2f}%")

    cleanup()

# 一种将计算和通信 overlap 的DDP，效率更高，通过异步通信，每计算完成一次就通信同步一次
class DDP:
    def __init__(self, module: nn.Module):
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # 1. 广播初始权重
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        # 2. 准备存放异步通信的列表
        self.handles = []
        # 3. register_post_accumulate_grad_hook
        for p in reversed(list(self.module.parameters())):
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._create_hook(p))
    
    def _create_hook(self, p: nn.Parameter):
        def hook(*args: Any) -> None:
            if p.grad is not None:
                handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)
        return hook

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        """等待所有异步通信完成"""
        for handle in self.handles:
            handle.wait()
        # 梯度平均
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.data /= self.world_size

        self.handles.clear()

# 分桶 DDP，结合了 overlap 和 batch sync
class BucketedDDP:
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        # broadcast initial weights
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        # 1. 参数分桶
        self.buckets: List[List[nn.Parameter]] = []
        self.param_to_bucket_id: dict[nn.Parameter, int] = {}
        current_bucket: List[nn.Parameter] = []
        current_bucket_size = 0
        # 逆序遍历参数，能有效减少重复的 overlapping
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            p_size = p.numel() * p.element_size()
            if current_bucket_size + p_size > self.bucket_size_bytes and current_bucket:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_bucket_size = 0
            current_bucket.append(p)
            current_bucket_size += p_size
            self.param_to_bucket_id[p] = len(self.buckets)
        if current_bucket:
            self.buckets.append(current_bucket)
        # 2. 状态管理，注册 hook
        self.ready_params_per_bucket: List[int] = [0] * len(self.buckets)
        self.bucket_communication_handles = [None] * len(self.buckets)
        self.bucket_flat_grads = [None] * len(self.buckets)
        for bucket_id, bucket in enumerate(self.buckets):
            for p in bucket:
                p.register_post_accumulate_grad_hook(self._create_hook(bucket_id, bucket))
        
    def _create_hook(self, bucket_id: int, bucket: List[nn.Parameter]):
        def hook(*args: Any) -> None:
            self.ready_params_per_bucket[bucket_id] += 1
            if self.ready_params_per_bucket[bucket_id] == len(bucket):
                self._trigger_communication(bucket_id, bucket)
        return hook
    
    def _trigger_communication(self, bucket_id: int, bucket: List[nn.Parameter]):
        grads = [p.grad.data for p in bucket]
        flat_grad = torch._utils._flatten_dense_tensors(grads)
        self.bucket_flat_grads[bucket_id] = flat_grad
        handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
        self.bucket_communication_handles[bucket_id] = handle
    
    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self) -> None:
        # 遍历所有 bucket，等待通信并解包梯度
        for bucket_id in range(len(self.buckets)):
            handle = self.bucket_communication_handles[bucket_id]
            if handle is None:
                continue
            handle.wait()

            flat_grad = self.bucket_flat_grads[bucket_id]
            flat_grad /= self.world_size
            # 解包并将梯度写回
            original_params_in_bucket = self.buckets[bucket_id]
            original_grads = [p.grad.data for p in original_params_in_bucket]
            new_grads = torch._utils._unflatten_dense_tensors(flat_grad, original_grads)
            for p, new_grad in zip(original_params_in_bucket, new_grads):
                p.grad.data.copy_(new_grad)
        
        # 重置状态准备下一次迭代
        self.ready_params_per_bucket = [0] * len(self.buckets)
        self.bucket_communication_handles = [None] * len(self.buckets)
        self.bucket_flat_grads = [None] * len(self.buckets)