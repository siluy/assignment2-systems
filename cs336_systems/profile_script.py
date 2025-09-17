import torch
import torch.nn as nn
import timeit
import numpy as np
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW
from table import generate_md_table


def profile_model_with_nvtx(model_config, context_length, batch_size, steps, run_backward=False):
    """
    用于 Nsight Systems 剖析的脚本，使用 NVTX ranges 标记关键区域
    此处不再需要 warmup 和手动 timeit，因为 nsys 能自动追踪每一次运行
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模型和数据
    model = Transformer(**model_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    # optimizer = AdamW(model.parameters())
    vocab_size = 10000
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    # 2. 使用 NVTX range 标记要剖析的循环
    for i in range(steps):
        # 用 f-string 给每个步骤一个唯一的标记
        with nvtx.range(f"Step_{i}"):
            optimizer.zero_grad()
            # 标记前向传播
            with nvtx.range("Forward Pass"):
                outputs = model(input_ids)
        if run_backward:
            loss = outputs.sum()
            # 标记反向传播
            with nvtx.range("Backward Pass"):
                loss.backward()
            # 标记优化器步骤
            with nvtx.range("Optimizer Step"):
                optimizer.step()
        
        torch.cuda.synchronize()
    
# if __name__ == '__main__':
#     # 选择一个模型尺寸开始，比如 'small'
#     small_config = {
#         'vocab_size': 10000,
#         'context_length': 512,
#         'd_model': 768,
#         'num_layers': 12,
#         'num_heads': 12,
#         'd_ff': 3072,
#         "rope_theta": 10000,
#     }

#     print("开始为 nsys 运行模型，请从命令行启动此脚本...")
#     # 运行一个完整的训练步骤（对应 d 小问）
#     profile_model_with_nvtx(
#         model_config=small_config,
#         context_length=512,
#         batch_size=4,
#         steps=5, # 运行5个步骤，方便在报告中观察
#         run_backward=True
#     )
#     print("运行结束。")

def profile_memory(model_config, context_length, batch_size, run_backward=True, profile_name="memory_snapshot"):
    """
    对模型进行内存剖析，并保存快照
    """
    device = torch.device("cuda")
    model = Transformer(**model_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, context_length), device=device)
    # 记录内存历史
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    # 运行要剖析的代码
    optimizer.zero_grad()
    outputs = model(input_ids)
    if run_backward:
        loss = outputs.sum()
        loss.backward()
        optimizer.step()
    # 确保 GPU 操作完成
    torch.cuda.synchronize()
    # 保存内存快照到文件
    torch.cuda.memory._dump_snapshot(f"{profile_name}.pickle")
    # 停止记录
    torch.cuda.memory._record_memory_history(enabled=None)

if __name__ == '__main__':
    large_config = {
        'vocab_size': 10000, 'context_length': 512, 'd_model': 1280,
        'num_layers': 36, 'num_heads': 20, 'd_ff': 5120, 'rope_theta': 10000,
    }

    # 剖析仅前向传播
    profile_memory(large_config, 512, 4, run_backward=False, profile_name="large_forward_only")
    
    # 剖析完整的训练步骤
    profile_memory(large_config, 512, 4, run_backward=True, profile_name="large_full_step")