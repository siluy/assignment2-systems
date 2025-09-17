import torch
import torch.nn as nn
import timeit
import numpy as np
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW
from table import generate_md_table
from contextlib import nullcontext

def benchmark_model(model_config, context_length, batch_size, warmup_steps, measure_steps, run_backward=True, use_amp=False):
    """
    对模型执行端到端的基准测试

    Args:
        model_config (dict): 模型超参的字典
        context_length: 输入长度
        batch_size: 批大小，默认4
        warmup_steps: warmup 步数
        measire_steps: 正式计时步数
        run_backward: 是否计入反向传播

    Returns:
        tuple: (平均耗时，标准差)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. 初始化模型和随机数据
    model = Transformer(**model_config).to(device)
    model = torch.compile(model) # compile
    model.train() # 在训练模式计入反向传播
    vocab_size = 10000
    input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    timings = []

    # A100 cannot run the experiment... even with autocast!!!!
    dtype = torch.bfloat16
    amp_context = torch.autocast(device_type="cuda", dtype=dtype) if use_amp else nullcontext()
    # 2. warmup
    print(f"正在进行{warmup_steps}次 warmup")
    for _ in range(warmup_steps):
        with amp_context: # 应用上下文
            outputs = model(input_ids)
            if run_backward:
                loss = outputs.sum()
        if run_backward:
            loss.backward()
    torch.cuda.synchronize()

    # 3. 测量
    print(f"正在进行{measure_steps}次测量")
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        
        with amp_context: # 同样应用上下文
            outputs = model(input_ids)
            if run_backward:
                loss = outputs.sum()
        if run_backward:
            loss.backward()

        torch.cuda.synchronize() # 等 GPU 完成当前所有计算
        end_time = timeit.default_timer()

        timings.append(end_time - start_time)

    avg_time = np.mean(timings)
    std_dev = np.std(timings)

    return avg_time, std_dev

def run_benchmarks():
    model_config = {
        "small": {"d_model":768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        # "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        # "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

    result = []

    for name, config in model_config.items():
        print(f"正在测试模型：{name}")
        full_config = {
            'vocab_size': 10000,
            'context_length': 512,
            'rope_theta': 10000.0,
            **config
        }

        # 测量前向传播
        fw_avg, fw_std = benchmark_model(full_config, 512, 4, 5, 10, run_backward=False, use_amp=True)
        # forward + backward
        bw_avg, bw_std = benchmark_model(full_config, 512, 4, 5, 10, run_backward=True, use_amp=True)

        result.append([
            name,
            f"{fw_avg*1000:.2f}",
            f"{fw_std*1000:.2f}",
            f"{bw_avg*1000:.2f}",
            f"{bw_std*1000:.2f}"
        ])

    columns = ["Model Size", "Fwd Avg(ms)", "Fwd Std(ms)", "Fwd+Bwd Avg(ms)", "Fwd+Bwd Std(ms)"]
    md_output = generate_md_table(result, columns)
    print(md_output)


run_benchmarks()

