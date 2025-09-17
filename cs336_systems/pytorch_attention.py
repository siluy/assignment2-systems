# import torch
# import timeit
# import numpy as np
# from cs336_basics.model import scaled_dot_product_attention
# from table import generate_md_table

# def benchmark_attention(d_dims, seq_lens, batch_size=8, n_steps=100, warmup_steps=10):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     results = []
#     print(f"Running benchmark with batch_size={batch_size}, steps={n_steps}, warmup={warmup_steps}")
#     for d_model in d_dims:
#         compiled_attention = torch.compile(scaled_dot_product_attention, mode="max-autotune")
#         for seq_len in seq_lens:
#             print(f"Benchmarking: d_model={d_model}, seq_len={seq_len}...")
#             try:
#                 # 1. 创建随机输入
#                 Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
#                 K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
#                 V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
#                 # warmup
#                 for _ in range(warmup_steps):
#                     # output = scaled_dot_product_attention(Q, K, V) # eager
#                     output = compiled_attention(Q, K, V) # compile
#                     loss = output.sum()
#                     loss.backward()
#                     # 清楚梯度
#                     Q.grad = K.grad = V.grad = None
#                 torch.cuda.synchronize()
#                 # 2. 测量前向传播耗时
#                 forward_timings = []
#                 # 重置峰值内存统计
#                 torch.cuda.reset_peak_memory_stats(device)

#                 for _ in range(n_steps):
#                     torch.cuda.synchronize()
#                     start = timeit.default_timer()
#                     # output = scaled_dot_product_attention(Q, K, V) # eager
#                     output = compiled_attention(Q, K, V) # compile
#                     torch.cuda.synchronize()
#                     end = timeit.default_timer()
#                     forward_timings.append(end - start)
#                 # 3. 在反向传播开始前，测量内存占用
#                 # 测的是反向传播保存的张量+输出张量等的总和
#                 peak_mem_bytes = torch.cuda.max_memory_allocated(device)
#                 peak_mem_gb = peak_mem_bytes / (1024**3)
#                 # 4. 测量反向传播耗时
#                 backward_timings = []
#                 for _ in range(n_steps):
#                     # 每次反向传播前都需要重新计算前向，以生成图片
#                     # output = scaled_dot_product_attention(Q, K, V) # eager
#                     output = compiled_attention(Q, K, V) # compile
#                     loss = output.sum()
#                     torch.cuda.synchronize()
#                     start = timeit.default_timer()
#                     loss.backward()
#                     torch.cuda.synchronize()
#                     end = timeit.default_timer()
#                     backward_timings.append(end - start)
#                     Q.grad = K.grad = V.grad = None

#                 results.append([
#                     d_model, seq_len, 
#                     f"{np.mean(forward_timings)*1000:.2f}",
#                     f"{np.mean(backward_timings)*1000:.2f}",
#                     f"{peak_mem_gb:.2f}"
#                 ])
#             except torch.cuda.OutOfMemoryError:
#                 print(f"OOM at: d_model={d_model}, seq_len={seq_len}")
#                 results.append([d_model, seq_len, "OOM", "OOM", "OOM"])
#                 # torch._dynamo.reset() # 捕获到 oom 后需要重置 compiler 状态，否则会报错
#                 break # 若一个 seq len 导致了 oom，那么后面更大的都会 oom，跳过即可
#             del Q, K, V
#             torch.cuda.empty_cache()
#     return results

# if __name__ == '__main__':
#     d_dims_to_test = [16, 32, 64, 128]
#     seq_lens_to_test = [256, 1024, 4096, 8192, 16384]

#     benchmark_results = benchmark_attention(d_dims_to_test, seq_lens_to_test)
#     columns = ["d_model", "Seq Len", "Fwd Avg(ms)", "Bwd Avg(ms)", "Peak Mem(GB)"]
#     table = generate_md_table(benchmark_results, columns)
#     print(table)

import torch
import timeit
import numpy as np
import argparse
from table import generate_md_table
from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention(d_model, seq_lens, batch_size=8, n_steps=100, warmup_steps=10):
    """
    对单个 d_model 值运行完整的基准测试。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    print(f"Running benchmark with batch_size={batch_size}, steps={n_steps}, warmup={warmup_steps}")

    # 为这一个 d_model 创建全新的编译函数，确保环境干净
    compiled_attention = torch.compile(scaled_dot_product_attention)

    for seq_len in seq_lens:
        print(f"Benchmarking: d_model={d_model}, seq_len={seq_len}...")
        try:
            # 1. 创建随机输入
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            
            # Warmup 预热
            for _ in range(warmup_steps):
                output = compiled_attention(Q, K, V)
                loss = output.sum()
                loss.backward()
                # 清除梯度
                Q.grad = K.grad = V.grad = None
            torch.cuda.synchronize()

            # 2. 测量前向传播耗时
            forward_timings = []
            torch.cuda.reset_peak_memory_stats(device) # 重置峰值内存统计

            for _ in range(n_steps):
                torch.cuda.synchronize()
                start = timeit.default_timer()
                output = compiled_attention(Q, K, V)
                torch.cuda.synchronize()
                end = timeit.default_timer()
                forward_timings.append(end - start)

            # 3. 测量峰值内存占用
            peak_mem_bytes = torch.cuda.max_memory_allocated(device)
            peak_mem_gb = peak_mem_bytes / (1024**3)

            # 4. 测量反向传播耗时
            backward_timings = []
            for _ in range(n_steps):
                # 每次反向传播前都需要重新计算前向，以生成计算图
                output = compiled_attention(Q, K, V)
                loss = output.sum()
                torch.cuda.synchronize()
                start = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                backward_timings.append(end - start)
                Q.grad = K.grad = V.grad = None

            results.append([
                d_model, seq_len,
                f"{np.mean(forward_timings)*1000:.2f}",
                f"{np.mean(backward_timings)*1000:.2f}",
                f"{peak_mem_gb:.2f}"
            ])
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at: d_model={d_model}, seq_len={seq_len}")
            results.append([d_model, seq_len, "OOM", "OOM", "OOM"])
            # 遇到 OOM 时，终止对当前 d_model 更大序列长度的测试
            break
        finally:
            # 确保在循环的每次迭代后都清理张量
            if 'Q' in locals():
                del Q
            if 'K' in locals():
                del K
            if 'V' in locals():
                del V
            torch.cuda.empty_cache()

    # 为这一个 d_model 的所有结果生成表格并打印
    columns = ["d_model", "Seq Len", "Fwd Avg(ms)", "Bwd Avg(ms)", "Peak Mem(GB)"]
    table = generate_md_table(results, columns)
    print(f"\n--- Results for d_model = {d_model} ---\n")
    print(table)


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Benchmark Attention for a specific d_model.")
    parser.add_argument("--d_model", type=int, required=True, help="The d_model dimension to benchmark.")
    args = parser.parse_args()

    # 定义要测试的序列长度
    seq_lens_to_test = [256, 1024, 4096, 8192, 16384]

    # 从命令行获取 d_model，然后调用主函数
    benchmark_attention(args.d_model, seq_lens_to_test)