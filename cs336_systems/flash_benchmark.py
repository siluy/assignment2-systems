import torch
import triton
import triton.testing
import pytest
from torch.nn.functional import scaled_dot_product_attention as pytroch_attention
from cs336_systems.flash import FlashAttentionTriton
from table import generate_md_table

def run_flash_benchmarking():
    seq_lens = [128,256,512,1024,2048,4096,8192,16384,32768,65536]
    d_models = [16,32,64,128]
    precisions = [torch.float32, torch.bfloat16]
    results = []

    batch_size = 1
    n_head = 1
    is_causal = True

    for p in precisions:
        for d in d_models:
            for n in seq_lens:
                print(f"Testing: precision={p.__str__()}, d_model={d}, seq_len={n}")
                q = torch.randn(batch_size, n_head, n, d, device='cuda', dtype=p, requires_grad=True)
                k = torch.randn(batch_size, n_head, n, d, device='cuda', dtype=p, requires_grad=True)
                v = torch.randn(batch_size, n_head, n, d, device='cuda', dtype=p, requires_grad=True)
                row = [p.__str__().split('.')[-1], d, n]
                fwd_torch = lambda: pytroch_attention(q, k, v, is_causal=is_causal)
                fwd_triton = lambda: FlashAttentionTriton.apply(q, k, v, is_causal)
                try:
                    o_torch = fwd_torch()
                    bwd_torch = lambda: o_torch.backward(torch.ones_like(o_torch), retain_graph=True)
                    fwd_bwd_torch = lambda: (fwd_torch(), bwd_torch())
                    ms_fwd_torch = triton.testing.do_bench(fwd_torch, rep=50)
                    ms_bwd_torch = triton.testing.do_bench(bwd_torch, rep=50)
                    ms_fwd_bwd_torch = triton.testing.do_bench(fwd_bwd_torch, rep=50)
                    row.extend([f"{ms_fwd_torch:.3f}", f"{ms_bwd_torch:.3f}", f"{ms_fwd_bwd_torch:.3f}"])
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    print("PyTorch OOM")
                    row.extend(["OOM"] * 3)
                
                try:
                    q_no_head, k_no_head, v_no_head = q.squeeze(1), k.squeeze(1), v.squeeze(1)
                    o_triton_no_head = FlashAttentionTriton.apply(q_no_head, k_no_head, v_no_head, is_causal)
                    o_triton = o_triton_no_head.unsqueeze(1)
                    bwd_triton = lambda: o_triton.backward(torch.ones_like(o_triton), retain_graph=True)
                    fwd_bwd_triton = lambda: (FlashAttentionTriton.apply(q_no_head, k_no_head, v_no_head, is_causal), bwd_triton())
                    ms_fwd_triton = triton.testing.do_bench(lambda: FlashAttentionTriton.apply(q_no_head, k_no_head, v_no_head, is_causal), rep=50)
                    ms_bwd_triton = triton.testing.do_bench(bwd_triton, rep=50)
                    ms_fwd_bwd_triton = triton.testing.do_bench(fwd_bwd_triton, rep=50)
                    row.extend([f"{ms_fwd_triton:.3f}", f"{ms_bwd_triton:.3f}", f"{ms_fwd_bwd_triton:.3f}"])
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    print("Triton OOM")
                    row.extend(["OOM"] * 3)
                
                results.append(row)
                del q, k, v
                torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    benchdata = run_flash_benchmarking()
    table_columns = [
        "Precision", "d_model", "Seq Len",
        "PyTorch Fwd (ms)", "PyTorch Bwd (ms)", "PyTorch Total (ms)",
        "Triton Fwd (ms)", "Triton Bwd (ms)", "Triton Total (ms)"
    ]
    table = generate_md_table(benchdata, table_columns)
    print(table)