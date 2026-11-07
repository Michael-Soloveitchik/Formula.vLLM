#!/usr/bin/env python3
"""Microbench for vLLM PR #29354: Add unpermute-aware fused MoE path and small-batch fallback.

base: fused_experts always runs moe_align_block_size (token sort / padding per expert) and the
      permuted fused_moe_kernel, even when a handful of tokens touch a few of many experts.
gold: when num_tokens * top_k * 4 <= num_experts (and no expert_map / block quant), skips the
      alignment and launches fused_moe_kernel with `sorted_token_ids is None` (naive block
      assignment: one block per (token, expert) pair, no permute / unpermute data movement).

Setup mirrors the author's first table (benchmarks/kernels/benchmark_moe.py on
Llama-4-Maverick-17B-128E, TP=8): E=128 experts, top_k=1, N=1024 (8192/8), K=5120, bf16,
decode batch sizes M in {1, 4, 16, 32} - all on the new path (M*1*4 <= 128). Routing is a plain
softmax + topk done in torch so both arms get identical topk_weights / topk_ids. Timed with
CUDA events, mean of ITERS fused_experts calls.
Author (H100, us): M=1 33.14->28.19, M=4 49.42->44.02, M=16 136.29->130.47, M=32 234.63->228.19.
Runs inside the task image: python3 /task/bench.py
"""
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
E, N, K, TOPK = 128, 1024, 5120, 1
BATCHES = [1, 4, 16, 32]
WARMUP = 20
ITERS = 100


def route(gating, topk):
    probs = torch.softmax(gating.float(), dim=-1)
    w, ids = torch.topk(probs, topk, dim=-1)
    return w.contiguous(), ids.to(torch.int32).contiguous()


def time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(0)
    w1 = (torch.randn(E, 2 * N, K, device=DEVICE, dtype=DTYPE) / 10)
    w2 = (torch.randn(E, K, N, device=DEVICE, dtype=DTYPE) / 10)
    for M in BATCHES:
        a = torch.randn(M, K, device=DEVICE, dtype=DTYPE) / 10
        gating = torch.randn(M, E, device=DEVICE, dtype=torch.float32)
        topk_w, topk_ids = route(gating, TOPK)

        def run():
            return fused_experts(a, w1, w2, topk_w, topk_ids, inplace=False,
                                 global_num_experts=E)

        out = run()
        torch.cuda.synchronize()
        assert out.shape == (M, K) and torch.isfinite(out.float()).all()
        us = time_us(run)
        print(f"fused_moe_e128_top1_m{M} {us:.3f}")
        print(f"# M={M} E={E} N={N} K={K} topk={TOPK} dtype={DTYPE} iters={ITERS}")


if __name__ == "__main__":
    main()
