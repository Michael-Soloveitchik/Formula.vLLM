#!/usr/bin/env python3
"""Microbench for vLLM PR #34846: [Perf] Improve default triton fused moe configs.

The PR only changes `get_default_config` in fused_moe.py - the Triton tile config used when no
tuned JSON exists for (E, N, device). base: BLOCK_SIZE_M/N/K = 16/32/64 (M <= E) or 64/64/32,
no num_warps / num_stages. gold: batch-aware BLOCK_SIZE_M 16..128, BLOCK_SIZE_K 64/128,
GROUP_SIZE_M by tokens-per-expert, explicit num_warps / num_stages.

The bench is the author's benchmarks/kernels/benchmark_moe_defaults.py core (fused_experts on
random bf16 weights) without the override_config plumbing: the arm's own default config is what
fused_experts picks on an untuned GPU (the RTX 5070 Ti has no tuned file). Shape = a many-expert
bf16 model like OLMoE / DeepSeek-V2-Lite: E=64, N=1408, K=2048, top_k=6; batches 1..4096.
Routing is a torch softmax + topk so both arms see identical topk tensors. CUDA-event us per call.
The author reports only New/Old ratios per model band (e.g. DeepSeek-V2 E=64: 0.93-1.01x at
batch 1-8, 0.74-1.00x at 16-128, 0.54-0.71x at 256-4096 on H100) and an fp8 Mixtral table, so
`reported` is null here. NOTE: if a tuned config for this GPU exists in the tree, both arms use
it and the bench measures nothing - check the "# config" lines.
Runs inside the task image: python3 /task/bench.py
"""
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, get_default_config

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
E, N, K, TOPK = 64, 1408, 2048, 6
BATCHES = [1, 16, 128, 1024, 4096]
WARMUP = 10
ITERS = 50


def route(gating, topk):
    probs = torch.softmax(gating.float(), dim=-1)
    w, ids = torch.topk(probs, topk, dim=-1)
    w = w / w.sum(dim=-1, keepdim=True)
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
    w1 = torch.randn(E, 2 * N, K, device=DEVICE, dtype=DTYPE) / 10
    w2 = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE) / 10
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
        print(f"fused_moe_e64_top6_m{M} {us:.3f}")
        try:
            cfg = get_default_config(M * TOPK, E, N, K, TOPK, None)
        except TypeError:
            cfg = get_default_config(M * TOPK, E, N, K, TOPK, None, None)
        print(f"# M={M} E={E} N={N} K={K} topk={TOPK} dtype={DTYPE} iters={ITERS} default_config={cfg}")


if __name__ == "__main__":
    main()
