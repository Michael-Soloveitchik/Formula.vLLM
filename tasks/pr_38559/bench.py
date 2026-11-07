#!/usr/bin/env python3
"""Microbench for vLLM PR #38559: [Perf] Optimize mean pooling using chunks and index_add.

base: MeanPool.forward = fp32 cumsum over ALL tokens (a [tokens, hidden] fp32 temp) then
      cumsum[end] - cumsum[start] + hidden[start], divided by prompt_lens.
gold: segment ids via repeat_interleave + index_add_ into a [num_seqs, hidden] fp32 accumulator,
      walking the token axis in 16 MB chunks (no large temp, no cumsum).

Setup mirrors the author's serving run (intfloat/multilingual-e5-base, --random-input-len 512,
64 concurrent requests, hidden 768, fp16): a batch of 64 x 512-token prompts, plus a wider
hidden (4096) case. The PoolingMetadata / cursor is built with the same calls the model runner
uses (build_pooling_cursor on CPU lengths). Wall-clock us per MeanPool.forward (the forward
ends in a GPU->CPU-free tensor, so we synchronise after each timed block). The PR reports
only e2e serving numbers (238.52 -> 252.72 req/s), kept as reported-only experiments.
Runs inside the task image: python3 /task/bench.py
"""
import time

import numpy as np
import torch

from vllm.model_executor.layers.pooler.seqwise.methods import MeanPool
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

DEVICE = torch.device("cuda:0")
WARMUP = 20
ITERS = 300
# (metric, num_seqs, tokens per seq, hidden, dtype)
SCENARIOS = [
    ("mean_pool_e5base_64x512", 64, 512, 768, torch.float16),
    ("mean_pool_32x256_h4096", 32, 256, 4096, torch.bfloat16),
]


def build_metadata(lengths):
    n = len(lengths)
    prompt_lens = torch.tensor(lengths, dtype=torch.int32)
    md = PoolingMetadata(
        prompt_lens=prompt_lens,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="embed") for _ in range(n)],
        pooling_states=[PoolingStates() for _ in range(n)],
    )
    md.build_pooling_cursor(np.asarray(lengths, dtype=np.int32),
                            seq_lens_cpu=prompt_lens, device=DEVICE)
    return md


def wall_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / ITERS


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(0)
    pool = MeanPool()
    for name, n, seq, hidden, dtype in SCENARIOS:
        lengths = [seq] * n
        md = build_metadata(lengths)
        hs = torch.randn(n * seq, hidden, device=DEVICE, dtype=dtype)
        out = pool(hs, md)
        ref = hs.float().view(n, seq, hidden).mean(1)
        torch.cuda.synchronize()
        assert out.shape == (n, hidden)
        err = (out.float() - ref).abs().max().item()
        us = wall_us(lambda: pool(hs, md))
        print(f"{name} {us:.3f}")
        print(f"# {name}: seqs={n} tokens/seq={seq} hidden={hidden} dtype={dtype} "
              f"max_abs_err_vs_ref={err:.5f} iters={ITERS}")


if __name__ == "__main__":
    main()
