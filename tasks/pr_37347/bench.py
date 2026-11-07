#!/usr/bin/env python3
"""Microbench for vLLM PR #37347: [Perf] Optimize token_embed for pooling models.

What survived review is item 2 of the PR: `PoolingMetadata.build_pooling_cursor` gains a
`query_start_loc_gpu` argument so the model runner hands over the token-offset cumsum it already
has on the GPU (`self.query_start_loc.gpu[:num_reqs+1]`) instead of allocating a pinned CPU
buffer, running torch.cumsum on it and copying it to the device on every pooling step.

The bench builds a PoolingMetadata the way the runner does and times `build_pooling_cursor`:
the gold tree accepts `query_start_loc_gpu` (passed, as the runner now does); the base tree
raises TypeError on the new kwarg, so there the bench calls it the only way that tree allows
(CPU cumsum + H2D). Batches of 128 (the author's --max-num-seqs) and 16 sequences. Wall-clock
us per call, synchronised per timed block. Author: ~1% token throughput on e5-small token_embed
serving (no absolute numbers), so `reported` is null.
Runs inside the task image: python3 /task/bench.py
"""
import time

import numpy as np
import torch

from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

DEVICE = torch.device("cuda:0")
SEQ_LEN = 256
WARMUP = 50
ITERS = 2000
BATCHES = [128, 16]


def build_metadata(n):
    prompt_lens = torch.full((n,), SEQ_LEN, dtype=torch.int32)
    return PoolingMetadata(
        prompt_lens=prompt_lens,
        prompt_token_ids=None,
        pooling_params=[PoolingParams(task="token_embed") for _ in range(n)],
        pooling_states=[PoolingStates() for _ in range(n)],
    ), prompt_lens


def make_runner(md, prompt_lens, n):
    sched = np.full(n, SEQ_LEN, dtype=np.int32)
    qsl_gpu = torch.arange(0, (n + 1) * SEQ_LEN, SEQ_LEN, device=DEVICE, dtype=torch.int64)
    try:
        md.build_pooling_cursor(sched, seq_lens_cpu=prompt_lens, device=DEVICE,
                                query_start_loc_gpu=qsl_gpu)
        mode = "reuse_query_start_loc_gpu"

        def run():
            md.build_pooling_cursor(sched, seq_lens_cpu=prompt_lens, device=DEVICE,
                                    query_start_loc_gpu=qsl_gpu)
    except TypeError:
        mode = "cpu_cumsum_h2d"

        def run():
            md.build_pooling_cursor(sched, seq_lens_cpu=prompt_lens, device=DEVICE)
    return run, mode


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
    for n in BATCHES:
        md, prompt_lens = build_metadata(n)
        run, mode = make_runner(md, prompt_lens, n)
        run()
        cur = md.get_pooling_cursor()
        torch.cuda.synchronize()
        assert int(cur.last_token_indices_gpu[-1]) == n * SEQ_LEN - 1
        us = wall_us(run)
        print(f"build_pooling_cursor_n{n} {us:.3f}")
        print(f"# n_seqs={n} tokens/seq={SEQ_LEN} path={mode} iters={ITERS}")


if __name__ == "__main__":
    main()
