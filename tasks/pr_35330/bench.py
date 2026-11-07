#!/usr/bin/env python3
"""Microbench for vLLM PR #35330: [Perf] Optimize maxsim scores computation for pooling models.

base: serving.py `_late_interaction_score` loops `compute_maxsim_score(q_emb, d_emb)` per pair on
      the (CPU) pooling outputs: one [q_len, d_len] matmul per pair, max over docs, sum.
gold: `compute_maxsim_scores(q_embs, d_embs)` pads pairs into mini-batches of 16, moves them to
      the GPU, one bmm per batch with masks, scores back to CPU.

The bench calls the arm's own code: it imports `compute_maxsim_scores` when the tree has it
(gold) and otherwise times the per-pair loop over the tree's `compute_maxsim_score` exactly as
serving.py does in the base tree. Inputs mimic the author's run (jinaai/jina-colbert-v2: dim 128,
--random-input-len 2048 --random-range-ratio 0.5 -> doc lengths 1024..2048, 64 concurrent
rerank pairs; query len 32) as CPU float32 tensors, which is what PoolingRequestOutput.data is.
Wall-clock ms per 64-pair batch. The PR reports e2e serving numbers only (141.38 -> 161.14 req/s).
Runs inside the task image: python3 /task/bench.py
"""
import time

import torch

from vllm.entrypoints.pooling.score.utils import compute_maxsim_score

try:
    from vllm.entrypoints.pooling.score.utils import compute_maxsim_scores
except ImportError:          # base tree: per-pair loop as in serving.py before the PR
    compute_maxsim_scores = None

NUM_PAIRS = 64
DIM = 128
Q_LEN = 32
D_MIN, D_MAX = 1024, 2048
WARMUP = 3
ITERS = 20


def make_pairs():
    g = torch.Generator().manual_seed(0)
    q_embs, d_embs = [], []
    for _ in range(NUM_PAIRS):
        d_len = int(torch.randint(D_MIN, D_MAX + 1, (1,), generator=g))
        q_embs.append(torch.randn(Q_LEN, DIM, generator=g))
        d_embs.append(torch.randn(d_len, DIM, generator=g))
    return q_embs, d_embs


def score_batch(q_embs, d_embs):
    if compute_maxsim_scores is not None:
        return compute_maxsim_scores(q_embs, d_embs)
    return [compute_maxsim_score(q, d) for q, d in zip(q_embs, d_embs)]


def main():
    q_embs, d_embs = make_pairs()
    scores = score_batch(q_embs, d_embs)
    ref = [compute_maxsim_score(q, d) for q, d in zip(q_embs, d_embs)]
    assert len(scores) == NUM_PAIRS
    err = max(abs(float(s) - float(r)) for s, r in zip(scores, ref))
    for _ in range(WARMUP):
        score_batch(q_embs, d_embs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        score_batch(q_embs, d_embs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / ITERS
    print(f"maxsim_64pairs {ms:.3f}")
    print(f"# path={'batched_gpu' if compute_maxsim_scores else 'per_pair_cpu'} pairs={NUM_PAIRS} "
          f"dim={DIM} q_len={Q_LEN} d_len={D_MIN}..{D_MAX} max_abs_err_vs_per_pair={err:.4f} iters={ITERS}")


if __name__ == "__main__":
    main()
