#!/usr/bin/env python3
"""Microbench for vLLM PR #26437: [PERF] Qwen3-next MTP speedup.

The PR has three parts: (1) boolean-mask indexing -> index_select in the Qwen3-Next model code
(model-level, needs the model, not measured here), (2) `@tensor_cache` in
vllm/model_executor/layers/fla/ops/utils.py grows from 4 to 8 entries - GDN attention calls the
cached index helpers twice per layer for 3 tensor groups (6 distinct cu_seqlens objects), so a
4-entry LRU keyed on tensor identity thrashes and recomputes every call, (3) early .contiguous().

This bench measures (2) with the arm's real cached helper `prepare_chunk_indices`
(fla/ops/index.py): 6 distinct cu_seqlens tensors are cycled round-robin exactly like the
per-step call pattern. base (4 entries) misses on every call and recomputes - prepare_lens +
cdiv + `.tolist()` (device sync) + per-sequence aranges + cat; gold (8 entries) hits after the
first round. Wall-clock us per call. Reported e2e (Qwen3-Next-80B MTP, 512 concurrency, B200):
mean TPOT 22.89 -> 21.12 ms, output 15947.50 -> 17872.27 tok/s - covers all three parts.
Runs inside the task image: python3 /task/bench.py
"""
import time

import torch

from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices

DEVICE = torch.device("cuda:0")
CHUNK = 64
NUM_KEYS = 6          # 2 cached calls per GDN layer x 3 tensor groups
NUM_SEQS = 32
WARMUP = 60
ITERS = 3000


def make_keys():
    g = torch.Generator().manual_seed(0)
    keys = []
    for _ in range(NUM_KEYS):
        lens = torch.randint(1, 1024, (NUM_SEQS,), generator=g)
        cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).to(DEVICE)
        keys.append(cu)
    return keys


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    keys = make_keys()
    ref = [prepare_chunk_indices(cu, CHUNK).clone() for cu in keys]

    def step(i):
        return prepare_chunk_indices(keys[i % NUM_KEYS], CHUNK)

    for i in range(WARMUP):
        step(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(ITERS):
        step(i)
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / ITERS
    for i, cu in enumerate(keys):
        assert torch.equal(prepare_chunk_indices(cu, CHUNK), ref[i])
    print(f"chunk_indices_6way_cycle {us:.3f}")
    print(f"# keys={NUM_KEYS} seqs/key={NUM_SEQS} chunk={CHUNK} iters={ITERS} "
          f"(round-robin over 6 cu_seqlens; base 4-entry cache thrashes, gold 8-entry hits)")


if __name__ == "__main__":
    main()
