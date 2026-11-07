#!/usr/bin/env python3
"""Microbench for vLLM PR #41043: [Perf][Spec Decode] Avoid per-step numpy allocation in
prepare_next_token_ids_padded.

base: builds the "backup" next-token ids with `.tolist()` + a list comprehension + `np.array(...)`
      and then copies into the pinned CpuGpuBuffer (3 temporaries per decode step).
gold: writes each id straight into `self.backup_next_token_ids.np[i]`.
Both arms then run the same Triton kernel (eagle_prepare_next_token_padded_kernel).

We call the arm's real `SpecDecodeBaseProposer.prepare_next_token_ids_padded` (unbound) with a
minimal `self` holding only `backup_next_token_ids` (a real CpuGpuBuffer, built as the proposer
builds it), a stand-in InputBatch (num_reqs, req_ids, num_tokens_no_spec, vocab_size) and
request objects exposing `get_token_id`. Decode batches of 16 (author: typical 1-16) and 256.
Wall-clock us per call, mean of ITERS calls (host work + kernel launch; synchronised per block).
The PR reports only e2e TPOT (Llama-3.1-8B + Eagle3 on H200: mean 2.60 -> 2.58 ms, P99 3.24 ->
2.94 ms), kept as reported-only experiments.
Runs inside the task image: python3 /task/bench.py
"""
import time

import numpy as np
import torch

from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.utils import CpuGpuBuffer

DEVICE = torch.device("cuda:0")
MAX_BATCH = 1024
VOCAB = 128256          # Llama-3 vocab
NUM_SPEC = 3            # num_speculative_tokens -> sampled_token_ids has NUM_SPEC + 1 columns
WARMUP = 100
ITERS = 2000
BATCHES = [16, 256]


class FakeRequest:
    def __init__(self, token_ids):
        self._ids = token_ids

    def get_token_id(self, idx):
        return self._ids[idx]


class FakeInputBatch:
    def __init__(self, num_reqs, rng):
        self.num_reqs = num_reqs
        self.req_ids = [f"req-{i}" for i in range(num_reqs)]
        self.num_tokens_no_spec = rng.integers(64, 4096, size=MAX_BATCH).astype(np.int32)
        self.vocab_size = VOCAB


class Shell:
    def __init__(self):
        self.backup_next_token_ids = CpuGpuBuffer(
            MAX_BATCH, dtype=torch.int32, pin_memory=is_pin_memory_available(),
            device=DEVICE, with_numpy=True)


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
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    shell = Shell()
    prepare = SpecDecodeBaseProposer.prepare_next_token_ids_padded
    for num_reqs in BATCHES:
        batch = FakeInputBatch(num_reqs, rng)
        requests = {
            rid: FakeRequest(rng.integers(0, VOCAB, size=4096).tolist()) for rid in batch.req_ids
        }
        sampled = torch.randint(0, VOCAB, (num_reqs, NUM_SPEC + 1), device=DEVICE, dtype=torch.int64)
        # every 4th request is "discarded" (still prefilling): its row carries no sampled token
        sampled[::4] = -1
        discard = torch.zeros(num_reqs, dtype=torch.bool, device=DEVICE)
        discard[::4] = True

        next_ids, valid = prepare(shell, sampled, requests, batch, discard)
        torch.cuda.synchronize()
        assert next_ids.shape == (num_reqs,) and valid.shape == (num_reqs,)
        exp0 = requests[batch.req_ids[0]].get_token_id(int(batch.num_tokens_no_spec[0]) - 1)
        assert int(next_ids[0]) == exp0, (int(next_ids[0]), exp0)

        us = wall_us(lambda: prepare(shell, sampled, requests, batch, discard))
        print(f"prepare_next_token_ids_b{num_reqs} {us:.3f}")
        print(f"# num_reqs={num_reqs} spec_tokens={NUM_SPEC} vocab={VOCAB} iters={ITERS}")


if __name__ == "__main__":
    main()
