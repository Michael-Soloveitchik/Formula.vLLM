#!/usr/bin/env python3
"""Microbench for vLLM PR #6442: [Core] Use numpy to speed up padded token processing.

The PR rewrites the penalty-token padding in `SamplingTensors.from_lists` (v0 sampler):
    pre-PR : prompt/output token lists are padded with python list arithmetic
             (`tokens + [vocab_size] * pad` per sequence) and turned into tensors with
             `torch.tensor(list_of_lists, ...)` - a slow nested-list conversion;
    PR     : `np.full` + row-wise slice assignment, then `torch.from_numpy`.
Author: ~50% sampler latency reduction at large batch (A100, penalties on, no absolute numbers).

The bench calls the arm's real `SamplingTensors.from_lists` on the author's kind of workload: a
decode batch of BATCH sequences with repetition penalties enabled, prompts of 512..1024 tokens
and 128 generated tokens each, vocab 128256 (Llama-3), device cpu. Only the CPU-side tensor
construction is timed (that is what the PR changes); pinned memory is disabled so the bench
never touches CUDA (wheel 0.5.3 pins torch 2.3 cu121, which has no sm_120 kernels). ms per
from_lists call, mean over ITERS.
Runs inside the task image: python3 /task/bench.py
"""
import random
import time

import torch

import vllm.model_executor.sampling_metadata as sm

sm.is_pin_memory_available = lambda: False      # CPU-only, identical for both arms

BATCH = 256
PROMPT_MIN, PROMPT_MAX = 512, 1024
OUTPUT_LEN = 128
VOCAB = 128256
WARMUP = 3
ITERS = 20


def make_batch(rng):
    prompt_tokens = [[rng.randrange(VOCAB) for _ in range(rng.randint(PROMPT_MIN, PROMPT_MAX))]
                     for _ in range(BATCH)]
    output_tokens = [[rng.randrange(VOCAB) for _ in range(OUTPUT_LEN)] for _ in range(BATCH)]
    return prompt_tokens, output_tokens


def main():
    rng = random.Random(0)
    prompt_tokens, output_tokens = make_batch(rng)
    kwargs = dict(
        temperatures=[1.0] * BATCH, top_ps=[1.0] * BATCH, top_ks=[VOCAB] * BATCH,
        min_ps=[0.0] * BATCH, presence_penalties=[0.0] * BATCH,
        frequency_penalties=[0.0] * BATCH, repetition_penalties=[1.1] * BATCH,
        sampling_seeds=[[rng.randrange(1 << 31)] for _ in range(BATCH)],
        sample_indices=list(range(BATCH)),
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        vocab_size=VOCAB, extra_seeds_to_generate=0,
        device=torch.device("cpu"), dtype=torch.float32,
    )

    def run():
        return sm.SamplingTensors.from_lists(**kwargs)

    t = run()
    assert t.prompt_tokens.shape == (BATCH, max(len(p) for p in prompt_tokens))
    assert t.output_tokens.shape == (BATCH, OUTPUT_LEN)
    assert int(t.prompt_tokens[0, len(prompt_tokens[0]) - 1]) == prompt_tokens[0][-1]
    for _ in range(WARMUP):
        run()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        run()
    ms = (time.perf_counter() - t0) * 1e3 / ITERS
    print(f"sampler_latency {ms:.4f}")
    print(f"# batch={BATCH} prompt={PROMPT_MIN}..{PROMPT_MAX} output={OUTPUT_LEN} vocab={VOCAB} "
          f"penalties=on device=cpu pin_memory=off iters={ITERS}")


if __name__ == "__main__":
    main()
