#!/usr/bin/env python3
"""Microbench for vLLM PR #29826: [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor.

base: MinTokensLogitsProcessor.apply does `logits[self.logits_slice] = -float("inf")`, which
      uploads a python scalar through pageable host memory -> a device-synchronising HtoD copy
      on every decode step (breaks async-scheduling overlap; author: up to +30% TPOT).
gold: a pre-built device scalar `self.neg_inf_tensor` + `logits.index_put_(...)` -> no sync.

The processor is built with its real constructor (device cuda:0, pinned memory) and its state
is set exactly as update_state() would leave it: NUM_REQS requests each with STOP_PER_REQ stop
tokens pending min_tokens. Each timed call enqueues a small filler kernel first, so a
synchronising apply() has to wait for it (that is what the author's nsys traces show); the
gold arm just queues the index_put and returns. Wall-clock us per apply, mean over ITERS.
The PR gives no micro number (nsys traces only), so `reported` is null.
Runs inside the task image: python3 /task/bench.py
"""
import time

import torch

from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

DEVICE = torch.device("cuda:0")
NUM_REQS = 64
VOCAB = 151936          # Qwen vocab
STOP_PER_REQ = 2
WARMUP = 50
ITERS = 2000
FILLER = 4096           # filler matmul size (keeps the GPU ~busy like a decode step)


def build_processor():
    proc = MinTokensLogitsProcessor(None, DEVICE, True)
    eos_ids = list(range(VOCAB - STOP_PER_REQ, VOCAB))
    proc.min_toks = {i: (16, [], set(eos_ids)) for i in range(NUM_REQS)}
    reqs, toks = [], []
    for i in range(NUM_REQS):
        reqs.extend([i] * STOP_PER_REQ)
        toks.extend(eos_ids)
    proc.logits_slice = (proc._device_tensor(reqs, torch.int32),
                         proc._device_tensor(toks, torch.int32))
    torch.cuda.synchronize()
    return proc


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(0)
    proc = build_processor()
    logits = torch.randn(NUM_REQS, VOCAB, device=DEVICE, dtype=torch.float32)
    a = torch.randn(FILLER, FILLER, device=DEVICE, dtype=torch.bfloat16)

    out = proc.apply(logits.clone())
    torch.cuda.synchronize()
    assert torch.isinf(out[0, VOCAB - 1]) and torch.isfinite(out[0, 0])

    def wall_us(fn):
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6 / ITERS

    # 1. apply() alone on an idle GPU: the pageable copy + device sync vs one async index_put
    us_alone = wall_us(lambda: proc.apply(logits))
    print(f"min_tokens_apply {us_alone:.3f}")

    # 2. apply() right after an async filler kernel (a decode step): a synchronising apply
    #    stalls the host until the filler finishes, an async one returns immediately
    def step():
        a @ a
        proc.apply(logits)

    us_overlap = wall_us(step)
    print(f"min_tokens_apply_after_kernel {us_overlap:.3f}")
    print(f"# reqs={NUM_REQS} vocab={VOCAB} stop_per_req={STOP_PER_REQ} iters={ITERS} "
          f"filler={FILLER}x{FILLER} bf16 matmul (wall-clock per call; base syncs on the HtoD copy)")


if __name__ == "__main__":
    main()
