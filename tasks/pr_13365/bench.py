#!/usr/bin/env python3
"""Benchmark PR #13365: N-gram matching optimization with Numba.

Mirrors the author's workload: "ShareGPT benchmark with 5K requests, the
cumulative overhead reduces from 54.3 sec to 1.9 sec". In the engine
NgramProposer.propose() runs ONCE PER DECODE STEP for every running request,
over the request's current context (prompt + tokens generated so far). So the
cumulative number is ~N_SEQ x output_len calls on growing prefixes, not one
call per sequence.

We import vllm/v1/spec_decode/ngram_proposer.py directly from the mounted arm
tree (it only depends on numpy / numba), build 5000 synthetic requests with a
ShareGPT-like prompt / output split, and for each request call
propose(ctx[:t], n, k) for every decode step t in [prompt_len, total_len).

baseline arm : pure-python KMP match loop.
PR arm       : numba @jit(nopython=True) kernels (first call JIT-compiles).

Metrics (one line each, `<metric> <seconds>`; parsed by parse_workload.py):
  ngram_match_cold_first_call  the very first propose() call of the process
                               (includes Numba JIT on the PR arm; the author's
                               1.9 s contains this cost once per engine start)
  ngram_match_overhead         cumulative seconds of all per-decode-step calls
                               over the 5000 requests, steady state (headline)

The workload is generated with a fixed seed regardless of arm so both arms do
exactly the same calls. SRC can be overridden with NGRAM_SRC for a host-side
smoke test; in the task image the arm tree is at /src.
"""
import importlib.util
import os
import sys
import time

import numpy as np

SRC = os.environ.get("NGRAM_SRC", "/src/vllm/v1/spec_decode/ngram_proposer.py")

# ---- workload: 5000 ShareGPT-like requests -------------------------------
N_SEQ = 5000
VOCAB = 32000
PROMPT_MIN, PROMPT_MAX = 50, 600      # prompt tokens
OUTPUT_MIN, OUTPUT_MAX = 50, 400      # decode steps per request (mean ~225)
NGRAM_N = 3          # n-gram size (match length)
K = 10               # number of follow tokens to return
WARMUP_SEQS = 50     # untimed requests after the cold call (steady-state JIT, caches)

rng = np.random.default_rng(1234)
seqs = []            # (prompt_len, full token array incl. all output tokens)
for _ in range(N_SEQ):
    p = int(rng.integers(PROMPT_MIN, PROMPT_MAX + 1))
    o = int(rng.integers(OUTPUT_MIN, OUTPUT_MAX + 1))
    arr = rng.integers(0, VOCAB, size=p + o, dtype=np.int32)
    # In ~half the requests, plant an earlier copy of a later n-gram so KMP
    # finds a match on some steps (exercises the "found" path); the rest scan
    # the whole context and miss, like most real decode steps.
    if rng.random() < 0.5 and p > NGRAM_N + K + 20:
        pos = int(rng.integers(0, p - NGRAM_N - K - 1))
        t = int(rng.integers(p, p + o))
        arr[pos:pos + NGRAM_N] = arr[t - NGRAM_N:t]
    seqs.append((p, arr))

TOTAL_CALLS = sum(len(arr) - p for p, arr in seqs)


def load_proposer(path):
    spec = importlib.util.spec_from_file_location("ngram_proposer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NgramProposer()


def run_request(proposer, p, arr):
    """one request: propose() at every decode step over the growing context"""
    propose = proposer.propose
    for t in range(p, len(arr)):
        propose(arr[:t], NGRAM_N, K)


def main():
    arm = os.environ.get("ARM", "unknown")
    proposer = load_proposer(SRC)
    # The PR arm's module does `from numba import jit`; the baseline never imports it.
    uses_numba = "numba" in sys.modules
    label = "numba" if uses_numba else "python"

    # cold: the first call of the process (JIT compile on the PR arm)
    p0, arr0 = seqs[0]
    t0 = time.perf_counter()
    proposer.propose(arr0[:p0], NGRAM_N, K)
    cold = time.perf_counter() - t0

    for p, arr in seqs[:WARMUP_SEQS]:      # untimed
        run_request(proposer, p, arr)

    t0 = time.perf_counter()
    for p, arr in seqs:
        run_request(proposer, p, arr)
    total = time.perf_counter() - t0

    print(f"ngram_match_cold_first_call {label} {cold:.6f}")
    print(f"ngram_match_overhead {label} {total:.6f}")
    print(f"# arm={arm} numba_in_use={uses_numba} calls={TOTAL_CALLS} "
          f"per_call_us={total / TOTAL_CALLS * 1e6:.2f} n_seq={N_SEQ}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
