#!/usr/bin/env python3
"""Benchmark PR #13365: N-gram matching optimization with Numba.

Approach: PREFERRED path. We import the standalone module
vllm/v1/spec_decode/ngram_proposer.py directly from the mounted source
tree (it only depends on numpy / numba / typing, so no heavy vLLM import
chain) and time NgramProposer().propose() over a synthetic workload.

base arm: pure-python KMP match loop.
gold arm: numba @jit(nopython=True) kernels.

For the gold arm the FIRST propose() call triggers JIT compilation, so we
warm up once (untimed) before timing steady-state over the full workload.
We repeat the timed cumulative pass 5 times.

The workload is generated identically (fixed seed) regardless of arm so
both arms do the same work.
"""
import importlib.util
import os
import sys
import time

import numpy as np

SRC = "/src/vllm/v1/spec_decode/ngram_proposer.py"

# ---- workload: 5000 synthetic context sequences ----
N_SEQ = 5000
VOCAB = 32000
MIN_LEN = 200
MAX_LEN = 800
NGRAM_N = 3          # n-gram size (match length)
K = 10               # number of follow tokens to return

rng = np.random.default_rng(1234)
seqs = []
for _ in range(N_SEQ):
    length = int(rng.integers(MIN_LEN, MAX_LEN + 1))
    arr = rng.integers(0, VOCAB, size=length, dtype=np.int32)
    # In ~half the sequences, plant the final n-gram earlier so KMP finds
    # a match (exercises the "found" path); otherwise it scans to the end.
    if rng.random() < 0.5 and length > NGRAM_N + K + 20:
        pos = int(rng.integers(0, length - NGRAM_N - K - 1))
        arr[pos:pos + NGRAM_N] = arr[-NGRAM_N:]
    seqs.append(arr)


def load_proposer(path):
    spec = importlib.util.spec_from_file_location("ngram_proposer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.NgramProposer()


def run_workload(proposer):
    total = 0.0
    t0 = time.perf_counter()
    for arr in seqs:
        proposer.propose(arr, NGRAM_N, K)
    total = time.perf_counter() - t0
    return total


def main():
    arm = os.environ.get("ARM", "unknown")
    proposer = load_proposer(SRC)

    # Detect whether numba is actually in use (gold arm): the gold module
    # does `from numba import jit`, so importing the source registers numba
    # in sys.modules. The base (pure-python) module never imports it.
    uses_numba = "numba" in sys.modules

    # Warm up once (untimed) - this triggers numba JIT compile for gold.
    _ = run_workload(proposer)

    runs = []
    for _ in range(5):
        runs.append(run_workload(proposer))

    label = "gold" if uses_numba else "base"
    for r in runs:
        print(f"ngram_match_overhead {label} {r:.6f}")
    mean = sum(runs) / len(runs)
    print(f"# arm={arm} label={label} numba_in_use={uses_numba} "
          f"mean={mean:.6f} runs={runs}")


if __name__ == "__main__":
    main()
