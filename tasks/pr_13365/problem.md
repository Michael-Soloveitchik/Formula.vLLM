# Task PR #13365 — [V1][Spec Decode] Optimize N-gram matching with Numba (#13365)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (the PR's solution)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
N-gram speculative decoding scans recent context for a repeated token pattern to guess upcoming tokens. The matching routine ran as plain interpreted Python, so it was slow. With many requests per batch, this per-step scanning accumulated into large overhead.

### Issue — detailed
The `NgramProposer.propose` method in `vllm/v1/spec_decode/ngram_proposer.py` called `_find_subarray_kmp`, a pure-Python KMP substring search over `context_token_ids`, plus `_kmp_lps_array` to build the lps (longest-prefix-suffix) table. Both were interpreted CPython loops with list operations (`lps = [0] * len(pattern)`). Python per-token loop overhead, multiplied across the batch on every decode step, made the scan cost dominate. The old code left a TODO to rewrite it in C++ for efficiency.

## Expert solution (reference — the merged PR)
**General:** The matching routine is JIT-compiled to native machine code at runtime using Numba, so the per-token loops run at near-C speed. Compilation is warmed up once at startup so the first real request does not pay the compile cost. This cuts the repeated scanning overhead by 20-30x at large batch sizes.

**Detailed:** `_kmp_lps_array` and `_find_subarray_kmp` were moved out of the `NgramProposer` class to module-level functions decorated with `@jit(nopython=True)` in `vllm/v1/spec_decode/ngram_proposer.py`, and `propose` now calls the free function `_find_subarray_kmp` directly. The lps array switched from a Python list to a typed NumPy array (`np.zeros(len(pattern), dtype=np.int32)`). `numba == 0.60.0` was added to `requirements-common.txt` (pinned because v0.61 drops Python 3.9). In `gpu_model_runner.py`, the drafter is created only on the last pipeline-parallel rank, and `self.drafter.propose(np.zeros(1024, dtype=np.int32), ...)` is called once to trigger Numba JIT compilation (noted as under 1 second) before serving.

**Author's words:** This PR optimizes the N-gram matching algorithm by JIT compiling it with Numba. I've observed 20-30x speedup with large batch sizes: For ShareGPT benchmark with 5K requests, the cumulative overhead reduces from 54.3 sec to 1.9 sec, which is ~2.5% of the entire running time.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **ngram_match_overhead** (s): reported 54.3 -> 1.9 s. workload: cumulative s

## Reported numbers (PR author)
- metric: cumulative N-gram matching overhead (ShareGPT, 5K requests); and speedup at large batch sizes
- value: 54.3 -> 1.9 (~2.5% of total runtime); 20-30
- unit: seconds; x speedup

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
