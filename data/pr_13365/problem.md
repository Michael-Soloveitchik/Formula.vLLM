# Task PR #13365 — [V1][Spec Decode] Optimize N-gram matching with Numba (#13365)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **ngram_match_overhead** in `s` (seconds); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **ngram_match_overhead** (headline) — `s` (seconds), ↓ lower is better. reported 54.3 -> 1.9 s. author: cumulative propose() overhead over a 5K-request ShareGPT run (per decode step); ours: same call pattern on synthetic requests, steady state
- **ngram_match_cold_first_call** — `s` (seconds), ↓ lower is better. no number reported in the PR. first propose() call of the process: Numba JIT compile on the PR's solution (author: 'under 1 second', paid once per engine start), plain call on the baseline

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Numba-JIT N-gram matcher; CPU-only. bench.py mirrors the author's workload: 5000 ShareGPT-like requests, NgramProposer.propose() called once per decode step on the growing context (~1.1M calls, like the engine does), cumulative seconds = ngram_match_overhead. The first call of the process (Numba JIT on the PR's solution) is reported separately as ngram_match_cold_first_call.

## The problem (issue)
(from `solution.md`, quoted)

> The V1 N-gram speculative decode proposer implements KMP subarray matching in pure Python, executing per-token loops in the interpreter. At large batch sizes this per-request matching work dominates CPU overhead, adding tens of seconds of cumulative latency across many requests. The Python-level loop over context tokens is the bottleneck, not the asymptotic complexity.

### Issue — detailed

> NgramProposer._find_subarray_kmp and _kmp_lps_array ran as interpreted Python methods using Python int lists and element-wise while-loops over context_token_ids. Every token comparison incurs Python interpreter and boxing overhead, so scanning long contexts across thousands of requests accumulates large CPU time in the decode path.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff JIT-compiles the matching code with Numba: _kmp_lps_array and _find_subarray_kmp move from static methods on NgramProposer to module-level functions decorated with @jit(nopython=True), operating on numpy arrays (np.int32) instead of Python lists. The lps array becomes np.zeros(..., dtype=np.int32) and the NgramProposer.propose call delegates to the free compiled function. gpu_model_runner now instantiates the drafter only on the last PP rank and warms up the JIT by calling propose on a np.zeros(1024) dummy input at init (compilation takes under 1 sec). numba==0.60.0 is pinned in requirements-common.txt (v0.61 drops Python 3.9). Net effect: the hot matching loop runs as compiled native code, yielding a 20-30x speedup and cutting ShareGPT 5K-request overhead from 54.3 sec to 1.9 sec.

**Author's words:**

> The author optimized the N-gram matching algorithm by JIT compiling it with Numba, observing a 20-30x speedup at large batch sizes. On ShareGPT with 5K requests, cumulative overhead dropped from 54.3 sec to 1.9 sec (~2.5% of total runtime).

## Reported numbers (PR author)
- gain: 30x, 2.5% (pct)
- evidence: I've observed 20-30x speedup with large batch sizes: For ShareGPT benchmark with 5K requests, the cumulative overhead reduces from 54.3 sec to 1.9 sec, which is
- reported = numbers from the PR body (author's machine); headline ngram_match_overhead

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
