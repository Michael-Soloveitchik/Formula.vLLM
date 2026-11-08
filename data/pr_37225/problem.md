# Task PR #37225 — [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler (#37225)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **topk_whole_b2048_v32768** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **topk_whole_b2048_v32768** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.761 -> 0.55 ms.
- **topk_whole_b2048_v131072** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 2.312 -> 1.891 ms.
- **topk_topp_whole_b2048_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 1.211 -> 0.907 ms.
- **topk_topp_whole_b2048_v131072** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 3.206 -> 2.652 ms.
- **topk_whole_b128_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.071 -> 0.056 ms.

Workload command (run per arm): `python3 benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20`
Mechanism / note: CUDA-only (aborts without torch.cuda). topk_whole b2048/32k 0.761->0.550ms.

## The problem (issue)
(from `solution.md`, quoted)

> The top-k ternary search in `apply_top_k_top_p_triton` computed `min_larger` as the global minimum of the tile values instead of the minimum of values strictly above the pivot. Because the termination/convergence condition depends on the smallest value that still exceeds the pivot, feeding it a global min prevented the search from converging, so it ran until the hard 18-iteration cap. Additionally, `num_min_larger` was computed in a separate second pass, re-scanning the full data for every search iteration.

### Issue — detailed

> In `_topk_topp_kernel`, both the truncated-buffer path and the full-vocab path set `min_larger` from `tl.min(logits_blk2)` (global tile min) rather than the min of values above the pivot, so the ternary-search convergence check never triggered early and hit the iteration cap. A separate second loop then recomputed `num_min_larger` by re-scanning the same data, adding one more full pass per iteration.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds a `_update_min_larger_stats` Triton helper that, per tile, computes the min of values above the pivot via `tl.min(tl.where(above_mask, data, sentinel))` and maintains a running (min, count) across tiles with a merge rule (replace on smaller, accumulate on equal, keep on larger). Both top-k paths now compute `above_0/above_1` masks once, derive `k_pivots_num` from `tl.sum(above_*)`, and update `min_larger`/`num_min_larger` in a single fused pass, deleting the redundant second scan loop. `min_larger` is now the min strictly above the pivot, so the convergence condition terminates in ~3–5 iterations instead of 18. Net effect: correct early termination plus one fewer full data scan per iteration, yielding 17–28% top-k and 16–25% top-k+top-p speedups.

**Author's words:**

> Fix `min_larger` in the top-k ternary search to compute the min of values above the pivot rather than the global min, making the termination condition converge in ~3–5 iterations instead of hitting the 18-iteration cap. Fuse the `num_min_larger` second pass into the first pass for the top-k paths, eliminating a redundant full data scan per search iteration. Top-p-only path left untouched.

## Reported numbers (PR author)
- gain: 28%, 25% (pct)
- evidence: Benchmark results (benchmark_topk_topp.py) - **top-k only: 17–28% faster** across all batch sizes and vocab sizes - **top-k + top-p combined: 16–25% faster**
- reported = PR author's table (ms), same scenario key as benchmark_topk_topp.py SUMMARY TABLE: <scenario>_b<batch>_v<vocab>

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
