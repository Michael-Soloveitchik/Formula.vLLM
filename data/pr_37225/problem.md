# Task PR #37225 — [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler (#37225)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **topk_whole_b2048_v32768** (ms): reported 0.761 -> 0.55 ms. workload: `python3 benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20`
- **topk_whole_b2048_v131072** (ms): reported 2.312 -> 1.891 ms.
- **topk_topp_whole_b2048_v32768** (ms): reported 1.211 -> 0.907 ms.
- **topk_topp_whole_b2048_v131072** (ms): reported 3.206 -> 2.652 ms.
- **topk_whole_b128_v32768** (ms): reported 0.071 -> 0.056 ms.
- note: CUDA-only (aborts without torch.cuda). topk_whole b2048/32k 0.761->0.550ms.

## Reported numbers (PR author)
- gain: 28%, 25% (pct)
- evidence: Benchmark results (benchmark_topk_topp.py) - **top-k only: 17–28% faster** across all batch sizes and vocab sizes - **top-k + top-p combined: 16–25% faster**
- reported = PR author's table (ms), same scenario key as benchmark_topk_topp.py SUMMARY TABLE: <scenario>_b<batch>_v<vocab>

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
