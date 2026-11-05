# Task PR #37225 — [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler (#37225)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (gold)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
The Triton top-k/top-p sampler finds the k-th largest logit per row with a ternary search over pivot values. Its convergence check needs the smallest value that is still strictly above the current pivot. The kernel instead feeds it the global minimum of the tile, so the check never triggers early and every search runs to the hard iteration cap. A second full scan per iteration recounts how many values equal that minimum, doubling the memory traffic.

### Issue — detailed
In `vllm/v1/sample/ops/topk_topp_triton.py`, `_topk_topp_kernel` sets `min_larger` from `tl.min(logits_blk2)` (the global tile min) in both the truncated-buffer path and the full-vocab path, instead of the min of values above the pivot. The termination condition therefore cannot converge and the search hits the 18-iteration cap (it should finish in roughly 3–5 iterations). After each search iteration a separate loop re-scans the same data to compute `num_min_larger`, adding one more full pass over the vocab per iteration. Both effects scale with batch size × vocab size, so large-batch, large-vocab top-k (and top-k + top-p) sampling pays 20–30 % extra latency. The top-p-only path is unaffected.

## Expert solution (reference — the merged PR)
**General:** Make `min_larger` the minimum of values strictly above the pivot so the ternary search terminates early, and fuse the count of that minimum into the same pass so each iteration scans the data once instead of twice.

**Detailed:** Added a `@triton.jit` helper `_update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel)` that computes the per-tile min over `above_mask` via `tl.min(tl.where(above_mask, data, sentinel))`, counts tile elements equal to it, and merges into the running `(min_larger, num_min_larger)` with the rule: tile min < running → replace both; equal → accumulate count; greater → keep. Both top-k paths now compute `above_0`/`above_1` masks once per tile, derive the counts above each pivot from `tl.sum(above_*)`, and call the helper to update `min_larger`/`num_min_larger` in that same loop; the redundant second scan loop is deleted. With the correct `min_larger`, the convergence condition fires after ~3–5 iterations.

**Author's words:** Fix `min_larger` in top-k ternary search to compute min of values above the pivot instead of the global min. This makes the termination condition converge correctly (~3-5 iterations instead of hitting the 18-iteration cap). Fuse the `num_min_larger` second pass into the first pass for the top-k paths, eliminating a redundant full data scan per search iteration. Top-k only: 17–28% faster; top-k + top-p: 16–25% faster; top-p only unchanged.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (gold). Your
edit is timed on the SAME workload. Beat gold on these metrics:
- **topk_whole b2048 v32k** (ms): reported 0.761 -> 0.550 ms. workload: `benchmarks/benchmark_topk_topp.py --batch-sizes 128 2048 --vocab-sizes 32768 131072`
- **topk_whole b2048 v128k** (ms): reported 2.312 -> 1.891 ms.
- **topk_topp_whole b2048 v32k** (ms): reported 1.211 -> 0.907 ms.
- **topk_topp_whole b2048 v128k** (ms): reported 3.206 -> 2.652 ms.

## Reported numbers (PR author)
- metric: benchmark_topk_topp.py Triton latency per scenario
- value: top-k only 17–28% faster, top-k + top-p 16–25% faster
- unit: ms

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than gold on the measured experiments wins. Same speed = ties gold.
