# PR #37225: [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler

- field: gpu_kernels  | area: v1_sample
- author: mgoin
- state: MERGED  created: 2026-03-16T20:36:45Z  merged: 2026-03-17T18:35:18Z
- url: https://github.com/vllm-project/vllm/pull/37225
- labels: performance, ready, v1

# Issue

## Problem

The top-k ternary search in `apply_top_k_top_p_triton` computed `min_larger` as the global minimum of the tile values instead of the minimum of values strictly above the pivot. Because the termination/convergence condition depends on the smallest value that still exceeds the pivot, feeding it a global min prevented the search from converging, so it ran until the hard 18-iteration cap. Additionally, `num_min_larger` was computed in a separate second pass, re-scanning the full data for every search iteration.

## Example

A topk_whole run at batch 2048, vocab 32k took 0.761 ms before and 0.550 ms after (-28%); at vocab 128k, 2.312 ms → 1.891 ms (-18%). Search iterations dropped from the 18-iteration cap to roughly 3–5.

## Use case

Sampling logits with top-k (and top-k+top-p) filtering during token generation in vLLM's v1 sampler.

## Resources

GPU compute and HBM bandwidth — each wasted search iteration re-reads the full logits buffer/vocab, and the redundant second pass doubles reads per iteration.

## Type of users

Anyone serving with top-k or top-k+top-p sampling enabled, especially at large batch sizes and large vocabularies where per-iteration scan cost is high.

## Kind of inputs

Large batch sizes (128–2048) and large vocab sizes (32k–128k) with top-k filtering active. Top-p-only paths are unaffected.

## Mechanism with the issue

In `_topk_topp_kernel`, both the truncated-buffer path and the full-vocab path set `min_larger` from `tl.min(logits_blk2)` (global tile min) rather than the min of values above the pivot, so the ternary-search convergence check never triggered early and hit the iteration cap. A separate second loop then recomputed `num_min_larger` by re-scanning the same data, adding one more full pass per iteration.

# Solution

## Fix in words

Adds a `_update_min_larger_stats` Triton helper that, per tile, computes the min of values above the pivot via `tl.min(tl.where(above_mask, data, sentinel))` and maintains a running (min, count) across tiles with a merge rule (replace on smaller, accumulate on equal, keep on larger). Both top-k paths now compute `above_0/above_1` masks once, derive `k_pivots_num` from `tl.sum(above_*)`, and update `min_larger`/`num_min_larger` in a single fused pass, deleting the redundant second scan loop. `min_larger` is now the min strictly above the pivot, so the convergence condition terminates in ~3–5 iterations instead of 18. Net effect: correct early termination plus one fewer full data scan per iteration, yielding 17–28% top-k and 16–25% top-k+top-p speedups.

## Author's solution

Fix `min_larger` in the top-k ternary search to compute the min of values above the pivot rather than the global min, making the termination condition converge in ~3–5 iterations instead of hitting the 18-iteration cap. Fuse the `num_min_larger` second pass into the first pass for the top-k paths, eliminating a redundant full data scan per search iteration. Top-p-only path left untouched.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/sample/ops/topk_topp_triton.py`

### Diff
```diff
diff --git a/vllm/v1/sample/ops/topk_topp_triton.py b/vllm/v1/sample/ops/topk_topp_triton.py
index 050165ea5dc8..4c7c3e99d44b 100644
--- a/vllm/v1/sample/ops/topk_topp_triton.py
+++ b/vllm/v1/sample/ops/topk_topp_triton.py
@@ -67,6 +67,29 @@
 # fmt: on
 
 
+@triton.jit
+def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
+    """Update running (min, count) of values above a pivot across tiles.
+
+    Tracks the smallest value strictly above a pivot and how many times
+    it occurs.  Called once per tile per pivot; the running state is
+    carried across tiles via `min_larger` / `num_min_larger`.
+
+    Merge rule:
+      - tile min < running min  → replace both
+      - tile min == running min → accumulate count
+      - tile min > running min  → keep running values
+    """
+    tile_min = tl.min(tl.where(above_mask, data, sentinel))
+    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
+    tile_cnt = tl.sum(tile_eq)
+    is_new = tile_min < min_larger
+    is_same = tl.abs(tile_min - min_larger) < 1e-9
+    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
+    min_larger = tl.minimum(min_larger, tile_min)
+    return min_larger, num_min_larger
+
+
 @triton.jit
 def _topk_topp_kernel(
     LOGITS,
@@ -188,7 +211,10 @@ def _topk_topp_kernel(
                         min_larger_1 = float("inf")
                         num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
 
-                        # First pass: Calculate k_pivots_num and min_larger
+                        # Single fused pass: compute k_pivots_num,
+                        # min_larger, and num_min_larger together to avoid
+                        # a second data scan. See _update_min_larger_stats
+                        # for the tile-level merge logic.
                         for i in range(0, search_iters):
                             offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                 0, BLOCK_SIZE_TRUNC
@@ -198,27 +224,24 @@ def _topk_topp_kernel(
                                 BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                             )
 
-                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
-                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
-
-                            min_larger_0 = tl.minimum(min_larger_0, tl.min(logits_blk2))
-                            min_larger_1 = tl.minimum(min_larger_1, tl.min(logits_blk2))
-
-                        # Second pass: Calculate num_min_larger
-                        for i in range(0, search_iters):
-                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
-                                0, BLOCK_SIZE_TRUNC
+                            above_0 = logits_blk2 > k_pivot_0
+                            above_1 = logits_blk2 > k_pivot_1
+                            k_pivots_num_0 += tl.sum(above_0)
+                            k_pivots_num_1 += tl.sum(above_1)
+
+                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
+                                logits_blk2,
+                                above_0,
+                                min_larger_0,
+                                num_min_larger_0,
+                                float("inf"),
                             )
-                            mask_n_2 = offs_n < search_range
-                            logits_blk2 = tl.load(
-                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
-                            )
-
-                            num_min_larger_0 += tl.sum(
-                                tl.abs(logits_blk2 - min_larger_0) < 1e-9
-                            )
-                            num_min_larger_1 += tl.sum(
-                                tl.abs(logits_blk2 - min_larger_1) < 1e-9
+                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
+                                logits_blk2,
+                                above_1,
+                                min_larger_1,
+                                num_min_larger_1,
+                                float("inf"),
                             )
 
                         # Check if any of the pivots satisfy termination condition
@@ -272,26 +295,8 @@ def _topk_topp_kernel(
                         min_larger_1 = float("inf")
                         num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
 
-                        # First pass: Calculate k_pivots_num and min_larger
-                        for i in range(0, NUM_TILES):
-                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
-                            mask_n = offs_n < VOCAB_SIZE
-                            logits_blk2 = tl.load(
-                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
-                            )
-
-                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
-                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
-
-                            # Exclude -inf from min_larger to avoid
-                            # poisoning the convergence check.
-                            finite_blk2 = tl.where(
-                                logits_blk2 > -float("inf"), logits_blk2, float("inf")
-                            )
-                            min_larger_0 = tl.minimum(min_larger_0, tl.min(finite_blk2))
-                            min_larger_1 = tl.minimum(min_larger_1, tl.min(finite_blk2))
-
-                        # Second pass: Calculate num_min_larger
+                        # Single fused pass over full vocab (same approach
+                        # as the buffer path above).
                         for i in range(0, NUM_TILES):
                             offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                             mask_n = offs_n < VOCAB_SIZE
@@ -299,11 +304,24 @@ def _topk_topp_kernel(
                                 LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                             )
 
-                            num_min_larger_0 += tl.sum(
-                                tl.abs(logits_blk2 - min_larger_0) < 1e-9
+                            above_0 = logits_blk2 > k_pivot_0
+                            above_1 = logits_blk2 > k_pivot_1
+                            k_pivots_num_0 += tl.sum(above_0)
+                            k_pivots_num_1 += tl.sum(above_1)
+
+                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
+                                logits_blk2,
+                                above_0,
+                                min_larger_0,
+                                num_min_larger_0,
+                                float("inf"),
                             )
-                            num_min_larger_1 += tl.sum(
-                                tl.abs(logits_blk2 - min_larger_1) < 1e-9
+                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
+                                logits_blk2,
+                                above_1,
+                                min_larger_1,
+                                num_min_larger_1,
+                                float("inf"),
                             )
 
                         # Check if any of the pivots satisfy termination condition

```
