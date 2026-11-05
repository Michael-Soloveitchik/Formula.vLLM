# PR #31984: [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel

- field: gpu_kernels  | area: v1_attention
- author: jvlunteren
- state: MERGED  created: 2026-01-08T17:18:46Z  merged: 2026-01-10T18:13:45Z
- url: https://github.com/vllm-project/vllm/pull/31984
- labels: ready, v1

# Issue

## Problem

The 3D Triton unified attention kernel (`kernel_unified_attention_3d`) always iterated over every tile of the KV cache up to `num_tiles`, even when a sliding window was active. For sliding-window attention, each query position can only attend to keys within `SLIDING_WINDOW` positions back, so tiles entirely outside that window contribute nothing after masking. Processing them wasted compute on loads, dot products, and masking that are discarded.

## Example

For `openai/gpt-oss-20b` on an H100 at batch size 1 (which triggers the 3D kernel), output lengths of 6,400 and 12,800 tokens showed ~3-4% latency improvement once out-of-window tiles were pruned.

## Use case

Long-context autoregressive decoding with sliding-window attention models served via the Triton attention backend (`VLLM_ATTENTION_BACKEND=TRITON_ATTN`).

## Resources

GPU compute (SM cycles) and HBM bandwidth spent loading and processing KV tiles that fall outside the sliding window and are fully masked out.

## Type of users

Users serving sliding-window attention models (e.g. gpt-oss) with long sequences through the Triton attention backend, especially at batch size 1 where the 3D kernel path runs.

## Kind of inputs

Requests with large sequence/context length and a positive `SLIDING_WINDOW` much smaller than total sequence length, so many KV tiles sit outside the attendable window.

## Mechanism with the issue

In `kernel_unified_attention_3d`, the per-segment tile loop ran `for j in range(segm_idx * tiles_per_segment, min((segm_idx+1)*tiles_per_segment, num_tiles))`, covering all tiles regardless of window. The sliding-window constraint was only enforced later via masking, so out-of-window tiles were still loaded and computed, then zeroed — pure wasted work. The 2D kernel already pruned these tiles; the 3D kernel did not.

# Solution

## Fix in words

The diff adds sliding-window tile pruning to `kernel_unified_attention_3d`. Before, `tile_start=0` and `tile_end=num_tiles` were effectively fixed and the loop spanned the whole segment range. After, when `SLIDING_WINDOW > 0` and not `USE_MM_PREFIX`, it computes the Q-block's query row span (`qpos_lo`/`qpos_hi`), derives the union of allowed key positions `[context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]`, converts those to clamped tile indices `tile_start`/`tile_end`. The loop bounds become `max(segm_idx*tiles_per_segment, tile_start)` to `min((segm_idx+1)*tiles_per_segment, tile_end)`, so only tiles intersecting the window are processed. Effect: skips loads/compute on fully-masked out-of-window tiles, cutting latency ~3-4% at long output lengths.

## Author's solution

Author applies the same optimization already used in the 2D kernel to the 3D kernel, ensuring only tiles within the defined sliding window are processed. This reduces computational overhead and improves performance, with gains growing for longer sequences (~3-4% at 6,400 and 12,800 output tokens). Correctness confirmed via existing unit tests (512 passed) and gsm8k lm_eval.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/ops/triton_unified_attention.py`

### Diff
```diff
diff --git a/vllm/v1/attention/ops/triton_unified_attention.py b/vllm/v1/attention/ops/triton_unified_attention.py
index c946dbd8a2c4..345889969d3e 100644
--- a/vllm/v1/attention/ops/triton_unified_attention.py
+++ b/vllm/v1/attention/ops/triton_unified_attention.py
@@ -545,10 +545,33 @@ def kernel_unified_attention_3d(
     # this prefix can be skipped)
     num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
 
-    # iterate through tiles within current segment
+    # ---- Sliding-window tile pruning --------------------
+    # Default: keep previous global behavior
+    tile_start = 0
+    tile_end = num_tiles
+    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
+    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
+        # Query rows covered by this Q-block
+        qpos_lo = q_block_local_idx * BLOCK_Q
+        qpos_hi = tl.minimum(
+            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
+            cur_batch_query_len - 1,
+        )
+        # For sliding window, each query position q can only attend to
+        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
+        # where q_abs = context_len + q
+        # The union of allowed key positions for this Q-block is:
+        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
+        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
+        last_allowed_key = context_len + qpos_hi
+        # Convert to tile indices and clamp
+        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
+        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
+
+    # iterate through tiles (now limited to the sliding window range)
     for j in range(
-        segm_idx * tiles_per_segment,
-        min((segm_idx + 1) * tiles_per_segment, num_tiles),
+        max(segm_idx * tiles_per_segment, tile_start),
+        min((segm_idx + 1) * tiles_per_segment, tile_end),
     ):
         seq_offset = j * TILE_SIZE + offs_t
         tile_mask = seq_offset < max_seq_prefix_len

```
