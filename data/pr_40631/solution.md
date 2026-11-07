# PR #40631: [Refactor] Unify 2D/3D kernels in triton_unified_attention (#40631)

# Issue

## Original PR description (captured source)

This PR is the refactor-only half of the split that @tdoublep and @bringlein requested on #39074. There are no new features and no behavior changes here. The goal is simply to remove duplication by collapsing two nearly identical attention kernels into one.

On main, triton_unified_attention.py contains two kernels that are almost the same. kernel_unified_attention_2d processes the full sequence in a single tile loop and writes directly to the output buffer, while kernel_unified_attention_3d limits that loop to a single segment and writes per-segment partials that reduce_segments combines later. In practice, the two implementations only differ in a few places around sequence scoping and the epilogue, even though each one is about 400 lines long.

This PR merges those paths into a single kernel, gated by IS_3D: tl.constexpr. Because Triton only traces the branch that is actually taken, each concrete launch still compiles down to the same code we would have gotten from separate layout-specific kernels.

The rest of the diff is mostly helper extraction into a new triton_attention_helpers.py. The prologue, online softmax bookkeeping, tile-loop bound computation, mask construction (ALiBi, softcap, qq_bias, sliding window, mm-prefix, chunked attention), and the scalar writes used for segment reduction were previously duplicated across both kernels. Now each of those pieces lives in one place, which means future fixes only need to land once and will automatically apply to both paths.

In terms of file size, triton_unified_attention.py goes from 1268 lines to 752 lines, for a net reduction of 237 lines overall after accounting for the new helpers file.

This is not a duplicate of #39074. It is the first half of that work, split out at reviewer request. The original PR will be closed once this refactor PR and its stacked follow-up feature PR have both landed.

## Full discussion

See [40631.md](./40631.md) and [40631.json](./40631.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/40631
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/40631
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `9fb6552371eb8f6472adc585823537cce705044b37540ed5dd2e2e564f306840`

Existing semantic notes in [40631.sem.json](./40631.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/v1/attention/ops/triton_attention_helpers.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`

### Diff

```diff
diff --git a/vllm/v1/attention/ops/triton_attention_helpers.py b/vllm/v1/attention/ops/triton_attention_helpers.py
new file mode 100644
index 000000000000..6ed50f6a2df2
--- /dev/null
+++ b/vllm/v1/attention/ops/triton_attention_helpers.py
@@ -0,0 +1,383 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""Shared ``@triton.jit`` helpers used by the unified attention kernel
+and ``reduce_segments``.
+
+These are plain attention-loop helpers — mask building, ALiBi / QQ-bias
+score post-processing, online-softmax bookkeeping, tile-loop bounds,
+sequence lookup — extracted so the 2D and 3D paths of the unified
+kernel (and any future consumer) share a single implementation.
+"""
+
+from __future__ import annotations
+
+from vllm.triton_utils import tl, triton
+
+# ===========================================================================
+# Scalar helpers (reused by every kernel + reduce_segments)
+# ===========================================================================
+
+
+@triton.jit
+def cdiv_fn(x, y):
+    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
+    return (x + y - 1) // y
+
+
+@triton.jit
+def apply_softcap(S, x):
+    """Softcap (aka tanh-style clamp) used to bound attention scores.
+
+    ``x * tanh(S / x)`` rewritten to avoid a direct ``tanh`` call.
+    """
+    Sdiv = S / x
+    p1 = tl.exp(Sdiv)
+    p2 = tl.exp(-Sdiv)
+    return x * (p1 - p2) / (p1 + p2)
+
+
+# ===========================================================================
+# Attention loop
+# ===========================================================================
+
+
+@triton.jit
+def resolve_seq_and_query_len(
+    query_start_len_ptr,
+    seq_lens_ptr,
+    q_block_global_idx,
+    num_seqs,
+    BLOCK_Q: tl.constexpr,
+):
+    """Resolve the (sequence, q-block-within-sequence) pair and load the
+    per-sequence lengths.
+
+    Shared across every attention kernel — the ``q_block_global_idx``
+    program id indexes into the flattened ``(seq, q_block_in_seq)``
+    space, and a binary search over ``query_start_len_ptr`` recovers
+    the (seq, local-q-block) pair.
+
+    Returns ``(seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
+    cur_batch_query_len, seq_len)``.  Callers must still early-return
+    when ``q_block_local_idx * BLOCK_Q >= cur_batch_query_len`` (Triton
+    helpers cannot return from the caller).
+    """
+    # find_seq_idx is defined below; forward use is fine inside @triton.jit.
+    seq_idx = find_seq_idx(
+        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
+    )
+    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
+    q_block_local_idx = q_block_global_idx - q_block_start_idx
+    cur_start = tl.load(query_start_len_ptr + seq_idx)
+    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
+    cur_batch_query_len = cur_stop - cur_start
+    seq_len = tl.load(seq_lens_ptr + seq_idx)
+    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len
+
+
+@triton.jit
+def find_seq_idx(
+    query_start_len_ptr,
+    target_idx,
+    num_seqs,
+    BLOCK_Q: tl.constexpr,
+    use_q_block_mode: tl.constexpr,
+):
+    """Binary search over the cumulative query-length prefix.
+
+    When ``use_q_block_mode`` is True, the prefix values are reshaped
+    into units of ``BLOCK_Q`` plus one entry per boundary — matching
+    the q-block grid laid out by the attention kernels.  When False
+    we search the plain cumulative-length prefix (used by
+    ``reduce_segments`` which iterates over raw query tokens).
+    """
+    left: tl.int32 = 0
+    right = num_seqs
+    while left < right:
+        mid = (left + right) // 2
+        val = tl.load(query_start_len_ptr + mid)
+        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
+
+        if mid_val <= target_idx:
+            left = mid + 1
+        else:
+            right = mid
+
+    return left - 1
+
+
+@triton.jit
+def init_softmax_M(
+    sink_ptr,
+    query_offset_1,
+    query_mask_1,
+    segm_idx_or_0,
+    BLOCK_M: tl.constexpr,
+    USE_SINKS: tl.constexpr,
+    IS_3D: tl.constexpr,
+):
+    """Initial row-max ``M`` for the online softmax.
+
+    Without sinks: ``-inf``.  With sinks: load the per-head sink bias
+    once.  In 3D mode only segment 0 loads — ``reduce_segments`` adds
+    the sink contribution exactly once across segments, so other
+    segments must start from ``-inf``.
+
+    ``segm_idx_or_0`` is the 3D segment index or 0 for 2D (caller
+    passes ``0`` when ``IS_3D`` is False).
+    """
+    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
+    if USE_SINKS:
+        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
+        if load_sinks:
+            M = tl.load(
+                sink_ptr + query_offset_1,
+                mask=query_mask_1,
+                other=float("-inf"),
+            ).to(tl.float32)
+    return M
+
+
+@triton.jit
+def compute_tile_loop_bounds(
+    context_len,
+    seq_len,
+    cur_batch_query_len,
+    q_block_local_idx,
+    segm_idx_or_0,
+    tiles_per_segment_or_0,
+    TILE_SIZE: tl.constexpr,
+    BLOCK_M: tl.constexpr,
+    BLOCK_Q: tl.constexpr,
+    num_queries_per_kv: tl.constexpr,
+    SLIDING_WINDOW: tl.constexpr,
+    USE_MM_PREFIX: tl.constexpr,
+    IS_3D: tl.constexpr,
+    CHUNK_LOOKBACK: tl.constexpr = -1,
+    CHUNK_SIZE: tl.constexpr = -1,
+):
+    """Compute the tile-loop bounds ``(loop_lo, loop_hi)`` and the
+    derived ``max_seq_prefix_len`` used for per-tile masking.
+
+    Combines three concerns into one helper:
+
+    1. Longest prefix spanned by any query token in this q-block.
+       Clamped to ``seq_len`` (causal) or extended to it when
+       mm_prefix is active (bidirectional ranges can reach past the
+       causal prefix).
+    2. Sliding-window pruning: narrows ``[tile_start, tile_end)`` to
+       only tiles that can contain an allowed key under SWA.
+    3. 3D scoping: when ``IS_3D`` is True, further narrows to the
+       segment's slice via ``(segm_idx * tiles_per_segment,
+       (segm_idx + 1) * tiles_per_segment)``.
+    """
+    # compute the length of the longest sequence prefix spanned by any
+    # query token in the current q_block (q_block_local_idx)
+    max_seq_prefix_len = (
+        context_len
+        + q_block_local_idx * BLOCK_Q
+        + (BLOCK_M - 1) // num_queries_per_kv
+        + 1
+    )
+    if USE_MM_PREFIX:
+        # image bidirectional attention ranges require a full range
+        # including q_block padding to make sure doc mask is correct
+        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
+    else:
+        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
+
+    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
+
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
+        q_abs = context_len + qpos_lo
+        if CHUNK_LOOKBACK > -1:
+            # Chunked attention: align lower bound to the start of the
+            # lookback'th previous chunk.
+            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
+        else:
+            first_allowed_key = q_abs - SLIDING_WINDOW + 1
+        last_allowed_key = context_len + qpos_hi
+        # Convert to tile indices and clamp
+        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
+        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
+
+    if IS_3D:
+        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
+        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
+    else:
+        loop_lo = tile_start
+        loop_hi = tile_end
+
+    return loop_lo, loop_hi, max_seq_prefix_len
+
+
+@triton.jit
+def store_segm_reduce_scalars(
+    segm_max_ptr,
+    segm_expsum_ptr,
+    query_offset_0,
+    query_offset_1,
+    segm_idx,
+    M,
+    L,
+    query_mask_0,
+    query_mask_1,
+    num_query_heads: tl.constexpr,
+    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
+):
+    """Store per-segment ``M`` and ``L`` for ``reduce_segments`` to
+    combine into the final softmax.
+
+    Shared across every 3D attention epilogue; the per-token output
+    stripes are mode-specific (flat / 2-stream split / 4-stream split)
+    and stay inlined.
+    """
+    segm_offset = (
+        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
+        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
+        + segm_idx
+    )
+    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
+    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)
+
+
+@triton.jit
+def compute_kv_seq_mask(
+    query_abs_pos,
+    seq_offset,
+    seq_idx,
+    mm_prefix_range_ptr,
+    SLIDING_WINDOW: tl.constexpr,
+    USE_MM_PREFIX: tl.constexpr,
+    MAX_MM_RANGES: tl.constexpr,
+    CHUNK_LOOKBACK: tl.constexpr = -1,
+    CHUNK_SIZE: tl.constexpr = -1,
+):
+    """Build the KV mask for one tile.
+
+    Causal (key <= query) by default; AND-ed with either chunked
+    attention (``CHUNK_LOOKBACK >= 0``) or sliding window
+    (``SLIDING_WINDOW > 0``); OR-ed with the bidirectional ranges from
+    ``mm_prefix_range`` when PrefixLM / multimodal attention is active.
+    Order matches FlexAttention: ``(causal AND window) OR mm_prefix``.
+    Chunked attention takes precedence over sliding window when both
+    are non-default — the launcher zeros ``CHUNK_LOOKBACK`` whenever
+    sliding window is disabled.
+    """
+    # Compute attention mask: causal by default (key <= query)
+    seq_mask = seq_offset[None, :] <= query_abs_pos
+
+    # Apply sliding window / chunked attention to base mask
+    # BEFORE mm_prefix OR.
+    # Order must match FlexAttention:
+    #   (causal AND sliding_window) OR mm_prefix
+    if CHUNK_LOOKBACK > -1:
+        seq_mask = seq_mask & (
+            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
+            <= CHUNK_LOOKBACK
+        )
+    elif SLIDING_WINDOW > 0:
+        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
+
+    # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
+    # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
+    if USE_MM_PREFIX:
+        for i in range(MAX_MM_RANGES):
+            range_start = tl.load(
+                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
+            )
+            range_end = tl.load(
+                mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
+            )
+            is_valid = range_start < range_end
+            q_in_range = (
+                (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
+            )
+            k_in_range = (
+                (seq_offset[None, :] >= range_start)
+                & (seq_offset[None, :] <= range_end)
+                & is_valid
+            )
+            seq_mask |= q_in_range & k_in_range
+    return seq_mask
+
+
+@triton.jit
+def apply_alibi_to_score(
+    S,
+    alibi_slope,
+    seq_offset,
+    context_len,
+    query_pos,
+    USE_ALIBI_SQRT: tl.constexpr,
+):
+    """Add the ALiBi positional bias (linear or sqrt variant) to S in-place."""
+    if USE_ALIBI_SQRT:
+        relative_pos = seq_offset - (context_len + query_pos[:, None])
+        alibi_offset = tl.where(
+            relative_pos <= 0,
+            -tl.sqrt((-relative_pos).to(tl.float32)),
+            0.0,
+        )
+    else:
+        alibi_offset = seq_offset - context_len
+    return S + alibi_slope[:, None] * alibi_offset
+
+
+@triton.jit
+def load_qq_bias_tile(
+    qq_bias_row_ptrs,
+    seq_offset,
+    context_len,
+    qq_bias_stride_0,
+):
+    """Load the qq-bias slice for keys that correspond to query rows."""
+    key_rel_pos = seq_offset - context_len
+    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
+    return tl.load(
+        qq_bias_row_ptrs + key_rel_pos[None, :],
+        mask=is_query_key[None, :],
+        other=0.0,
+    )
+
+
+@triton.jit
+def softmax_step(S, M, L):
+    """Online softmax update for one tile.
+
+    Returns ``(M_new, L_new, P, alpha)``.  Caller is responsible for
+    rescaling its accumulator(s) by ``alpha[:, None]`` — done outside so
+    kernels with a different number / shape of accumulators can reuse
+    the same step.
+    """
+    # compute running maximum
+    # m_j : (BLOCK_M,)
+    m_j = tl.maximum(M, tl.max(S, axis=1))
+    # For sliding window there's a chance the max is -inf due to masking of
+    # the entire row. In this case we need to set m_j 0 to avoid NaN
+    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
+    # P : (BLOCK_M, TILE_SIZE)
+    P = tl.exp(S - m_j[:, None])
+    # l_j : (BLOCK_M,)
+    l_j = tl.sum(P, axis=1)
+    # alpha : (BLOCK_M, )
+    alpha = tl.exp(M - m_j)
+    # update constants
+    L_new = L * alpha + l_j
+    return m_j, L_new, P, alpha
diff --git a/vllm/v1/attention/ops/triton_unified_attention.py b/vllm/v1/attention/ops/triton_unified_attention.py
index 285c7b263beb..1774f1b30570 100644
--- a/vllm/v1/attention/ops/triton_unified_attention.py
+++ b/vllm/v1/attention/ops/triton_unified_attention.py
@@ -7,12 +7,27 @@
 #  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
 #  - Thomas Parnell <tpa@zurich.ibm.com>
 
+from typing import Any
+
 import torch
 
 import vllm.envs as envs
 from vllm.logger import init_logger
 from vllm.platforms import current_platform
 from vllm.triton_utils import tl, triton
+from vllm.v1.attention.ops.triton_attention_helpers import (
+    apply_alibi_to_score,
+    apply_softcap,
+    cdiv_fn,
+    compute_kv_seq_mask,
+    compute_tile_loop_bounds,
+    find_seq_idx,
+    init_softmax_M,
+    load_qq_bias_tile,
+    resolve_seq_and_query_len,
+    softmax_step,
+    store_segm_reduce_scalars,
+)
 from vllm.v1.kv_cache_interface import KVQuantMode
 
 logger = init_logger(__name__)
@@ -21,114 +36,53 @@
 
 
 @triton.jit
-def cdiv_fn(x, y):
-    return (x + y - 1) // y
-
-
-@triton.jit
-def apply_softcap(S, x):
-    Sdiv = S / x
-    p1 = tl.exp(Sdiv)
-    p2 = tl.exp(-Sdiv)
-    return x * (p1 - p2) / (p1 + p2)
-
-
-@triton.jit
-def _prepare_kv_tile(
-    data,
-    Q,
-    tensor_scale,
-    scale_cache_ptr,
-    physical_block_idx,
-    seq_offset,
-    kv_head_idx,
-    stride_s_blk,
-    stride_s_slot,
-    stride_s_head,
-    tile_mask,
-    BLOCK_SIZE: tl.constexpr,
-    KV_QUANT_MODE: tl.constexpr,
-):
-    """Prepare a loaded KV tile for attention computation.
+def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
+    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.
 
-    Casts the raw KV data to Q's dtype and loads per-token-head scales
-    when applicable:
+    Modes handled inside the core kernel:
 
-    - ``KV_QUANT_MODE == 0``: cast only (no-op for bf16/fp16).
-    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize inline
-      using the tensor-wide scale.
-    - ``KV_QUANT_MODE >= 2`` (per-token-head int8/fp8): cast to Q's
-      dtype and return per-head scales separately — the caller applies
-      them after the dot product for better numerical efficiency.
-
-    Returns ``(data, token_head_scales)``.  *token_head_scales* is only
-    meaningful when ``KV_QUANT_MODE >= 2``; callers gate its use on
-    the same constexpr so the compiler eliminates dead code.
+    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
+      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
+      their scales separately on S/P inside the loop.
+    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
+      tensor-wide scale.
     """
-    # KV_QUANT_MODE values: 0=none, 1=fp8 per-tensor,
-    #                       2=int8 per-token-head, 3=fp8 per-token-head
-
-    # Placeholder scales (float32) — never read when KV_QUANT_MODE < 2.
-    unused_scales = tile_mask.to(tl.float32)
-
-    if KV_QUANT_MODE == 1:  # FP8 per-tensor
+    if KV_QUANT_MODE == 1:
         if Q.dtype.is_fp8():
-            return data.to(Q.dtype), unused_scales
-        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype), unused_scales
-    if KV_QUANT_MODE >= 2:  # per-token-head (int8 or fp8)
-        scale_idx = (
-            physical_block_idx * stride_s_blk
-            + (seq_offset % BLOCK_SIZE) * stride_s_slot
-            + kv_head_idx * stride_s_head
-        )
-        token_head_scales = tl.load(
-            scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
-        )
-        return data.to(Q.dtype), token_head_scales
-    # .to(Q.dtype) is a no-op when data is already Q's type (bf16/fp16),
-    # but required so Triton sees consistent return types across branches.
-    return data.to(Q.dtype), unused_scales
+            return data.to(Q.dtype)
+        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
+    return data.to(Q.dtype)
 
 
 @triton.jit
-def find_seq_idx(
-    query_start_len_ptr,
-    target_idx,
-    num_seqs,
-    BLOCK_Q: tl.constexpr,
-    use_q_block_mode: tl.constexpr,
-):
-    left: tl.int32 = 0
-    right = num_seqs
-    while left < right:
-        mid = (left + right) // 2
-        val = tl.load(query_start_len_ptr + mid)
-        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
-
-        if mid_val <= target_idx:
-            left = mid + 1
-        else:
-            right = mid
-
-    return left - 1
-
-
-@triton.jit
-def kernel_unified_attention_2d(
-    output_ptr,  # [num_tokens, num_query_heads, head_size]
-    query_ptr,  # [num_tokens, num_query_heads, head_size]
-    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
-    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
-    sink_ptr,  # [num_query_heads]
-    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
-    seq_lens_ptr,  # [num_seqs]
-    alibi_slopes_ptr,  # [num_query_heads]
-    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
-    scale,  # float32
-    k_scale,  # float32
-    v_scale,  # float32
-    out_scale,  # float32
-    softcap,  # float32
+def kernel_unified_attention(
+    # Output destinations.  In 2D mode we write the final result into
+    # ``output_ptr``; in 3D mode we write per-segment partials into the
+    # three ``segm_*`` tensors and ``output_ptr`` is unused (callers may
+    # pass any non-null pointer).
+    output_ptr,
+    segm_output_ptr,
+    segm_max_ptr,
+    segm_expsum_ptr,
+    # Inputs
+    query_ptr,
+    key_cache_ptr,
+    value_cache_ptr,
+    sink_ptr,
+    block_tables_ptr,
+    seq_lens_ptr,
+    alibi_slopes_ptr,
+    qq_bias_ptr,
+    # Per-(token, head) scale caches (used iff KV_QUANT_MODE in {2, 3}).
+    # For other modes callers may pass any non-null pointer.
+    k_scale_cache_ptr,
+    v_scale_cache_ptr,
+    # Scalars
+    scale,
+    k_scale,
+    v_scale,
+    out_scale,
+    softcap,
     num_query_heads: tl.constexpr,  # int
     num_queries_per_kv: tl.constexpr,  # int
     block_table_stride: tl.int64,  # int
@@ -149,7 +103,7 @@ def kernel_unified_attention_2d(
     SLIDING_WINDOW: tl.constexpr,  # int
     USE_MM_PREFIX: tl.constexpr,  # bool
     MAX_MM_RANGES: tl.constexpr,  # int
-    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
+    mm_prefix_range_ptr,
     stride_k_cache_0: tl.int64,  # int
     stride_k_cache_1: tl.int64,  # int
     stride_k_cache_2: tl.int64,  # int
@@ -158,47 +112,61 @@ def kernel_unified_attention_2d(
     stride_v_cache_1: tl.int64,  # int
     stride_v_cache_2: tl.int64,  # int
     stride_v_cache_3: tl.constexpr,  # int
-    query_start_len_ptr,  # [num_seqs+1]
-    BLOCK_Q: tl.constexpr,  # int
+    stride_ks_blk: tl.int64,
+    stride_ks_slot: tl.int64,
+    stride_ks_head: tl.int64,
+    stride_vs_blk: tl.int64,
+    stride_vs_slot: tl.int64,
+    stride_vs_head: tl.int64,
+    query_start_len_ptr,
+    BLOCK_Q: tl.constexpr,
     num_seqs: tl.int32,
-    BLOCK_M: tl.constexpr,  # int
-    USE_FP8: tl.constexpr,  # bool
-    # KV cache quantization: 0=none, 1=fp8, 2=per-token-head
+    BLOCK_M: tl.constexpr,
+    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
+    USE_FP8: tl.constexpr,
+    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
+    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
+    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
+    # per-segment partials, finalized by ``reduce_segments``.
+    IS_3D: tl.constexpr,
+    # KV cache quantization mode handled inside this kernel via constexpr
+    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
+    # FP8_PER_TOKEN_HEAD (3).
     KV_QUANT_MODE: tl.constexpr = 0,
     FP8_MIN: tl.constexpr = float8_info.min,
     FP8_MAX: tl.constexpr = float8_info.max,
-    # Per-token-head scale caches (KV_QUANT_MODE >= 2)
-    # Shape: [num_blocks, block_size, num_kv_heads]
-    k_scale_cache_ptr=None,
-    v_scale_cache_ptr=None,
-    stride_ks_blk=0,
-    stride_ks_slot=0,
-    stride_ks_head=0,
-    stride_vs_blk=0,
-    stride_vs_slot=0,
-    stride_vs_head=0,
+    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
+    # chunked masking (used by Gemma3 block-local layers); takes precedence
+    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
     CHUNK_LOOKBACK: tl.constexpr = -1,
     CHUNK_SIZE: tl.constexpr = -1,
 ):
+    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2
+
     q_block_global_idx = tl.program_id(0)
     kv_head_idx = tl.program_id(1)
-
-    seq_idx = find_seq_idx(
-        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
+    segm_idx = tl.program_id(2) if IS_3D else 0
+
+    (
+        seq_idx,
+        q_block_local_idx,
+        cur_batch_in_all_start_index,
+        cur_batch_query_len,
+        seq_len,
+    ) = resolve_seq_and_query_len(
+        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
     )
 
-    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
-
-    q_block_local_idx = q_block_global_idx - q_block_start_idx
-
-    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
-    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
-
-    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
-
     if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
         return
 
+    if IS_3D:
+        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
+        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
+            return
+    else:
+        tiles_per_segment = 0
+
     offs_m = tl.arange(0, BLOCK_M)
     offs_d = tl.arange(0, HEAD_SIZE_PADDED)
     offs_t = tl.arange(0, TILE_SIZE)
@@ -225,88 +193,43 @@ def kernel_unified_attention_2d(
 
     block_table_offset = seq_idx * block_table_stride
 
-    if not USE_SINKS:
-        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
-    else:
-        M = tl.load(
-            sink_ptr + query_offset_1,
-            mask=query_mask_1,
-            other=float("-inf"),
-        ).to(dtype=tl.float32)
-
+    M = init_softmax_M(
+        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
+    )
     L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
+    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
     acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
 
-    # sequence len for this particular sequence
-    seq_len = tl.load(seq_lens_ptr + seq_idx)
-
-    # context length for this particular sequences
     context_len = seq_len - cur_batch_query_len
 
-    # alibi slope for this head
     if USE_ALIBI_SLOPES:
         alibi_slope = tl.load(
             alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
         )
 
-    # query-query attention bias
     if USE_QQ_BIAS:
-        qq_bias_row_ptrs = (
-            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
-        )  # shape: [BLOCK_M]
-
-    # compute the length of the longest sequence prefix spanned by any
-    # query token in the current q_block (q_block_local_idx)
-    max_seq_prefix_len = (
-        context_len
-        + q_block_local_idx * BLOCK_Q
-        + (BLOCK_M - 1) // num_queries_per_kv
-        + 1
+        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
+
+    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
+        context_len,
+        seq_len,
+        cur_batch_query_len,
+        q_block_local_idx,
+        segm_idx,
+        tiles_per_segment,
+        TILE_SIZE,
+        BLOCK_M,
+        BLOCK_Q,
+        num_queries_per_kv,
+        SLIDING_WINDOW,
+        USE_MM_PREFIX,
+        IS_3D,
+        CHUNK_LOOKBACK,
+        CHUNK_SIZE,
     )
 
-    if USE_MM_PREFIX:
-        # image bidirectional attention ranges require a full range
-        # including q_block padding to make sure doc mask is correct
-        max_seq_prefix_len = tl.maximum(max_seq_prefix_len, seq_len)
-    else:
-        # adjust for potential padding in the last q_block by considering the
-        # actual sequence length
-        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
-
-    # calculate the number of tiles that need to be processed to
-    # cover the longest sequence prefix (due to causal masking, tiles beyond
-    # this prefix can be skipped)
-    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
-
-    # ---- Sliding-window tile pruning --------------------
-    # Default: keep previous global behavior
-    tile_start = 0
-    tile_end = num_tiles
-    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
-    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
-        # Query rows covered by this Q-block
-        qpos_lo = q_block_local_idx * BLOCK_Q
-        qpos_hi = tl.minimum(
-            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
-            cur_batch_query_len - 1,
-        )
-        # For sliding window, each query position q can only attend to
-        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
-        # where q_abs = context_len + q
-        # The union of allowed key positions for this Q-block is:
-        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
-        q_abs = context_len + qpos_lo
-        if CHUNK_LOOKBACK > -1:
-            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
-        else:
-            first_allowed_key = q_abs - SLIDING_WINDOW + 1
-        last_allowed_key = context_len + qpos_hi
-        # Convert to tile indices and clamp
-        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
-        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
-
     # iterate through tiles (now limited to the sliding window range)
-    for j in range(tile_start, tile_end):
+    for j in range(loop_lo, loop_hi):
         seq_offset = j * TILE_SIZE + offs_t
         tile_mask = seq_offset < max_seq_prefix_len
 
@@ -320,107 +243,64 @@ def kernel_unified_attention_2d(
             + offs_d[None, :] * stride_v_cache_3
             + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
         )
-
         k_offset = (
             physical_block_idx[None, :] * stride_k_cache_0
             + kv_head_idx * stride_k_cache_2
             + offs_d[:, None] * stride_k_cache_3
             + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
         )
-
         # K : (HEAD_SIZE, TILE_SIZE)
         K_load = tl.load(
             key_cache_ptr + k_offset,
             mask=dim_mask[:, None] & tile_mask[None, :],
             other=0.0,
         )
-        K, k_token_head_scales = _prepare_kv_tile(
-            K_load,
-            Q,
-            k_scale,
-            k_scale_cache_ptr,
-            physical_block_idx,
-            seq_offset,
-            kv_head_idx,
-            stride_ks_blk,
-            stride_ks_slot,
-            stride_ks_head,
-            tile_mask,
-            BLOCK_SIZE,
-            KV_QUANT_MODE,
-        )
-
+        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
         # V : (TILE_SIZE, HEAD_SIZE)
         V_load = tl.load(
             value_cache_ptr + v_offset,
             mask=dim_mask[None, :] & tile_mask[:, None],
             other=0.0,
         )
-        V, v_token_head_scales = _prepare_kv_tile(
-            V_load,
-            Q,
-            v_scale,
-            v_scale_cache_ptr,
-            physical_block_idx,
-            seq_offset,
-            kv_head_idx,
-            stride_vs_blk,
-            stride_vs_slot,
-            stride_vs_head,
-            tile_mask,
-            BLOCK_SIZE,
-            KV_QUANT_MODE,
-        )
+        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)
+
+        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
+        if USE_PER_TOKEN_HEAD_SCALES:
+            scale_idx = (
+                physical_block_idx * stride_ks_blk
+                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
+                + kv_head_idx * stride_ks_head
+            )
+            k_token_head_scales = tl.load(
+                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
+            )
+            v_scale_idx = (
+                physical_block_idx * stride_vs_blk
+                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
+                + kv_head_idx * stride_vs_head
+            )
+            v_token_head_scales = tl.load(
+                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
+            )
 
-        # Compute attention mask: causal by default (key <= query)
         query_abs_pos = context_len + query_pos[:, None]
-        seq_mask = seq_offset[None, :] <= query_abs_pos
-
-        # Apply sliding window / chunked attention to base mask
-        # BEFORE mm_prefix OR.
-        # Order must match FlexAttention:
-        #   (causal AND sliding_window) OR mm_prefix
-        if CHUNK_LOOKBACK > -1:
-            seq_mask = seq_mask & (
-                (
-                    (context_len + query_pos[:, None]) // CHUNK_SIZE
-                    - (seq_offset[None, :] // CHUNK_SIZE)
-                )
-                <= CHUNK_LOOKBACK
-            )
-        elif SLIDING_WINDOW > 0:
-            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
-
-        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
-        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
-        if USE_MM_PREFIX:
-            for i in range(MAX_MM_RANGES):
-                range_start = tl.load(
-                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
-                )
-                range_end = tl.load(
-                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
-                )
-
-                is_valid = range_start < range_end
-                q_in_range = (
-                    (query_abs_pos >= range_start)
-                    & (query_abs_pos <= range_end)
-                    & is_valid
-                )
-                k_in_range = (
-                    (seq_offset[None, :] >= range_start)
-                    & (seq_offset[None, :] <= range_end)
-                    & is_valid
-                )
-                seq_mask |= q_in_range & k_in_range
+        seq_mask = compute_kv_seq_mask(
+            query_abs_pos,
+            seq_offset,
+            seq_idx,
+            mm_prefix_range_ptr,
+            SLIDING_WINDOW,
+            USE_MM_PREFIX,
+            MAX_MM_RANGES,
+            CHUNK_LOOKBACK,
+            CHUNK_SIZE,
+        )
 
         # S : (BLOCK_M, TILE_SIZE)
         S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
-
-        # Per-token-head quant: fuse softmax_scale with per-head k_scale
-        # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
-        if KV_QUANT_MODE >= 2:
+        if USE_PER_TOKEN_HEAD_SCALES:
+            # Per-token-head quant: fuse softmax_scale with per-head k_scale
+            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
             S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
         else:
             S += scale * tl.dot(Q, K)
@@ -433,494 +313,76 @@ def kernel_unified_attention_2d(
         )
 
         if USE_ALIBI_SLOPES:
-            if USE_ALIBI_SQRT:
-                relative_pos = seq_offset - (context_len + query_pos[:, None])
-                alibi_offset = tl.where(
-                    relative_pos <= 0,
-                    -tl.sqrt((-relative_pos).to(tl.float32)),
-                    0.0,
-                )
-            else:
-                alibi_offset = seq_offset - context_len
-            S += alibi_slope[:, None] * alibi_offset
+            S = apply_alibi_to_score(
+                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
+            )
 
         if USE_QQ_BIAS:
-            # compute key positions relative to query section
-            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
-            # load bias only for keys that correspond to queries
-            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
-            qq_bias = tl.load(
-                qq_bias_row_ptrs + key_rel_pos[None, :],
-                mask=is_query_key[None, :],  # avoid OOB for context keys
-                other=0.0,
+            S += load_qq_bias_tile(
+                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
             )
-            S += qq_bias
-
-        # compute running maximum
-        # m_j : (BLOCK_M,)
-        m_j = tl.maximum(M, tl.max(S, axis=1))
-
-        # For sliding window there's a chance the max is -inf due to masking of
-        # the entire row. In this case we need to set m_j 0 to avoid NaN
-        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
-
-        # P : (BLOCK_M, TILE_SIZE)
-        P = tl.exp(S - m_j[:, None])
-
-        # l_j : (BLOCK_M,)
-        l_j = tl.sum(P, axis=1)
 
-        # alpha : (BLOCK_M, )
-        alpha = tl.exp(M - m_j)
-
-        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
+        M, L, P, alpha = softmax_step(S, M, L)
         acc = acc * alpha[:, None]
 
-        # update constants
-        L = L * alpha + l_j
-        M = m_j
-
         if SLIDING_WINDOW:
             qpos_lo = q_block_local_idx * BLOCK_Q
             V = tl.where(
-                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
+                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
+                V,
+                0.0,
             )
-
-        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
-        # Per-token-head quant: apply v_scale to P instead of V.
-        if KV_QUANT_MODE >= 2:
+        if USE_PER_TOKEN_HEAD_SCALES:
+            # Per-token-head quant: apply v_scale to P instead of V.
             P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
             acc += tl.dot(P_v, V)
         else:
             acc += tl.dot(P.to(V.dtype), V)
 
-    # epilogue
-    acc = acc / L[:, None]
-    if USE_FP8:
-        acc = acc * tl.load(out_scale)
-        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
-
-    output_offset = (
-        query_offset_0[:, None] * output_stride_0
-        + query_offset_1[:, None] * output_stride_1
-        + offs_d[None, :]
-    )
-
-    tl.store(
-        output_ptr + output_offset,
-        acc,
-        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
-    )
-
-
-@triton.jit
-def kernel_unified_attention_3d(
-    segm_output_ptr,
-    # [num_tokens, num_query_heads, num_segments, head_size_padded]
-    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
-    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
-    query_ptr,  # [num_tokens, num_query_heads, head_size]
-    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
-    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
-    sink_ptr,  # [num_query_heads]
-    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
-    seq_lens_ptr,  # [num_seqs]
-    alibi_slopes_ptr,  # [num_query_heads]
-    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
-    scale,  # float32
-    k_scale,  # float32
-    v_scale,  # float32
-    softcap,  # float32
-    num_query_heads: tl.constexpr,  # int
-    num_queries_per_kv: tl.constexpr,  # int
-    block_table_stride: tl.int64,  # int
-    query_stride_0: tl.int64,  # int
-    query_stride_1: tl.int64,  # int, should be equal to head_size
-    qq_bias_stride_0: tl.int64,  # int
-    BLOCK_SIZE: tl.constexpr,  # int
-    TILE_SIZE: tl.constexpr,  # int, must be power of 2
-    HEAD_SIZE: tl.constexpr,  # int
-    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
-    USE_ALIBI_SLOPES: tl.constexpr,  # bool
-    USE_ALIBI_SQRT: tl.constexpr,  # bool
-    USE_QQ_BIAS: tl.constexpr,  # bool
-    USE_SOFTCAP: tl.constexpr,  # bool
-    USE_SINKS: tl.constexpr,  # bool
-    SLIDING_WINDOW: tl.constexpr,  # int
-    stride_k_cache_0: tl.int64,  # int
-    stride_k_cache_1: tl.int64,  # int
-    stride_k_cache_2: tl.int64,  # int
-    stride_k_cache_3: tl.constexpr,  # int
-    stride_v_cache_0: tl.int64,  # int
-    stride_v_cache_1: tl.int64,  # int
-    stride_v_cache_2: tl.int64,  # int
-    stride_v_cache_3: tl.constexpr,  # int
-    query_start_len_ptr,  # [num_seqs+1]
-    BLOCK_Q: tl.constexpr,  # int
-    num_seqs: tl.int32,
-    BLOCK_M: tl.constexpr,  # int
-    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
-    USE_MM_PREFIX: tl.constexpr,  # bool
-    MAX_MM_RANGES: tl.constexpr,  # int
-    mm_prefix_range_ptr,  # [num_seqs] - prefix length for each sequence
-    # KV cache quantization: 0=none, 1=fp8, 2=per-token-head
-    KV_QUANT_MODE: tl.constexpr = 0,
-    # Per-token-head scale caches (KV_QUANT_MODE >= 2)
-    # Shape: [num_blocks, block_size, num_kv_heads]
-    k_scale_cache_ptr=None,
-    v_scale_cache_ptr=None,
-    stride_ks_blk=0,
-    stride_ks_slot=0,
-    stride_ks_head=0,
-    stride_vs_blk=0,
-    stride_vs_slot=0,
-    stride_vs_head=0,
-    CHUNK_LOOKBACK: tl.constexpr = -1,
-    CHUNK_SIZE: tl.constexpr = -1,
-):
-    q_block_global_idx = tl.program_id(0)
-    kv_head_idx = tl.program_id(1)
-    segm_idx = tl.program_id(2)
-
-    seq_idx = find_seq_idx(
-        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
-    )
-
-    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
-
-    q_block_local_idx = q_block_global_idx - q_block_start_idx
-
-    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
-    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
-
-    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
-
-    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
-        return
-
-    # sequence len for this particular sequence
-    seq_len = tl.load(seq_lens_ptr + seq_idx)
-
-    # number of segments for this particular sequence
-    num_segments = NUM_SEGMENTS_PER_SEQ
-    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
-
-    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
-        return
-
-    offs_m = tl.arange(0, BLOCK_M)
-    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
-    offs_t = tl.arange(0, TILE_SIZE)
-    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
-
-    query_offset_0 = cur_batch_in_all_start_index + query_pos
-    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
-    query_offset = (
-        query_offset_0[:, None] * query_stride_0
-        + query_offset_1[:, None] * query_stride_1
-        + offs_d[None, :]
-    )
-
-    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
-    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
-    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
-
-    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
-    Q = tl.load(
-        query_ptr + query_offset,
-        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
-        other=0.0,
-    )
-
-    block_table_offset = seq_idx * block_table_stride
-
-    if USE_SINKS:
-        if segm_idx == 0:
-            M = tl.load(
-                sink_ptr + query_offset_1,
-                mask=query_mask_1,
-                other=float("-inf"),
-            ).to(dtype=tl.float32)
-        else:
-            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
-    else:
-        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
-
-    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
-    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
-
-    # context length for this particular sequences
-    context_len = seq_len - cur_batch_query_len
-
-    # alibi slope for this head
-    if USE_ALIBI_SLOPES:
-        alibi_slope = tl.load(
-            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
-        )
-
-    # query-query attention bias
-    if USE_QQ_BIAS:
-        qq_bias_row_ptrs = (
-            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
-        )  # shape: [BLOCK_M]
-
-    # compute the length of the longest sequence prefix spanned by any
-    # query token in the current q_block (q_block_local_idx)
-    max_seq_prefix_len = (
-        context_len
-        + q_block_local_idx * BLOCK_Q
-        + (BLOCK_M - 1) // num_queries_per_kv
-        + 1
-    )
-
-    # adjust for potential padding in the last q_block by considering the
-    # actual sequence length
-    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
-
-    # calculate the number of tiles that need to be processed to
-    # cover the longest sequence prefix (due to causal masking, tiles beyond
-    # this prefix can be skipped)
-    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)
-
-    # ---- Sliding-window tile pruning --------------------
-    # Default: keep previous global behavior
-    tile_start = 0
-    tile_end = num_tiles
-    # TODO(Isotr0py): sliding window pruning with image bidirectional mask
-    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
-        # Query rows covered by this Q-block
-        qpos_lo = q_block_local_idx * BLOCK_Q
-        qpos_hi = tl.minimum(
-            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
-            cur_batch_query_len - 1,
-        )
-        # For sliding window, each query position q can only attend to
-        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
-        # where q_abs = context_len + q
-        # The union of allowed key positions for this Q-block is:
-        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
-        q_abs = context_len + qpos_lo
-        if CHUNK_LOOKBACK > -1:
-            first_allowed_key = ((q_abs // CHUNK_SIZE) - CHUNK_LOOKBACK) * CHUNK_SIZE
-        else:
-            first_allowed_key = q_abs - SLIDING_WINDOW + 1
-        last_allowed_key = context_len + qpos_hi
-        # Convert to tile indices and clamp
-        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
-        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
-
-    # iterate through tiles (now limited to the sliding window range)
-    for j in range(
-        max(segm_idx * tiles_per_segment, tile_start),
-        min((segm_idx + 1) * tiles_per_segment, tile_end),
-    ):
-        seq_offset = j * TILE_SIZE + offs_t
-        tile_mask = seq_offset < max_seq_prefix_len
-
-        physical_block_idx = tl.load(
-            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
-        ).to(tl.int64)
-
-        v_offset = (
-            physical_block_idx[:, None] * stride_v_cache_0
-            + kv_head_idx * stride_v_cache_2
-            + offs_d[None, :] * stride_v_cache_3
-            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
-        )
-
-        k_offset = (
-            physical_block_idx[None, :] * stride_k_cache_0
-            + kv_head_idx * stride_k_cache_2
-            + offs_d[:, None] * stride_k_cache_3
-            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
+    # ---- Epilogue ---------------------------------------------------------
+    if IS_3D:
+        # Store per-segment partials; finalized by ``reduce_segments``.
+        segm_output_offset = (
+            query_offset_0[:, None].to(tl.int64)
+            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
+            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
+            + segm_idx * HEAD_SIZE_PADDED
+            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
         )
-
-        # K : (HEAD_SIZE, TILE_SIZE)
-        K_load = tl.load(
-            key_cache_ptr + k_offset,
-            mask=dim_mask[:, None] & tile_mask[None, :],
-            other=0.0,
+        tl.store(
+            segm_output_ptr + segm_output_offset,
+            acc,
+            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
         )
-        K, k_token_head_scales = _prepare_kv_tile(
-            K_load,
-            Q,
-            k_scale,
-            k_scale_cache_ptr,
-            physical_block_idx,
-            seq_offset,
-            kv_head_idx,
-            stride_ks_blk,
-            stride_ks_slot,
-            stride_ks_head,
-            tile_mask,
-            BLOCK_SIZE,
-            KV_QUANT_MODE,
+        store_segm_reduce_scalars(
+            segm_max_ptr,
+            segm_expsum_ptr,
+            query_offset_0,
+            query_offset_1,
+            segm_idx,
+            M,
+            L,
+            query_mask_0,
+            query_mask_1,
+            num_query_heads,
+            NUM_SEGMENTS_PER_SEQ,
         )
-
-        # V : (TILE_SIZE, HEAD_SIZE)
-        V_load = tl.load(
-            value_cache_ptr + v_offset,
-            mask=dim_mask[None, :] & tile_mask[:, None],
-            other=0.0,
+    else:
+        acc = acc / L[:, None]
+        if USE_FP8:
+            acc = acc * tl.load(out_scale)
+            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
+        output_offset = (
+            query_offset_0[:, None] * output_stride_0
+            + query_offset_1[:, None] * output_stride_1
+            + offs_d[None, :]
         )
-        V, v_token_head_scales = _prepare_kv_tile(
-            V_load,
-            Q,
-            v_scale,
-            v_scale_cache_ptr,
-            physical_block_idx,
-            seq_offset,
-            kv_head_idx,
-            stride_vs_blk,
-            stride_vs_slot,
-            stride_vs_head,
-            tile_mask,
-            BLOCK_SIZE,
-            KV_QUANT_MODE,
+        tl.store(
+            output_ptr + output_offset,
+            acc,
+            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
         )
 
-        # Compute attention mask: causal by default (key <= query)
-        query_abs_pos = context_len + query_pos[:, None]
-        seq_mask = seq_offset[None, :] <= query_abs_pos
-
-        # Apply sliding window / chunked attention to base mask
-        # BEFORE mm_prefix OR.
-        # Order must match FlexAttention:
-        #   (causal AND sliding_window) OR mm_prefix
-        if CHUNK_LOOKBACK > -1:
-            seq_mask = seq_mask & (
-                (
-                    (context_len + query_pos[:, None]) // CHUNK_SIZE
-                    - (seq_offset[None, :] // CHUNK_SIZE)
-                )
-                <= CHUNK_LOOKBACK
-            )
-        elif SLIDING_WINDOW > 0:
-            seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
-
-        # PrefixLM: extend mask with bidirectional ranges for multimodal tokens.
-        # Applied AFTER sliding window so mm_prefix ranges override SW restriction.
-        if USE_MM_PREFIX:
-            for i in range(MAX_MM_RANGES):
-                range_start = tl.load(
-                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
-                )
-                range_end = tl.load(
-                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
-                )
-
-                is_valid = range_start < range_end
-                q_in_range = (
-                    (query_abs_pos >= range_start)
-                    & (query_abs_pos <= range_end)
-                    & is_valid
-                )
-                k_in_range = (
-                    (seq_offset[None, :] >= range_start)
-                    & (seq_offset[None, :] <= range_end)
-                    & is_valid
-                )
-                seq_mask |= q_in_range & k_in_range
-
-        # S : (BLOCK_M, TILE_SIZE)
-        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
-
-        # Per-token-head quant: fuse softmax_scale with per-head k_scale
-        # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
-        if KV_QUANT_MODE >= 2:
-            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
-        else:
-            S += scale * tl.dot(Q, K)
-
-        if USE_SOFTCAP:
-            S = apply_softcap(S, softcap)
-
-        S = tl.where(
-            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
-        )
-
-        if USE_ALIBI_SLOPES:
-            if USE_ALIBI_SQRT:
-                relative_pos = seq_offset - (context_len + query_pos[:, None])
-                alibi_offset = tl.where(
-                    relative_pos <= 0,
-                    -tl.sqrt((-relative_pos).to(tl.float32)),
-                    0.0,
-                )
-            else:
-                alibi_offset = seq_offset - context_len
-            S += alibi_slope[:, None] * alibi_offset
-
-        if USE_QQ_BIAS:
-            # compute key positions relative to query section
-            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
-            # load bias only for keys that correspond to queries
-            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
-            qq_bias = tl.load(
-                qq_bias_row_ptrs + key_rel_pos[None, :],
-                mask=is_query_key[None, :],  # avoid OOB for context keys
-                other=0.0,
-            )
-            S += qq_bias
-
-        # compute running maximum
-        # m_j : (BLOCK_M,)
-        m_j = tl.maximum(M, tl.max(S, axis=1))
-
-        # For sliding window there's a chance the max is -inf due to masking of
-        # the entire row. In this case we need to set m_j 0 to avoid NaN
-        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
-
-        # P : (BLOCK_M, TILE_SIZE,)
-        P = tl.exp(S - m_j[:, None])
-
-        # l_j : (BLOCK_M,)
-        l_j = tl.sum(P, axis=1)
-
-        # alpha : (BLOCK_M, )
-        alpha = tl.exp(M - m_j)
-
-        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
-        acc = acc * alpha[:, None]
-
-        # update constants
-        L = L * alpha + l_j
-        M = m_j
-
-        if SLIDING_WINDOW:
-            qpos_lo = q_block_local_idx * BLOCK_Q
-            V = tl.where(
-                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW, V, 0.0
-            )
-
-        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
-        # Per-token-head quant: apply v_scale to P instead of V.
-        if KV_QUANT_MODE >= 2:
-            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
-            acc += tl.dot(P_v, V)
-        else:
-            acc += tl.dot(P.to(V.dtype), V)
-
-    segm_output_offset = (
-        query_offset_0[:, None].to(tl.int64)
-        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
-        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
-        + segm_idx * HEAD_SIZE_PADDED
-        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
-    )
-    tl.store(
-        segm_output_ptr + segm_output_offset,
-        acc,
-        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
-    )
-    segm_offset = (
-        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
-        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
-        + segm_idx
-    )
-    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
-    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)
-
 
 @triton.jit
 def reduce_segments(
@@ -1028,12 +490,7 @@ def _get_tile_size(
     element_size: int,
     is_prefill: bool,
 ) -> int:
-    """Select tile size with Gemma3-specific optimization.
-
-    For Gemma3, use 32 for both prefill and decode to better utilize
-    the larger head dimension (128/256). For other models, use
-    the default vLLM behavior.
-    """
+    """Select tile size with Gemma3-specific optimization."""
     if _is_gemma3_attention(head_size, sliding_window):
         # Gemma3: use 32 for decode (default is 16)
         return 32
@@ -1041,6 +498,7 @@ def _get_tile_size(
     # Default behavior
     if is_prefill:
         return 32
+    # Note: tile size must be at least 32 for fp8 (element_size == 1).
     return 16 if element_size >= 2 else 32
 
 
@@ -1087,6 +545,15 @@ def unified_attention(
     if sinks is not None:
         assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"
 
+    use_per_token_head_scales = kv_quant_mode in (
+        KVQuantMode.INT8_PER_TOKEN_HEAD,
+        KVQuantMode.FP8_PER_TOKEN_HEAD,
+    )
+    if use_per_token_head_scales:
+        assert k_scale_cache is not None and v_scale_cache is not None, (
+            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
+        )
+
     use_mm_prefix = False
     max_mm_ranges = 0
     if mm_prefix_range is not None:
@@ -1124,8 +591,6 @@ def unified_attention(
     #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
     total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
 
-    # Tile sizes for prefill and decode. Gemma3 models use optimized values.
-    # Note: tile size must be at least 32 for fp8 (element_size == 1).
     sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
 
     # Compute chunked block size from sliding window if needed.
@@ -1137,16 +602,10 @@ def unified_attention(
         chunk_lookback = -1
 
     TILE_SIZE_PREFILL = _get_tile_size(
-        head_size,
-        sliding_window_val,
-        q.element_size(),
-        is_prefill=True,
+        head_size, sliding_window_val, q.element_size(), is_prefill=True
     )
     TILE_SIZE_DECODE = _get_tile_size(
-        head_size,
-        sliding_window_val,
-        q.element_size(),
-        is_prefill=False,
+        head_size, sliding_window_val, q.element_size(), is_prefill=False
     )
 
     # Launch the 2D kernel if
@@ -1154,7 +613,7 @@ def unified_attention(
     # 2. The batch includes at least one prefill request, or
     # 3. The number of sequences exceeds the configured threshold, or
     # 4. Batch invariance is enabled
-    if (
+    use_3d = not (
         seq_threshold_3D is None
         or num_par_softmax_segments is None
         or softmax_segm_output is None
@@ -1163,136 +622,110 @@ def unified_attention(
         or max_seqlen_q > 1
         or num_seqs > seq_threshold_3D
         or is_batch_invariant
-    ):
-        kernel_unified_attention_2d[
-            (
-                total_num_q_blocks,
-                num_kv_heads,
-            )
-        ](
-            output_ptr=out,
-            query_ptr=q,
-            key_cache_ptr=k,
-            value_cache_ptr=v,
-            sink_ptr=sinks,
-            block_tables_ptr=block_table,
-            seq_lens_ptr=seqused_k,
-            alibi_slopes_ptr=alibi_slopes,
-            qq_bias_ptr=qq_bias,
-            scale=softmax_scale,
-            k_scale=k_descale,
-            v_scale=v_descale,
-            out_scale=1 / output_scale if output_scale is not None else 1.0,
-            softcap=softcap,
-            num_query_heads=num_query_heads,
-            num_queries_per_kv=num_queries_per_kv,
-            block_table_stride=block_table.stride(0),
-            query_stride_0=q.stride(0),
-            query_stride_1=q.stride(1),
-            output_stride_0=out.stride(0),
-            output_stride_1=out.stride(1),
-            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
-            BLOCK_SIZE=block_size,
-            TILE_SIZE=TILE_SIZE_PREFILL,
-            HEAD_SIZE=head_size,
-            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
-            USE_ALIBI_SLOPES=use_alibi_slopes,
-            USE_ALIBI_SQRT=use_alibi_sqrt,
-            USE_QQ_BIAS=use_qq_bias,
-            USE_SOFTCAP=(softcap > 0),
-            USE_SINKS=(sinks is not None),
-            USE_MM_PREFIX=use_mm_prefix,
-            MAX_MM_RANGES=max_mm_ranges,
-            mm_prefix_range_ptr=mm_prefix_range,
-            SLIDING_WINDOW=(1 + window_size[0]),
-            stride_k_cache_0=k.stride(0),
-            stride_k_cache_1=k.stride(1),
-            stride_k_cache_2=k.stride(2),
-            stride_k_cache_3=k.stride(3),
-            stride_v_cache_0=v.stride(0),
-            stride_v_cache_1=v.stride(1),
-            stride_v_cache_2=v.stride(2),
-            stride_v_cache_3=v.stride(3),
-            query_start_len_ptr=cu_seqlens_q,
-            BLOCK_Q=BLOCK_Q,
-            num_seqs=num_seqs,
-            BLOCK_M=BLOCK_M,
-            USE_FP8=output_scale is not None,
-            KV_QUANT_MODE=kv_quant_mode,
-            k_scale_cache_ptr=k_scale_cache,
-            v_scale_cache_ptr=v_scale_cache,
-            stride_ks_blk=k_scale_cache.stride(0) if k_scale_cache is not None else 0,
-            stride_ks_slot=k_scale_cache.stride(1) if k_scale_cache is not None else 0,
-            stride_ks_head=k_scale_cache.stride(2) if k_scale_cache is not None else 0,
-            stride_vs_blk=v_scale_cache.stride(0) if v_scale_cache is not None else 0,
-            stride_vs_slot=v_scale_cache.stride(1) if v_scale_cache is not None else 0,
-            stride_vs_head=v_scale_cache.stride(2) if v_scale_cache is not None else 0,
-            CHUNK_LOOKBACK=chunk_lookback,
-            CHUNK_SIZE=chunk_size,
-        )
+    )
+
+    # The kernel signature is the same for 2D and 3D — only the launch
+    # grid + a handful of constexpr toggles differ.  Per-token-head scale
+    # caches and their strides are required arguments; non-per-token-head
+    # modes pass dummy zeros (the code path is dead-code eliminated by
+    # the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch in the kernel).
+    if use_per_token_head_scales:
+        ks_strides = k_scale_cache.stride()
+        vs_strides = v_scale_cache.stride()
+        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
+        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
+        k_scale_ptr = k_scale_cache
+        v_scale_ptr = v_scale_cache
     else:
-        kernel_unified_attention_3d[
-            (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
-        ](
-            segm_output_ptr=softmax_segm_output,
-            segm_max_ptr=softmax_segm_max,
-            segm_expsum_ptr=softmax_segm_expsum,
-            query_ptr=q,
-            key_cache_ptr=k,
-            value_cache_ptr=v,
-            sink_ptr=sinks,
-            block_tables_ptr=block_table,
-            seq_lens_ptr=seqused_k,
-            alibi_slopes_ptr=alibi_slopes,
-            qq_bias_ptr=qq_bias,
-            scale=softmax_scale,
-            k_scale=k_descale,
-            v_scale=v_descale,
-            softcap=softcap,
-            num_query_heads=num_query_heads,
-            num_queries_per_kv=num_queries_per_kv,
-            block_table_stride=block_table.stride(0),
-            query_stride_0=q.stride(0),
-            query_stride_1=q.stride(1),
-            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
-            BLOCK_SIZE=block_size,
-            TILE_SIZE=TILE_SIZE_DECODE,
-            HEAD_SIZE=head_size,
-            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
-            USE_ALIBI_SLOPES=use_alibi_slopes,
-            USE_ALIBI_SQRT=use_alibi_sqrt,
-            USE_QQ_BIAS=use_qq_bias,
-            USE_SOFTCAP=(softcap > 0),
-            USE_SINKS=(sinks is not None),
-            USE_MM_PREFIX=use_mm_prefix,
-            MAX_MM_RANGES=max_mm_ranges,
-            mm_prefix_range_ptr=mm_prefix_range,
-            SLIDING_WINDOW=(1 + window_size[0]),
-            stride_k_cache_0=k.stride(0),
-            stride_k_cache_1=k.stride(1),
-            stride_k_cache_2=k.stride(2),
-            stride_k_cache_3=k.stride(3),
-            stride_v_cache_0=v.stride(0),
-            stride_v_cache_1=v.stride(1),
-            stride_v_cache_2=v.stride(2),
-            stride_v_cache_3=v.stride(3),
-            query_start_len_ptr=cu_seqlens_q,
-            BLOCK_Q=BLOCK_Q,
-            num_seqs=num_seqs,
-            BLOCK_M=BLOCK_M,
-            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
-            KV_QUANT_MODE=kv_quant_mode,
-            k_scale_cache_ptr=k_scale_cache,
-            v_scale_cache_ptr=v_scale_cache,
-            stride_ks_blk=k_scale_cache.stride(0) if k_scale_cache is not None else 0,
-            stride_ks_slot=k_scale_cache.stride(1) if k_scale_cache is not None else 0,
-            stride_ks_head=k_scale_cache.stride(2) if k_scale_cache is not None else 0,
-            stride_vs_blk=v_scale_cache.stride(0) if v_scale_cache is not None else 0,
-            stride_vs_slot=v_scale_cache.stride(1) if v_scale_cache is not None else 0,
-            stride_vs_head=v_scale_cache.stride(2) if v_scale_cache is not None else 0,
-            CHUNK_LOOKBACK=chunk_lookback,
-            CHUNK_SIZE=chunk_size,
-        )
+        ks_blk = ks_slot = ks_head = 0
+        vs_blk = vs_slot = vs_head = 0
+        # Pass the K cache as a stand-in pointer; never dereferenced.
+        k_scale_ptr = k
+        v_scale_ptr = v
+
+    # 3D needs real segm tensors; 2D never touches them but Triton wants
+    # a non-null pointer.  Reuse ``out`` as the placeholder.
+    segm_output_ptr = softmax_segm_output if use_3d else out
+    segm_max_ptr = softmax_segm_max if use_3d else out
+    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
+    num_segments = num_par_softmax_segments if use_3d else 1
+
+    grid: tuple[Any, ...]
+    if not use_3d:
+        grid = (total_num_q_blocks, num_kv_heads)
+        tile_size = TILE_SIZE_PREFILL
+    else:
+        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
+        tile_size = TILE_SIZE_DECODE
+
+    kernel_unified_attention[grid](
+        output_ptr=out,
+        segm_output_ptr=segm_output_ptr,
+        segm_max_ptr=segm_max_ptr,
+        segm_expsum_ptr=segm_expsum_ptr,
+        query_ptr=q,
+        key_cache_ptr=k,
+        value_cache_ptr=v,
+        sink_ptr=sinks,
+        block_tables_ptr=block_table,
+        seq_lens_ptr=seqused_k,
+        alibi_slopes_ptr=alibi_slopes,
+        qq_bias_ptr=qq_bias,
+        k_scale_cache_ptr=k_scale_ptr,
+        v_scale_cache_ptr=v_scale_ptr,
+        scale=softmax_scale,
+        k_scale=k_descale,
+        v_scale=v_descale,
+        out_scale=1 / output_scale if output_scale is not None else 1.0,
+        softcap=softcap,
+        num_query_heads=num_query_heads,
+        num_queries_per_kv=num_queries_per_kv,
+        block_table_stride=block_table.stride(0),
+        query_stride_0=q.stride(0),
+        query_stride_1=q.stride(1),
+        output_stride_0=out.stride(0),
+        output_stride_1=out.stride(1),
+        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
+        BLOCK_SIZE=block_size,
+        TILE_SIZE=tile_size,
+        HEAD_SIZE=head_size,
+        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
+        USE_ALIBI_SLOPES=use_alibi_slopes,
+        USE_ALIBI_SQRT=use_alibi_sqrt,
+        USE_QQ_BIAS=use_qq_bias,
+        USE_SOFTCAP=(softcap > 0),
+        USE_SINKS=(sinks is not None),
+        USE_MM_PREFIX=use_mm_prefix,
+        MAX_MM_RANGES=max_mm_ranges,
+        mm_prefix_range_ptr=mm_prefix_range,
+        SLIDING_WINDOW=(1 + window_size[0]),
+        stride_k_cache_0=k.stride(0),
+        stride_k_cache_1=k.stride(1),
+        stride_k_cache_2=k.stride(2),
+        stride_k_cache_3=k.stride(3),
+        stride_v_cache_0=v.stride(0),
+        stride_v_cache_1=v.stride(1),
+        stride_v_cache_2=v.stride(2),
+        stride_v_cache_3=v.stride(3),
+        stride_ks_blk=ks_blk,
+        stride_ks_slot=ks_slot,
+        stride_ks_head=ks_head,
+        stride_vs_blk=vs_blk,
+        stride_vs_slot=vs_slot,
+        stride_vs_head=vs_head,
+        query_start_len_ptr=cu_seqlens_q,
+        BLOCK_Q=BLOCK_Q,
+        num_seqs=num_seqs,
+        BLOCK_M=BLOCK_M,
+        NUM_SEGMENTS_PER_SEQ=num_segments,
+        USE_FP8=output_scale is not None,
+        IS_3D=use_3d,
+        KV_QUANT_MODE=kv_quant_mode,
+        CHUNK_LOOKBACK=chunk_lookback,
+        CHUNK_SIZE=chunk_size,
+    )
+
+    if use_3d:
         reduce_segments[(q.shape[0], num_query_heads)](
             output_ptr=out,
             segm_output_ptr=softmax_segm_output,
```
