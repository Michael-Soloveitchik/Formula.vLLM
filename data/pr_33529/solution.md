# PR #33529: Triton MLA perf fixes

- field: attention  | area: v1_attention
- author: koush
- state: MERGED  created: 2026-02-02T01:40:05Z  merged: 2026-04-02T13:40:01Z
- url: https://github.com/vllm-project/vllm/pull/33529
- labels: performance, ready, v1, deepseek

# Issue

## Problem

Triton MLA decode uses a fixed `num_kv_splits = 4` regardless of context length or GPU SM count, so at low batch size only ~4 CTAs are launched for the KV reduction, leaving the vast majority of SMs idle. Separately, the grouped stage-1 kernel loads V from the KV buffer as if it were distinct from K, but MLA stores a single combined latent `c_kv` that serves as both K and V, so V is redundantly re-loaded from HBM.

## Example

Kimi K2.5 with `-dcp 8` at 80K context, batch size 1: peak throughput was 34 tok/s before vs 52 tok/s after (+52.94%). At 128 tokens the effect is negligible (54 -> 55 tok/s), confirming the problem scales with context length.

## Use case

Low-concurrency (batch size 1) long-context decode serving of large MLA models like DeepSeek V2/V3 and Kimi K2/K2.5.

## Resources

GPU compute (SM occupancy/parallelism underutilized) and HBM bandwidth (redundant V loads of the same c_kv latent).

## Type of users

Users serving MLA-architecture models (DeepSeek, Kimi) on sm120 and similar GPUs, especially single-request long-context inference with decode-context-parallel.

## Kind of inputs

Small batch (B=1), long sequence lengths (tens of thousands of tokens); impact grows with max_seq_len and vanishes at short context.

## Mechanism with the issue

In `forward_mqa` (triton_mla.py) the split count was hardcoded to 4, so `_fwd_grouped_kernel_stage1` launched a grid too small to saturate SMs when context is long and batch is tiny. Inside that kernel, V was loaded via `offs_buf_v` into `V_Buffer` even though MLA's c_kv already holds the value data as K, doubling the latent reads from the KV cache.

# Solution

## Fix in words

`num_kv_splits` is now computed dynamically: `ideal_splits = max(1, max_seq_len // 512)` rounded up to the next power of 2, then capped at `sm_count * 2` (occupancy multiplier), replacing the fixed 4 (batch-invariant path still forces 1). The SM count is cached once in `__init__` via `torch.cuda.get_device_properties(0).multi_processor_count`. An `is_mla`/`IS_MLA` flag is threaded through `decode_attention_fwd[_grouped]` into `_fwd_grouped_kernel_stage1`; when set, the kernel skips the separate V load and instead reuses K via `v = tl.trans(k)`, eliminating the redundant c_kv read. Load instructions also gain `cache_modifier` hints (`.ca` for Q/page table, `.cg` for K/KPE) and base offsets are hoisted out of the inner loop with `tl.range` to better overlap load/compute.

## Author's solution

The author identified two primary issues: suboptimal KV splitting during low batch count that underutilizes SMs, and unnecessary loading of the Q/V vector since c_kv already contains both. The fix makes KV splitting adaptive to context length and SM count, and avoids the redundant c_kv load in the MLA path. Validated on Kimi 2.5, DeepSeek V2, and Qwen 235B, showing ~53% throughput gain at batch 1 / 80K context.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/backends/mla/triton_mla.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`

### Diff
```diff
diff --git a/vllm/v1/attention/backends/mla/triton_mla.py b/vllm/v1/attention/backends/mla/triton_mla.py
index 6fa1bbf20874..e8b09a436567 100644
--- a/vllm/v1/attention/backends/mla/triton_mla.py
+++ b/vllm/v1/attention/backends/mla/triton_mla.py
@@ -14,6 +14,7 @@
     MLACommonMetadata,
 )
 from vllm.platforms.interface import DeviceCapability
+from vllm.triton_utils import triton
 from vllm.utils.torch_utils import is_quantized_kv_cache
 from vllm.v1.attention.backend import (
     AttentionLayer,
@@ -115,6 +116,8 @@ def __init__(
         if is_quantized_kv_cache(self.kv_cache_dtype):
             self.supports_quant_query_input = False
 
+        self._sm_count = torch.cuda.get_device_properties(0).multi_processor_count
+
     def _flash_attn_varlen_diff_headdims(
         self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
     ):
@@ -149,7 +152,24 @@ def forward_mqa(
         lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)
 
         # For batch invariance, use only 1 split to ensure deterministic reduction
-        num_kv_splits = 1 if envs.VLLM_BATCH_INVARIANT else 4
+        if envs.VLLM_BATCH_INVARIANT:
+            num_kv_splits = 1
+        else:
+            # Minimum work per split
+            # hardware dependent
+            min_work_per_split = 512
+
+            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)
+
+            # use power of 2 to avoid excessive kernel instantiations
+            ideal_splits = triton.next_power_of_2(ideal_splits)
+
+            # Calculate SM-based maximum splits with occupancy multiplier
+            # 2-4x allows multiple blocks per SM for latency hiding
+            # hardware dependent
+            occupancy_multiplier = 2
+            max_splits = self._sm_count * occupancy_multiplier
+            num_kv_splits = min(ideal_splits, max_splits)
 
         # TODO(lucas) Allocate ahead of time
         attn_logits = torch.empty(
@@ -186,6 +206,7 @@ def forward_mqa(
             PAGE_SIZE,
             k_scale=layer._k_scale,
             v_scale=layer._k_scale,
+            is_mla=True,
         )
 
         return o, lse
diff --git a/vllm/v1/attention/ops/triton_decode_attention.py b/vllm/v1/attention/ops/triton_decode_attention.py
index 63263bc92e24..8118db0da8cf 100644
--- a/vllm/v1/attention/ops/triton_decode_attention.py
+++ b/vllm/v1/attention/ops/triton_decode_attention.py
@@ -291,6 +291,7 @@ def _fwd_grouped_kernel_stage1(
     logit_cap: tl.constexpr,
     Lk: tl.constexpr,
     Lv: tl.constexpr,
+    IS_MLA: tl.constexpr = False,
 ):
     cur_batch = tl.program_id(0)
     cur_head_id = tl.program_id(1)
@@ -310,7 +311,12 @@ def _fwd_grouped_kernel_stage1(
     cur_batch_req_idx = cur_batch
 
     offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
-    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
+    q = tl.load(
+        Q + offs_q,
+        mask=(mask_h[:, None]) & (mask_d[None, :]),
+        other=0.0,
+        cache_modifier=".ca",
+    )
 
     if BLOCK_DPE > 0:
         offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
@@ -319,7 +325,10 @@ def _fwd_grouped_kernel_stage1(
             cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
         )
         qpe = tl.load(
-            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
+            Q + off_qpe,
+            mask=(mask_h[:, None]) & (mask_dpe[None, :]),
+            other=0.0,
+            cache_modifier=".ca",
         )
 
     kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
@@ -331,9 +340,14 @@ def _fwd_grouped_kernel_stage1(
     acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
 
     if split_kv_end > split_kv_start:
+        base_offs_k = cur_kv_head * stride_buf_kh + offs_d[:, None]
+        base_offs_v = cur_kv_head * stride_buf_vh + offs_dv[None, :]
+        if BLOCK_DPE > 0:
+            base_offs_kpe = cur_kv_head * stride_buf_kh + offs_dpe[:, None]
+
         ks = tl.load(k_scale)
         vs = tl.load(v_scale)
-        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
+        for start_n in tl.range(split_kv_start, split_kv_end, BLOCK_N):
             offs_n = start_n + tl.arange(0, BLOCK_N)
             kv_page_number = tl.load(
                 Req_to_tokens
@@ -341,31 +355,29 @@ def _fwd_grouped_kernel_stage1(
                 + offs_n // PAGE_SIZE,
                 mask=offs_n < split_kv_end,
                 other=0,
+                cache_modifier=".ca",
             )
             kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
-            offs_buf_k = (
-                kv_loc[None, :] * stride_buf_kbs
-                + cur_kv_head * stride_buf_kh
-                + offs_d[:, None]
-            )
+
+            # explicitly facilitate overlapping load/compute
+            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + base_offs_k
             k = tl.load(
                 K_Buffer + offs_buf_k,
                 mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                 other=0.0,
+                cache_modifier=".cg",
             )
+
             if k.dtype.is_fp8():
                 k = (k.to(tl.float32) * ks).to(q.dtype)
             qk = tl.dot(q, k.to(q.dtype))
             if BLOCK_DPE > 0:
-                offs_buf_kpe = (
-                    kv_loc[None, :] * stride_buf_kbs
-                    + cur_kv_head * stride_buf_kh
-                    + offs_dpe[:, None]
-                )
+                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + base_offs_kpe
                 kpe = tl.load(
                     K_Buffer + offs_buf_kpe,
                     mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                     other=0.0,
+                    cache_modifier=".cg",
                 )
                 if kpe.dtype.is_fp8():
                     kpe = (kpe.to(tl.float32) * ks).to(qpe.dtype)
@@ -379,18 +391,20 @@ def _fwd_grouped_kernel_stage1(
                 mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
             )
 
-            offs_buf_v = (
-                kv_loc[:, None] * stride_buf_vbs
-                + cur_kv_head * stride_buf_vh
-                + offs_dv[None, :]
-            )
-            v = tl.load(
-                V_Buffer + offs_buf_v,
-                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
-                other=0.0,
-            )
-            if v.dtype.is_fp8():
-                v = (v.to(tl.float32) * vs).to(q.dtype)
+            if not IS_MLA:
+                offs_buf_v = kv_loc[:, None] * stride_buf_vbs + base_offs_v
+                v = tl.load(
+                    V_Buffer + offs_buf_v,
+                    mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
+                    other=0.0,
+                )
+                if v.dtype.is_fp8():
+                    v = (v.to(tl.float32) * vs).to(q.dtype)
+            else:
+                # MLA uses a single c_kv.
+                # loading the same c_kv to interpret it as v is not necessary.
+                # transpose the existing c_kv (aka k) for the dot product.
+                v = tl.trans(k)
 
             n_e_max = tl.maximum(tl.max(qk, 1), e_max)
             re_scale = tl.exp(e_max - n_e_max)
@@ -441,7 +455,10 @@ def _decode_grouped_att_m_fwd(
     logit_cap,
     k_scale,
     v_scale,
+    is_mla=False,
 ):
+    # with is_mla there is only a single c_kv in smem.
+    # could increase BLOCK or num_stages.
     BLOCK = 32
     Lk = k_buffer.shape[-1]
     Lv = v_buffer.shape[-1]
@@ -514,6 +531,7 @@ def _decode_grouped_att_m_fwd(
         num_stages=num_stages,
         Lk=Lk,
         Lv=Lv,
+        IS_MLA=is_mla,
         **extra_kargs,
     )
 
@@ -673,6 +691,7 @@ def decode_attention_fwd_grouped(
     logit_cap=0.0,
     k_scale=None,
     v_scale=None,
+    is_mla=False,
 ):
     _decode_grouped_att_m_fwd(
         q,
@@ -687,6 +706,7 @@ def decode_attention_fwd_grouped(
         logit_cap,
         k_scale,
         v_scale,
+        is_mla=is_mla,
     )
     _decode_softmax_reducev_fwd(
         attn_logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits
@@ -708,6 +728,7 @@ def decode_attention_fwd(
     logit_cap=0.0,
     k_scale=None,
     v_scale=None,
+    is_mla=False,
 ):
     assert num_kv_splits == attn_logits.shape[2]
 
@@ -753,4 +774,5 @@ def decode_attention_fwd(
             logit_cap,
             k_scale,
             v_scale,
+            is_mla=is_mla,
         )

```
