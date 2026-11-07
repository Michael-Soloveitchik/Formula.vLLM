# PR #40941: [Attention][TurboQuant] Share dequant buffers, eliminate float16_copy

- field: attention  | area: v1_attention
- author: bhoomit
- state: MERGED  created: 2026-04-26T22:49:44Z  merged: 2026-04-27T05:48:36Z
- url: https://github.com/vllm-project/vllm/pull/40941
- labels: ready, v1

# Issue

## Problem

The TurboQuant continuation-prefill path allocated per-layer FP16 dequant buffers (`_tq_k_dequant_buf`, `_tq_v_dequant_buf`) that grew lazily to the max continuation length seen per layer. With ~60 active TQ layers × 2 buffers, memory scaled as O(layers × context_len), wasting tens of GB at long context. The lazy per-layer growth also triggers cudaMalloc on the first long request, which crashes if it happens inside a captured CUDA Graph, blocking CG capture for continuation prefill. Separately, the decode stage2 reduction emitted float32 then launched a redundant `float16_copy` kernel to cast to query dtype.

## Example

At 1M context on L4 TP=4 (GPTQ-Int4 + TQ k8v4), per-layer dequant buffers consumed 58.6 GB vs 976 MB when shared — a 57.6 GB waste. The stage2 float16 cast added ~240K extra kernel launches per profile window.

## Use case

Serving long-context LLM inference with TurboQuant KV-cache quantization, where requests are prefilled in chunks (continuation prefill) and decoded incrementally.

## Resources

Primarily HBM/GPU memory (dequant buffers scaling with layers×context), plus kernel-launch overhead (redundant float16_copy, memset .zero_() kernels, per-call host→device tensor creation) and memory bandwidth (fp32 rotation matmul).

## Type of users

Users serving long-context workloads with TurboQuant KV quantization on multi-GPU TP setups (e.g. L4 TP=4), especially anyone wanting CUDA Graph capture enabled for the prefill path.

## Kind of inputs

Long sequences (8K–1M tokens) processed via chunked/continuation prefill, FP16 query dtype, many quantized attention layers; prefix-cached or multi-chunk requests that exercise the continuation path.

## Mechanism with the issue

`_continuation_prefill` in turboquant_attn.py stored dequant buffers as per-layer attributes that only grew, never shared, so each of the 60 layers held its own max-length buffer. `_decode_attention` pulled `output_buf` from layer attributes allocated as float32 while stage2 could now write float16, causing a dtype mismatch. `_fwd_kernel_stage2` stored `acc / e_sum` as float32, forcing a downstream `.to(query.dtype)` copy kernel.

# Solution

## Fix in words

The diff removes `_init_turboquant_buffers` and the per-layer buffer registration from attention.py, moving centroid init into the backend's `_ensure_on_device`. In `_continuation_prefill`, per-layer `_tq_k/v_dequant_buf` are replaced by `current_workspace_manager().get_simultaneous(...)` so the FP16 dequant buffers are drawn from the global per-ubatch workspace pool (shared across all layers, fixed-size for CUDA Graph capture, counted in memory profiling); the redundant `.zero_()` and `.contiguous()` copies are dropped, rotation is done in fp16 via `_tq_Pi_half`, and `k_full`/`v_full` are pre-allocated and slice-filled instead of using `torch.cat`. `_decode_attention` now acquires `mid_o_buf/output_buf/lse_buf` from the workspace with `output_buf` in query dtype. `_fwd_kernel_stage2` gains an `OUTPUT_FP16` constexpr that casts `acc / e_sum` to float16 on the final `tl.store`, and `triton_turboquant_decode_attention` outputs directly in query dtype — eliminating the separate float16_copy kernel.

## Author's solution

The author shares dequant buffers across all layers via `current_workspace_manager().get_simultaneous()`, extending the shared-decode-buffer pattern from PR #40655 into the continuation-prefill path so buffers are pre-allocated at profiling time (fixed addresses enabling CUDA Graph capture) and saving 57 GB at 1M context. They remove the redundant `.zero_()`, do the inverse Hadamard rotation in fp16 (the ±1/√D matrix is fp16-exact), and pre-size `k_full`/`v_full` to eliminate `.contiguous()` and `torch.cat` allocation storms (-14% TTFT at 15K). For decode, they add an `OUTPUT_FP16` constexpr to stage2 so it casts on store, eliminating the float16_copy kernel, and fix the resulting `output_buf` dtype mismatch by allocating it in query dtype.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/attention/backends/turboquant_attn.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/triton_turboquant_decode.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/attention/attention.py b/vllm/model_executor/layers/attention/attention.py
index bc10538ca6c9..db9ae2bbda34 100644
--- a/vllm/model_executor/layers/attention/attention.py
+++ b/vllm/model_executor/layers/attention/attention.py
@@ -386,10 +386,6 @@ def __init__(
         # Initialize KV cache quantization attributes
         _init_kv_cache_quant(self, quant_config, prefix)
 
-        # Initialize TurboQuant buffers (Pi, S, centroids) if tq cache dtype
-        if kv_cache_dtype.startswith("turboquant_"):
-            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)
-
         # for attn backends supporting query quantization
         self.query_quant = None
         if (
@@ -410,50 +406,6 @@ def __init__(
                 else GroupShape.PER_TENSOR,
             )
 
-    def _init_turboquant_buffers(
-        self, cache_dtype: str, head_size: int, prefix: str
-    ) -> None:
-        """Initialize TurboQuant centroids for Lloyd-Max quantization."""
-        from vllm.model_executor.layers.quantization.turboquant.centroids import (
-            get_centroids,
-        )
-        from vllm.model_executor.layers.quantization.turboquant.config import (
-            TurboQuantConfig,
-        )
-
-        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
-
-        self.register_buffer(
-            "_tq_centroids",
-            get_centroids(head_size, tq_config.centroid_bits),
-        )
-        self._tq_config = tq_config
-
-        # Pre-allocate decode intermediate buffers so model.to(device) moves
-        # them to GPU *before* the memory profiler runs.  Without this the
-        # profiler gives all free memory to KV cache blocks and the first
-        # decode OOMs when these buffers are lazily allocated.
-        _vllm_cfg = get_current_vllm_config()
-        B = _vllm_cfg.scheduler_config.max_num_seqs
-        Hq = self.num_heads
-        S = _vllm_cfg.attention_config.tq_max_kv_splits_for_cuda_graph
-        D = head_size
-        self.register_buffer(
-            "_tq_mid_o_buf",
-            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),
-            persistent=False,
-        )
-        self.register_buffer(
-            "_tq_output_buf",
-            torch.empty(B, Hq, D, dtype=torch.float32),
-            persistent=False,
-        )
-        self.register_buffer(
-            "_tq_lse_buf",
-            torch.empty(B, Hq, dtype=torch.float32),
-            persistent=False,
-        )
-
     def forward(
         self,
         query: torch.Tensor,
diff --git a/vllm/v1/attention/backends/turboquant_attn.py b/vllm/v1/attention/backends/turboquant_attn.py
index a0bcc252d857..af2d0fb0830f 100644
--- a/vllm/v1/attention/backends/turboquant_attn.py
+++ b/vllm/v1/attention/backends/turboquant_attn.py
@@ -26,6 +26,9 @@
 
 from vllm.config import get_current_vllm_config
 from vllm.config.cache import CacheDType
+from vllm.model_executor.layers.quantization.turboquant.centroids import (
+    get_centroids,
+)
 from vllm.triton_utils import triton
 from vllm.v1.attention.backend import (
     AttentionBackend,
@@ -49,6 +52,10 @@
     triton_turboquant_decode_attention,
 )
 from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
+from vllm.v1.worker.workspace import (
+    current_workspace_manager,
+    is_workspace_manager_initialized,
+)
 
 _HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
 if _HAS_FLASH_ATTN:
@@ -335,9 +342,15 @@ def _ensure_on_device(self, layer, device):
             H = _build_hadamard(D, str(device))
             layer._tq_PiT = H
             layer._tq_Pi = H
+            # fp16 copy for rotation in continuation prefill path
+            layer._tq_Pi_half = H.to(torch.float16)
 
-            c = layer._tq_centroids.to(device=device, dtype=torch.float32)
-            c_sorted, _ = c.sort()
+            # Centroids for Lloyd-Max quantization.
+            layer._tq_centroids = get_centroids(D, self.tq_config.centroid_bits).to(
+                device=device, dtype=torch.float32
+            )
+
+            c_sorted, _ = layer._tq_centroids.sort()
             layer._tq_midpoints = (c_sorted[:-1] + c_sorted[1:]) / 2
             layer._tq_cached = True
 
@@ -572,7 +585,17 @@ def _prefill_attention(
 
         # Pre-allocate cu_seqlens for single-request flash_attn calls
         # to avoid per-request host→device tensor creation.
-        _cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
+        if not hasattr(self, "_cu_2"):
+            self._cu_2 = torch.zeros(2, device=query.device, dtype=torch.int32)
+        # Cache arange on self (avoid per-call kernel launch).
+        _max_seq = attn_metadata.max_seq_len
+        _ac: torch.Tensor | None = getattr(self, "_arange_cache", None)
+        if _ac is None or _ac.shape[0] <= _max_seq:
+            _ac = torch.arange(
+                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
+            )
+            self._arange_cache = _ac
+        _arange_cache: torch.Tensor = _ac
 
         for i in range(num_reqs):
             q_start = qsl[i]
@@ -589,8 +612,8 @@ def _prefill_attention(
             if q_len == seq_len:
                 # First-chunk prefill: all K/V are in the current batch.
                 if _HAS_FLASH_ATTN:
-                    _cu_2[1] = q_len
-                    cu = _cu_2
+                    self._cu_2[1] = q_len
+                    cu = self._cu_2
                     out = self._flash_attn_varlen(
                         q=q_seq,
                         k=k_seq,
@@ -622,12 +645,8 @@ def _prefill_attention(
                 if q_len <= _CONTINUATION_DECODE_THRESHOLD:
                     # Fast path: treat each query as a decode request
                     # with incremental seq_lens for causal masking.
-                    synth_seq_lens = torch.arange(
-                        cached_len + 1,
-                        seq_len + 1,
-                        device=query.device,
-                        dtype=attn_metadata.seq_lens.dtype,
-                    )
+                    # Slice from pre-built arange (no kernel launch)
+                    synth_seq_lens = _arange_cache[cached_len + 1 : seq_len + 1]
                     synth_bt = attn_metadata.block_table[i : i + 1].expand(q_len, -1)
                     out = triton_turboquant_decode_attention(
                         query=q_seq,
@@ -695,16 +714,17 @@ def _continuation_prefill(
         # Reuse cached buffers to avoid per-call allocation (~16MB at 8K).
         alloc_len = math.ceil(cached_len / block_size) * block_size
         buf_shape = (1, Hk, alloc_len, D)
-        k_buf = getattr(layer, "_tq_k_dequant_buf", None)
-        if k_buf is None or k_buf.shape[2] < alloc_len:
-            k_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
-            v_buf = torch.empty(buf_shape, dtype=torch.float16, device=device)
-            layer._tq_k_dequant_buf = k_buf
-            layer._tq_v_dequant_buf = v_buf
-        else:
-            v_buf = layer._tq_v_dequant_buf
-        k_cached = k_buf[:, :, :alloc_len, :].zero_()
-        v_cached = v_buf[:, :, :alloc_len, :].zero_()
+        # Use WorkspaceManager for dequant buffers.
+        # Shared across all layers — saves 60× memory at long context.
+        # Required for CUDA Graph capture (per-layer growth incompatible with CG).
+        k_buf, v_buf = current_workspace_manager().get_simultaneous(
+            (buf_shape, torch.float16),
+            (buf_shape, torch.float16),
+        )
+        # Skip .zero_() — kernel writes all positions up to cached_len,
+        # and we only read [:cached_len] afterwards.
+        k_cached = k_buf[:, :, :alloc_len, :]
+        v_cached = v_buf[:, :, :alloc_len, :]
 
         grid = (alloc_len, 1 * Hk)
         _tq_full_dequant_kv[grid](
@@ -740,29 +760,41 @@ def _continuation_prefill(
 
         # Inverse-rotate MSE keys back to original space
         if not self.tq_config.key_fp8:
-            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D).float()
-            k_flat = k_flat @ Pi
-            k_cached_trim = (
-                k_flat.to(torch.float16).reshape(Hk, cached_len, D).transpose(0, 1)
-            )  # (cached_len, Hk, D)
+            # fp16 matmul for rotation (2× less bandwidth, uses fp16 tensor cores)
+            Pi_half = layer._tq_Pi_half
+            k_flat = k_cached[0, :, :cached_len, :].reshape(-1, D)
+            k_flat = k_flat @ Pi_half
+            k_cached_trim = k_flat.reshape(Hk, cached_len, D).transpose(
+                0, 1
+            )  # (cached_len, Hk, D) — already fp16
         else:
-            k_cached_trim = (
-                k_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
+            k_cached_trim = k_cached[0, :, :cached_len, :].transpose(
+                0, 1
             )  # (cached_len, Hk, D)
 
-        v_cached_trim = (
-            v_cached[0, :, :cached_len, :].transpose(0, 1).contiguous()
-        )  # (cached_len, Hk, D)
+        # Skip .contiguous() — the copy into k_full/v_full handles layout
+        v_cached_trim = v_cached[0, :, :cached_len, :].transpose(0, 1)
 
         # Concatenate cached + current chunk K/V (match query dtype)
+        # Pre-allocate full K/V buffer, copy into slices (no cat alloc)
         qdtype = query.dtype
-        k_full = torch.cat([k_cached_trim.to(qdtype), key_chunk], dim=0)
-        v_full = torch.cat([v_cached_trim.to(qdtype), val_chunk], dim=0)
+        k_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
+        v_full = torch.empty(seq_len, Hk, D, dtype=qdtype, device=device)
+        k_full[:cached_len] = k_cached_trim.to(qdtype)
+        k_full[cached_len:] = key_chunk
+        v_full[:cached_len] = v_cached_trim.to(qdtype)
+        v_full[cached_len:] = val_chunk
 
         # Attention: q_len queries attending to seq_len K/V with causal mask
         if _HAS_FLASH_ATTN:
-            cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
-            cu_seqlens_k = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
+            # Reuse pre-allocated cu_seqlens (avoid host→device transfer)
+            if not hasattr(self, "_cu_2_q"):
+                self._cu_2_q = torch.zeros(2, device=device, dtype=torch.int32)
+                self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
+            self._cu_2_q[1] = q_len
+            self._cu_2_k[1] = seq_len
+            cu_seqlens_q = self._cu_2_q
+            cu_seqlens_k = self._cu_2_k
             return self._flash_attn_varlen(
                 q=query,
                 k=k_full,
@@ -805,12 +837,23 @@ def _decode_attention(
         PiT: torch.Tensor | None = None,
         layer: torch.nn.Module | None = None,
     ) -> torch.Tensor:
-        # Grab cached decode buffers from the layer (lazily allocated).
+        # Acquire shared decode scratch buffers from WorkspaceManager.
+        # Layers execute sequentially so one set of buffers is sufficient.
+        # Falls back to kernel-internal allocation if workspace unavailable.
+        B = query.shape[0]
+        D = self.head_size
+        S = self.max_num_kv_splits
+        Hq = self.num_heads
         mid_o_buf = output_buf = lse_buf = None
-        if layer is not None:
-            mid_o_buf = getattr(layer, "_tq_mid_o_buf", None)
-            output_buf = getattr(layer, "_tq_output_buf", None)
-            lse_buf = getattr(layer, "_tq_lse_buf", None)
+        if is_workspace_manager_initialized():
+            # output_buf in query dtype — matches the in-kernel fp16 cast in stage2.
+            mid_o_buf, output_buf, lse_buf = (
+                current_workspace_manager().get_simultaneous(
+                    ((B, Hq, S, D + 1), torch.float32),
+                    ((B, Hq, D), query.dtype),
+                    ((B, Hq), torch.float32),
+                )
+            )
 
         result = triton_turboquant_decode_attention(
             query=query,
diff --git a/vllm/v1/attention/ops/triton_decode_attention.py b/vllm/v1/attention/ops/triton_decode_attention.py
index 8118db0da8cf..e1059b47bcba 100644
--- a/vllm/v1/attention/ops/triton_decode_attention.py
+++ b/vllm/v1/attention/ops/triton_decode_attention.py
@@ -551,6 +551,7 @@ def _fwd_kernel_stage2(
     NUM_KV_SPLITS: tl.constexpr,
     BLOCK_DV: tl.constexpr,
     Lv: tl.constexpr,
+    OUTPUT_FP16: tl.constexpr = 0,
 ):
     cur_batch = tl.program_id(0)
     cur_head = tl.program_id(1)
@@ -587,9 +588,12 @@ def _fwd_kernel_stage2(
             e_sum = e_sum * old_scale + exp_logic
             e_max = n_e_max
 
+    result = acc / e_sum
+    if OUTPUT_FP16:
+        result = result.to(tl.float16)
     tl.store(
         o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
-        acc / e_sum,
+        result,
         mask=mask_d,
     )
     lse_val = e_max + tl.log(e_sum)
diff --git a/vllm/v1/attention/ops/triton_turboquant_decode.py b/vllm/v1/attention/ops/triton_turboquant_decode.py
index a789f9be7bb2..3adaf2610d8d 100644
--- a/vllm/v1/attention/ops/triton_turboquant_decode.py
+++ b/vllm/v1/attention/ops/triton_turboquant_decode.py
@@ -588,10 +588,16 @@ def triton_turboquant_decode_attention(
     )
 
     # Stage 2: Reduce across KV splits
-    if output_buf is not None and output_buf.shape[0] >= B:
+    # Output in query dtype — eliminates float16_copy kernel after stage2
+    out_dtype = query.dtype
+    if (
+        output_buf is not None
+        and output_buf.shape[0] >= B
+        and output_buf.dtype == out_dtype
+    ):
         output = output_buf[:B, :Hq, :D]
     else:
-        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)
+        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
         if buf_holder is not None:
             buf_holder._tq_output_buf = output
     if lse_buf is not None and lse_buf.shape[0] >= B:
@@ -616,8 +622,9 @@ def triton_turboquant_decode_attention(
         NUM_KV_SPLITS=NUM_KV_SPLITS,
         BLOCK_DV=cfg["BLOCK_D"],
         Lv=D,
+        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
         num_warps=4,
         num_stages=2,
     )
 
-    return output.to(query.dtype)
+    return output  # already in query dtype

```
