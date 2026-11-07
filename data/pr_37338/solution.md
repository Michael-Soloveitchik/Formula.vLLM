# PR #37338: [Perf] [Bugfix] Fix Triton autotuning in inference for Qwen3.5

- field: gpu_kernels  | area: model_executor_layers
- author: arpera
- state: MERGED  created: 2026-03-17T19:03:23Z  merged: 2026-03-23T07:37:58Z
- url: https://github.com/vllm-project/vllm/pull/37338
- labels: bug, ready, qwen

# Issue

## Problem

PR #36599 added a GDN Triton warmup (`_warmup_prefill_kernels()`) meant to pre-compile and autotune FLA kernels before serving. But the warmup builds dummy tensors whose dtypes and flags don't match the real inference path. Triton's autotuner caches specializations keyed partly on tensor dtype/shape, so the warmup-generated cache entries never match real inference keys and are never reused. Autotuning therefore re-runs on the first live inference batch, which the warmup was explicitly designed to prevent.

## Example

Serving `Qwen/Qwen3.5-397B-A17B-FP8` with `-dp 8 --enable-expert-parallel` on 8x B200, the first inference batch triggers 746 Triton autotuning events across 6 FLA kernels (e.g. `chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_local_cumsum_scalar_kernel`), all occurring during live request handling instead of startup.

## Use case

Online serving of Qwen3.5 (GDN / gated delta-rule linear-attention) models where a warmup phase is expected to absorb all kernel compilation/autotuning before the server accepts traffic.

## Resources

Kernel compile/autotune time on the GPU critical path — autotuning stalls the first inference batch, inflating first-batch / TTFT latency rather than being amortized during idle startup.

## Type of users

Users serving large FP8 MoE / GDN-based models (Qwen3.5-397B-A17B-FP8) in multi-GPU data-parallel + expert-parallel setups on B200-class hardware who care about consistent low latency from the first request.

## Kind of inputs

The first real prefill batch after startup — variable-length sequences described by an `int32` `cu_seqlens`, `float32` gate `g` from `fused_gdn_gating()`, with `output_final_state=True`.

## Mechanism with the issue

In `qwen3_next.py` warmup (lines ~707–750), three tensors diverge from inference: gate `g` is `torch.randn(..., bfloat16)` instead of `fused_gdn_gating()`'s `float32` output; `cu_seqlens` is `torch.long` (int64) instead of `int32`; and `output_final_state=False` instead of `True`. Each mismatch produces a distinct Triton specialization key, so the FLA kernels autotuned during warmup are cache-missed at inference and re-autotuned.

# Solution

## Fix in words

The diff rewrites the warmup tensor construction in `qwen3_next.py` to match the real inference path: `g`/`beta` are produced via `fused_gdn_gating()` (yielding `float32`) instead of `torch.randn` bfloat16, `cu_seqlens` is built with `dtype=torch.int32` instead of `torch.long`, and `output_final_state` is set to `True`. With dtypes and flags aligned, Triton's autotuner cache keys generated during warmup now match those at inference, so cache entries are reused and no autotuning leaks into the first batch. Separately, across 10 FLA op files, `cu_seqlens` type annotations are corrected from `torch.LongTensor` to `torch.Tensor` (and doc examples from `dtype=torch.long` to `torch.int32`) to reflect the actual int32 runtime dtype — a no-op cleanup for consistency.

## Author's solution

Author produces the GDN warmup tensors with dtypes/flags matching real inference: generate `g, beta` via `fused_gdn_gating()` (float32) instead of `torch.randn` (bfloat16), use `torch.int32` for `cu_seqlens`, and set `output_final_state=True`. This makes Triton autotuner cache keys from warmup match inference, so all 746 previously-leaked autotuning events complete during startup (verified: benchmark autotuning drops from 746 to 0). Additionally corrects `cu_seqlens` annotations from `torch.LongTensor` to `torch.Tensor` across FLA ops as a cleanup with no performance impact.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fla/ops/chunk.py`
- `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`
- `vllm/model_executor/layers/fla/ops/chunk_o.py`
- `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py`
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
- `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`
- `vllm/model_executor/layers/fla/ops/index.py`
- `vllm/model_executor/layers/fla/ops/kda.py`
- `vllm/model_executor/layers/fla/ops/wy_fast.py`
- `vllm/model_executor/models/qwen3_next.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fla/ops/chunk.py b/vllm/model_executor/layers/fla/ops/chunk.py
index 40f8c3c2a167..9261885956e5 100644
--- a/vllm/model_executor/layers/fla/ops/chunk.py
+++ b/vllm/model_executor/layers/fla/ops/chunk.py
@@ -30,7 +30,7 @@ def chunk_gated_delta_rule_fwd(
     scale: float,
     initial_state: torch.Tensor,
     output_final_state: bool,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
 ):
     g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
     # obtain WY representation. u is actually the new v.
@@ -84,7 +84,7 @@ def forward(
         scale: float,
         initial_state: torch.Tensor,
         output_final_state: bool,
-        cu_seqlens: torch.LongTensor | None = None,
+        cu_seqlens: torch.Tensor | None = None,
         use_qk_l2norm_in_kernel: bool = False,
     ):
         if use_qk_l2norm_in_kernel:
@@ -117,7 +117,7 @@ def chunk_gated_delta_rule(
     scale: float = None,
     initial_state: torch.Tensor = None,
     output_final_state: bool = False,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = False,
 ):
     r"""
@@ -141,7 +141,7 @@ def chunk_gated_delta_rule(
             Default: `None`.
         output_final_state (Optional[bool]):
             Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
-        cu_seqlens (torch.LongTensor):
+        cu_seqlens (torch.Tensor):
             Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
             consistent with the FlashAttention API.
     Returns:
@@ -171,7 +171,7 @@ def chunk_gated_delta_rule(
         # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
         >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
         # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
-        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
+        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
         >>> o_var, ht_var = chunk_gated_delta_rule(
             q, k, v, g, beta,
             initial_state=h0,
diff --git a/vllm/model_executor/layers/fla/ops/chunk_delta_h.py b/vllm/model_executor/layers/fla/ops/chunk_delta_h.py
index 98a3d61e4360..ce60ca46f6c9 100644
--- a/vllm/model_executor/layers/fla/ops/chunk_delta_h.py
+++ b/vllm/model_executor/layers/fla/ops/chunk_delta_h.py
@@ -288,7 +288,7 @@ def chunk_gated_delta_rule_fwd_h(
     output_final_state: bool = False,
     chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
     save_new_value: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     # This kernel is slightly different from fla to support Q/K with different head numbers.
     # In fla, Q/K always have the same head number, so Hg is always equal to H.
diff --git a/vllm/model_executor/layers/fla/ops/chunk_o.py b/vllm/model_executor/layers/fla/ops/chunk_o.py
index 2ccf1d4e2549..130781276259 100644
--- a/vllm/model_executor/layers/fla/ops/chunk_o.py
+++ b/vllm/model_executor/layers/fla/ops/chunk_o.py
@@ -145,7 +145,7 @@ def chunk_fwd_o(
     h: torch.Tensor,
     g: torch.Tensor | None = None,  # cumsum of log decay
     scale: float | None = None,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     chunk_size: int = 64,
 ) -> torch.Tensor:
     B, T, Hg, K, V = *q.shape, v.shape[-1]
diff --git a/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py b/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py
index 7724fa513d92..31bd489ebd87 100644
--- a/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py
+++ b/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py
@@ -102,7 +102,7 @@ def chunk_scaled_dot_kkt_fwd(
     k: torch.Tensor,
     g: torch.Tensor | None = None,
     beta: torch.Tensor | None = None,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     chunk_size: int = 64,
     output_dtype: torch.dtype = torch.float32,
 ) -> torch.Tensor:
@@ -116,7 +116,7 @@ def chunk_scaled_dot_kkt_fwd(
             The beta tensor of shape `[B, T, H]`.
         g (torch.Tensor):
             The cumulative sum of the gate tensor of shape `[B, T, H]`. Default: `None`.
-        cu_seqlens (torch.LongTensor):
+        cu_seqlens (torch.Tensor):
             The cumulative sequence lengths of the input tensor.
             Default: None
         chunk_size (int):
diff --git a/vllm/model_executor/layers/fla/ops/fused_recurrent.py b/vllm/model_executor/layers/fla/ops/fused_recurrent.py
index f7b562f64771..17b59b5bce71 100644
--- a/vllm/model_executor/layers/fla/ops/fused_recurrent.py
+++ b/vllm/model_executor/layers/fla/ops/fused_recurrent.py
@@ -184,7 +184,7 @@ def fused_recurrent_gated_delta_rule_fwd(
     scale: float,
     initial_state: torch.Tensor,
     inplace_final_state: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     ssm_state_indices: torch.Tensor | None = None,
     num_accepted_tokens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = False,
@@ -489,7 +489,7 @@ def forward(
         scale: float,
         initial_state: torch.Tensor,
         inplace_final_state: bool = True,
-        cu_seqlens: torch.LongTensor | None = None,
+        cu_seqlens: torch.Tensor | None = None,
         ssm_state_indices: torch.Tensor | None = None,
         num_accepted_tokens: torch.Tensor | None = None,
         use_qk_l2norm_in_kernel: bool = False,
@@ -521,7 +521,7 @@ def fused_recurrent_gated_delta_rule(
     scale: float = None,
     initial_state: torch.Tensor = None,
     inplace_final_state: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     ssm_state_indices: torch.Tensor | None = None,
     num_accepted_tokens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = False,
@@ -549,7 +549,7 @@ def fused_recurrent_gated_delta_rule(
         inplace_final_state: bool:
             Whether to store the final state in-place to save memory.
             Default: `True`.
-        cu_seqlens (torch.LongTensor):
+        cu_seqlens (torch.Tensor):
             Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
             consistent with the FlashAttention API.
         ssm_state_indices (Optional[torch.Tensor]):
@@ -583,7 +583,7 @@ def fused_recurrent_gated_delta_rule(
         # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
         >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
         # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
-        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
+        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
         >>> o_var, ht_var = fused_gated_recurrent_delta_rule(
             q, k, v, g, beta,
             initial_state=h0,
diff --git a/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py b/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
index 414891fd8d69..07ed185413f6 100644
--- a/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
+++ b/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py
@@ -191,7 +191,7 @@ def fused_sigmoid_gating_delta_rule_update(
     scale: float = None,
     initial_state: torch.Tensor = None,
     inplace_final_state: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     ssm_state_indices: torch.Tensor | None = None,
     num_accepted_tokens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = False,
diff --git a/vllm/model_executor/layers/fla/ops/index.py b/vllm/model_executor/layers/fla/ops/index.py
index f023e1378bb8..810d32c18b85 100644
--- a/vllm/model_executor/layers/fla/ops/index.py
+++ b/vllm/model_executor/layers/fla/ops/index.py
@@ -15,14 +15,12 @@
 
 
 @tensor_cache
-def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
+def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
     return cu_seqlens[1:] - cu_seqlens[:-1]
 
 
 @tensor_cache
-def prepare_chunk_indices(
-    cu_seqlens: torch.LongTensor, chunk_size: int
-) -> torch.LongTensor:
+def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
     indices = torch.cat(
         [
             torch.arange(n)
@@ -33,9 +31,7 @@ def prepare_chunk_indices(
 
 
 @tensor_cache
-def prepare_chunk_offsets(
-    cu_seqlens: torch.LongTensor, chunk_size: int
-) -> torch.LongTensor:
+def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
     return torch.cat(
         [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
     ).cumsum(-1)
diff --git a/vllm/model_executor/layers/fla/ops/kda.py b/vllm/model_executor/layers/fla/ops/kda.py
index 460be44c8402..b8c07d1dc896 100644
--- a/vllm/model_executor/layers/fla/ops/kda.py
+++ b/vllm/model_executor/layers/fla/ops/kda.py
@@ -38,7 +38,7 @@ def fused_recurrent_kda_fwd(
     scale: float,
     initial_state: torch.Tensor,
     inplace_final_state: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     ssm_state_indices: torch.Tensor | None = None,
     num_accepted_tokens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = False,
@@ -116,7 +116,7 @@ def fused_recurrent_kda(
     initial_state: torch.Tensor = None,
     inplace_final_state: bool = True,
     use_qk_l2norm_in_kernel: bool = True,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     ssm_state_indices: torch.LongTensor | None = None,
     **kwargs,
 ) -> tuple[torch.Tensor, torch.Tensor]:
@@ -720,7 +720,7 @@ def chunk_kda_scaled_dot_kkt_fwd(
     gk: torch.Tensor | None = None,
     beta: torch.Tensor | None = None,
     scale: float | None = None,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     chunk_size: int = 64,
     output_dtype: torch.dtype = torch.float32,
 ) -> tuple[torch.Tensor, torch.Tensor]:
@@ -734,7 +734,7 @@ def chunk_kda_scaled_dot_kkt_fwd(
             The beta tensor of shape `[B, T, H]`.
         gk (torch.Tensor):
             The cumulative sum of the gate tensor of shape `[B, T, H, K]` applied to the key tensor. Default: `None`.
-        cu_seqlens (torch.LongTensor):
+        cu_seqlens (torch.Tensor):
             The cumulative sequence lengths of the input tensor.
             Default: None
         chunk_size (int):
@@ -964,7 +964,7 @@ def recompute_w_u_fwd(
     A: torch.Tensor,
     q: torch.Tensor | None = None,
     gk: torch.Tensor | None = None,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     B, T, H, K, V = *k.shape, v.shape[-1]
     BT = A.shape[-1]
@@ -1132,7 +1132,7 @@ def chunk_gla_fwd_o_gk(
     h: torch.Tensor,
     o: torch.Tensor,
     scale: float,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     chunk_size: int = 64,
 ):
     B, T, H, K, V = *q.shape, v.shape[-1]
@@ -1176,7 +1176,7 @@ def chunk_kda_fwd(
     scale: float,
     initial_state: torch.Tensor,
     output_final_state: bool,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
 ):
     chunk_size = 64
     g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
@@ -1236,7 +1236,7 @@ def chunk_kda(
     initial_state: torch.Tensor = None,
     output_final_state: bool = False,
     use_qk_l2norm_in_kernel: bool = False,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     **kwargs,
 ):
     if scale is None:
diff --git a/vllm/model_executor/layers/fla/ops/wy_fast.py b/vllm/model_executor/layers/fla/ops/wy_fast.py
index a66ec1d60d66..6baa08ab4996 100644
--- a/vllm/model_executor/layers/fla/ops/wy_fast.py
+++ b/vllm/model_executor/layers/fla/ops/wy_fast.py
@@ -122,7 +122,7 @@ def recompute_w_u_fwd(
     beta: torch.Tensor,
     g_cumsum: torch.Tensor,
     A: torch.Tensor,
-    cu_seqlens: torch.LongTensor | None,
+    cu_seqlens: torch.Tensor | None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     B, T, Hg, K, V = *k.shape, v.shape[-1]
     H = v.shape[-2]
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index cfd4c7a56b43..35474a07a617 100644
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -116,7 +116,7 @@ def fi_chunk_gated_delta_rule(
     beta: torch.Tensor,
     initial_state: torch.Tensor,
     output_final_state: bool,
-    cu_seqlens: torch.LongTensor | None = None,
+    cu_seqlens: torch.Tensor | None = None,
     use_qk_l2norm_in_kernel: bool = True,
 ):
     from flashinfer.gdn_prefill import (
@@ -178,7 +178,7 @@ def forward_cuda(
         beta: torch.Tensor,
         initial_state: torch.Tensor,
         output_final_state: bool,
-        cu_seqlens: torch.LongTensor | None = None,
+        cu_seqlens: torch.Tensor | None = None,
         use_qk_l2norm_in_kernel: bool = True,
     ):
         return fi_chunk_gated_delta_rule(
@@ -202,7 +202,7 @@ def forward_native(
         beta: torch.Tensor,
         initial_state: torch.Tensor,
         output_final_state: bool,
-        cu_seqlens: torch.LongTensor | None = None,
+        cu_seqlens: torch.Tensor | None = None,
         use_qk_l2norm_in_kernel: bool = True,
     ):
         return fla_chunk_gated_delta_rule(
@@ -707,8 +707,13 @@ def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
             v = torch.randn(
                 1, T, num_v_heads, self.head_v_dim, device=device, dtype=dtype
             )
-            g = torch.randn(1, T, num_v_heads, device=device, dtype=dtype)
-            beta = torch.randn(1, T, num_v_heads, device=device, dtype=dtype)
+            # NOTE: g and beta must have the same dtypes as during
+            # inference, so we construct them with the same function
+            # (fused_gdn_gating). dummy_a and dummy_b are throwaway
+            # inputs required by that function.
+            dummy_a = torch.randn(T, num_v_heads, device=device, dtype=dtype)
+            dummy_b = torch.randn(T, num_v_heads, device=device, dtype=dtype)
+            g, beta = fused_gdn_gating(self.A_log, dummy_a, dummy_b, self.dt_bias)
             state = torch.zeros(
                 1,
                 num_v_heads,
@@ -717,7 +722,7 @@ def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
                 device=device,
                 dtype=state_dtype,
             )
-            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.long)
+            cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
 
             try:
                 self.chunk_gated_delta_rule(
@@ -727,7 +732,7 @@ def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
                     g=g,
                     beta=beta,
                     initial_state=state,
-                    output_final_state=False,
+                    output_final_state=True,
                     cu_seqlens=cu_seqlens,
                     use_qk_l2norm_in_kernel=True,
                 )
@@ -747,7 +752,7 @@ def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
                     self.prefix,
                 )
             finally:
-                del q, k, v, g, beta, state, cu_seqlens
+                del q, k, v, dummy_a, dummy_b, g, beta, state, cu_seqlens
 
         torch.accelerator.empty_cache()
 

```
