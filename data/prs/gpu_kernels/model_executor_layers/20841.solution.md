# PR #20841: [Perf] Use Triton instead of Torch for DeepGEMM Per Token Group Quant

- field: gpu_kernels  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2025-07-11T20:20:12Z  merged: 2025-07-13T02:38:45Z
- url: https://github.com/vllm-project/vllm/pull/20841
- labels: performance, ready, deepseek

# Issue

## Problem

Per-token group FP8 quantization in the DeepGEMM MoE path routed through a wrapper (`per_token_group_cast_to_fp8`) that, on Blackwell, called DeepGEMM's native Torch-based `per_token_cast_to_fp8` instead of vLLM's Triton kernel. The Torch path was slower and forced a dispatch branch keyed on `is_blackwell_deep_gemm_used()`. The Triton kernel also lacked UE8M0 scale rounding, so it could not replace the Torch path on Blackwell without first gaining that capability.

## Example

Running Qwen3-30B-A3B-FP8 throughput bench (input-len 1000, output-len 100, expert-parallel, enforce-eager) with VLLM_USE_DEEP_GEMM=1 gave 20.70 req/s before; the Triton quant path lifted it to 23.85 req/s (~15% gain).

## Use case

FP8-quantized MoE inference served through the DeepGEMM fused-expert path, where activations are per-token-group quantized to FP8 before each grouped GEMM.

## Resources

GPU compute and kernel-launch efficiency during the activation-quantization step feeding block-scaled FP8 GEMMs.

## Type of users

Users serving FP8 block-quantized MoE models with DeepGEMM enabled, including on Blackwell (ue8m0 scale format) hardware.

## Kind of inputs

BF16/FP32 activation tensors with inner dim divisible by group size (128), quantized per token-group to FP8 with per-block scales.

## Mechanism with the issue

`vllm/utils/deep_gemm.py` exposed `per_token_group_cast_to_fp8`, which chose DeepGEMM's `per_token_cast_to_fp8` (Torch) when `is_blackwell_deep_gemm_used()` else fell back to the Triton `per_token_group_quant_fp8`. `deep_gemm_moe.py` and `fused_moe/utils._fp8_quantize` both went through this branch, incurring the slower Torch op on Blackwell.

# Solution

## Fix in words

Diff deletes the `per_token_group_cast_to_fp8` wrapper and the `_per_token_cast_impl` resolution/exports in `deep_gemm.py`, and rewires all callers (`deep_gemm_moe.py`, `fused_moe/utils.py`, tests) to call Triton `per_token_group_quant_fp8` directly — removing the Blackwell-vs-non-Blackwell branch in `_fp8_quantize`. To keep Blackwell correctness, the two Triton kernels `_per_token_group_quant_fp8` and `_per_token_group_quant_fp8_colmajor` gain a `use_ue8m0: tl.constexpr` flag; when set, the scale is rounded to a power of two via `y_s = exp2(ceil(log2(scale_raw)))` instead of raw `absmax/fp8_max`. `per_token_group_quant_fp8` passes `use_ue8m0=is_blackwell_deep_gemm_used()` to both kernel launches. Net effect: one Triton quant kernel serves all hardware, matching DeepGEMM's UE8M0 scaling on Blackwell while being faster than the Torch path.

## Author's solution

Follow-up to PR #20087: use Triton instead of Torch for Per Token Group Quant. Author verified gsm8k accuracy held (0.8476/0.8855 flexible/strict with deepgemm) and measured Qwen3-30B-A3B-FP8 throughput rising from 20.70 to 23.85 req/s on one card, guessing R1 could improve ~15%+.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/kernels/moe/test_deepgemm.py`
- `tests/kernels/quantization/test_block_fp8.py`
- `vllm/model_executor/layers/fused_moe/deep_gemm_moe.py`
- `vllm/model_executor/layers/fused_moe/utils.py`
- `vllm/model_executor/layers/quantization/utils/fp8_utils.py`
- `vllm/utils/deep_gemm.py`

### Diff
```diff
diff --git a/tests/kernels/moe/test_deepgemm.py b/tests/kernels/moe/test_deepgemm.py
index 6a04edafd96c..1460fdd3aeaf 100644
--- a/tests/kernels/moe/test_deepgemm.py
+++ b/tests/kernels/moe/test_deepgemm.py
@@ -13,9 +13,10 @@
 
 # vLLM fused-expert reference (Triton fallback + DeepGEMM option)
 from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
+from vllm.model_executor.layers.quantization.utils.fp8_utils import (
+    per_token_group_quant_fp8)
 from vllm.utils import has_deep_gemm
-from vllm.utils.deep_gemm import (calc_diff, per_block_cast_to_fp8,
-                                  per_token_group_cast_to_fp8)
+from vllm.utils.deep_gemm import calc_diff, per_block_cast_to_fp8
 
 BLOCK_SIZE = [128, 128]
 
@@ -81,7 +82,7 @@ def run_single_case(m, n, k, topk, num_experts, block_size):
     """
     tokens_bf16 = torch.randn(
         m, k, device="cuda", dtype=torch.bfloat16).clamp_min_(-1).clamp_max_(1)
-    _, a1_scale = per_token_group_cast_to_fp8(tokens_bf16, block_size[1])
+    _, a1_scale = per_token_group_quant_fp8(tokens_bf16, block_size[1])
 
     # expert weight tensors
     w1, w2, w1_s, w2_s = make_block_quant_fp8_weights(num_experts, n, k,
diff --git a/tests/kernels/quantization/test_block_fp8.py b/tests/kernels/quantization/test_block_fp8.py
index 97b5102dd478..26aa8d652e63 100644
--- a/tests/kernels/quantization/test_block_fp8.py
+++ b/tests/kernels/quantization/test_block_fp8.py
@@ -15,8 +15,7 @@
     w8a8_block_fp8_matmul)
 from vllm.platforms import current_platform
 from vllm.utils import has_deep_gemm
-from vllm.utils.deep_gemm import (fp8_gemm_nt, per_block_cast_to_fp8,
-                                  per_token_group_cast_to_fp8)
+from vllm.utils.deep_gemm import fp8_gemm_nt, per_block_cast_to_fp8
 
 if current_platform.get_device_capability() < (9, 0):
     pytest.skip("FP8 Triton requires CUDA 9.0 or higher",
@@ -117,7 +116,7 @@ def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
     A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
     B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
 
-    A_fp8, As_fp8 = per_token_group_cast_to_fp8(A_fp32, block_size[1])
+    A_fp8, As_fp8 = per_token_group_quant_fp8(A_fp32, block_size[1])
     B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32)
 
     As = As_fp8.to(torch.float32)
diff --git a/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py b/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py
index 4c0e6665bdc6..cb07c1c72b27 100644
--- a/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py
+++ b/vllm/model_executor/layers/fused_moe/deep_gemm_moe.py
@@ -15,9 +15,10 @@
 from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
     TopKWeightAndReduceDelegate)
 from vllm.model_executor.layers.fused_moe.utils import _resize_cache
+from vllm.model_executor.layers.quantization.utils.fp8_utils import (
+    per_token_group_quant_fp8)
 from vllm.utils import has_deep_gemm, round_up
-from vllm.utils.deep_gemm import (m_grouped_fp8_gemm_nt_contiguous,
-                                  per_token_group_cast_to_fp8)
+from vllm.utils.deep_gemm import m_grouped_fp8_gemm_nt_contiguous
 
 logger = init_logger(__name__)
 
@@ -170,10 +171,10 @@ def apply(
         self.activation(activation, act_out, mm1_out.view(-1, N))
 
         a2q_scale: Optional[torch.Tensor] = None
-        a2q, a2q_scale = per_token_group_cast_to_fp8(act_out,
-                                                     self.block_shape[1],
-                                                     column_major_scales=True,
-                                                     out_q=quant_out)
+        a2q, a2q_scale = per_token_group_quant_fp8(act_out,
+                                                   self.block_shape[1],
+                                                   column_major_scales=True,
+                                                   out_q=quant_out)
 
         m_grouped_fp8_gemm_nt_contiguous((a2q, a2q_scale), (w2, w2_scale),
                                          mm2_out, expert_ids)
diff --git a/vllm/model_executor/layers/fused_moe/utils.py b/vllm/model_executor/layers/fused_moe/utils.py
index 6638f423a32e..c120d964b3cd 100644
--- a/vllm/model_executor/layers/fused_moe/utils.py
+++ b/vllm/model_executor/layers/fused_moe/utils.py
@@ -15,8 +15,6 @@
 from vllm.platforms import current_platform
 from vllm.triton_utils import tl, triton
 from vllm.utils import cdiv
-from vllm.utils.deep_gemm import (is_blackwell_deep_gemm_used,
-                                  per_token_group_cast_to_fp8)
 
 
 @triton.jit
@@ -119,10 +117,7 @@ def _fp8_quantize(
         assert not per_act_token
         assert len(block_shape) == 2
         _, block_k = block_shape[0], block_shape[1]
-        if is_blackwell_deep_gemm_used():
-            A, A_scale = per_token_group_cast_to_fp8(A, block_k)
-        else:
-            A, A_scale = per_token_group_quant_fp8(A, block_k)
+        A, A_scale = per_token_group_quant_fp8(A, block_k)
         assert cdiv(A.size(-1), block_k) == A_scale.size(-1)
 
     return A, A_scale
diff --git a/vllm/model_executor/layers/quantization/utils/fp8_utils.py b/vllm/model_executor/layers/quantization/utils/fp8_utils.py
index 1780cc5de2d5..9c78dea17e5c 100644
--- a/vllm/model_executor/layers/quantization/utils/fp8_utils.py
+++ b/vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -20,6 +20,7 @@
 from vllm.platforms import current_platform
 from vllm.triton_utils import tl, triton
 from vllm.utils import cdiv, direct_register_custom_op, has_deep_gemm
+from vllm.utils.deep_gemm import is_blackwell_deep_gemm_used
 
 logger = init_logger(__name__)
 
@@ -256,6 +257,7 @@ def _per_token_group_quant_fp8(
     # Information for float8
     fp8_min,
     fp8_max,
+    use_ue8m0: tl.constexpr,
     # Meta-parameters
     BLOCK: tl.constexpr,
 ):
@@ -285,7 +287,8 @@ def _per_token_group_quant_fp8(
     y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
     # Quant
     _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
-    y_s = _absmax / fp8_max
+    scale_raw = _absmax / fp8_max
+    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
     y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
 
     tl.store(y_q_ptr + cols, y_q, mask=mask)
@@ -309,6 +312,7 @@ def _per_token_group_quant_fp8_colmajor(
     # Information for float8
     fp8_min,
     fp8_max,
+    use_ue8m0: tl.constexpr,
     # Meta-parameters
     BLOCK: tl.constexpr,
 ):
@@ -347,7 +351,8 @@ def _per_token_group_quant_fp8_colmajor(
     y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
     # Quant
     _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
-    y_s = _absmax / fp8_max
+    scale_raw = _absmax / fp8_max
+    y_s = tl.math.exp2(tl.ceil(tl.log2(scale_raw))) if use_ue8m0 else scale_raw
     y_q = tl.clamp(y / y_s, fp8_min, fp8_max).to(y_q_ptr.dtype.element_ty)
 
     tl.store(y_q_ptr + cols, y_q, mask=mask)
@@ -373,9 +378,11 @@ def per_token_group_quant_fp8(
         is supported for now.
         column_major_scales: Outputs scales in column major.
         out_q: Optional output tensor. If not provided, function will create.
-    Returns:
         tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
         scaling factor for quantization.
+    Returns:
+        tuple[torch.Tensor, torch.Tensor]: The quantized tensor and the
+        scaling factor.
     """
     dtype = current_platform.fp8_dtype() if dtype is None else dtype
     assert (x.shape[-1] % group_size == 0), (
@@ -418,6 +425,7 @@ def per_token_group_quant_fp8(
             eps,
             fp8_min=fp8_min,
             fp8_max=fp8_max,
+            use_ue8m0=is_blackwell_deep_gemm_used(),
             BLOCK=BLOCK,
             num_warps=num_warps,
             num_stages=num_stages,
@@ -433,6 +441,7 @@ def per_token_group_quant_fp8(
             eps,
             fp8_min=fp8_min,
             fp8_max=fp8_max,
+            use_ue8m0=is_blackwell_deep_gemm_used(),
             BLOCK=BLOCK,
             num_warps=num_warps,
             num_stages=num_stages,
diff --git a/vllm/utils/deep_gemm.py b/vllm/utils/deep_gemm.py
index 1684d6754f50..56326c9315ba 100644
--- a/vllm/utils/deep_gemm.py
+++ b/vllm/utils/deep_gemm.py
@@ -49,7 +49,6 @@ def _resolve_symbol(module, new: str, old: str) -> Callable[..., Any] | None:
     _fp8_gemm_nt_impl: Callable[..., Any] | None = None
     _grouped_impl: Callable[..., Any] | None = None
     _grouped_masked_impl: Callable[..., Any] | None = None
-    _per_token_cast_impl: Callable[..., Any] | None = None
     _per_block_cast_impl: Callable[..., Any] | None = None
 else:
     _dg = importlib.import_module("deep_gemm")  # type: ignore
@@ -74,12 +73,9 @@ def _resolve_symbol(module, new: str, old: str) -> Callable[..., Any] | None:
     try:
         _math_mod = importlib.import_module(
             "deep_gemm.utils.math")  # type: ignore
-        _per_token_cast_impl = getattr(_math_mod, "per_token_cast_to_fp8",
-                                       None)
         _per_block_cast_impl = getattr(_math_mod, "per_block_cast_to_fp8",
                                        None)
     except ModuleNotFoundError:
-        _per_token_cast_impl = None
         _per_block_cast_impl = None
 
 
@@ -101,22 +97,6 @@ def fp8_m_grouped_gemm_nt_masked(*args, **kwargs):
     return _grouped_masked_impl(*args, **kwargs)
 
 
-def per_token_group_cast_to_fp8(x, group_size, *args, **kwargs):
-    """Wrapper for token-wise FP8 quantisation.
-
-    • If DeepGEMM provides ``per_token_cast_to_fp8`` (new API), use it.
-    • Otherwise, fall back to vLLM's ``per_token_group_quant_fp8``
-    """
-
-    if _per_token_cast_impl is not None and is_blackwell_deep_gemm_used():
-        assert group_size == 128, "group_size must be 128 for deepgemm"
-        return _per_token_cast_impl(x)
-
-    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
-        per_token_group_quant_fp8 as _ptg)
-    return _ptg(x, group_size, *args, **kwargs)
-
-
 def per_block_cast_to_fp8(x, *args, **kwargs):
     if _per_block_cast_impl is not None and is_blackwell_deep_gemm_used():
         return _per_block_cast_impl(x)
@@ -146,7 +126,6 @@ def calc_diff(x: torch.Tensor, y: torch.Tensor):
     "fp8_gemm_nt",
     "m_grouped_fp8_gemm_nt_contiguous",
     "fp8_m_grouped_gemm_nt_masked",
-    "per_token_group_cast_to_fp8",
     "per_block_cast_to_fp8",
     "is_blackwell_deep_gemm_used",
 ]

```
