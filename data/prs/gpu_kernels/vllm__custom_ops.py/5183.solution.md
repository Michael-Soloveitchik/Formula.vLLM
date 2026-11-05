# PR #5183: [Kernel] Switch fp8 layers to use the CUTLASS kernels

- field: gpu_kernels  | area: vllm__custom_ops.py
- author: tlrmchlsmth
- state: MERGED  created: 2024-06-01T18:55:56Z  merged: 2024-06-07T08:42:35Z
- url: https://github.com/vllm-project/vllm/pull/5183
- labels: 

# Issue

## Problem

FP8 linear layers used `torch._scaled_mm` for the quantized GEMM dequant step. This path is suboptimal for the matrix shapes seen in FP8 serving, leaving 5-15% end-to-end throughput on the table versus vLLM's own CUTLASS fp8 kernels. The code also had to pad the batch dimension to work around `torch._scaled_mm` being slow for small batch GEMMs.

## Example

Serving `neuralmagic/Meta-Llama-3-8B-Instruct-FP8`, switching the GEMM backend yielded 5-15% e2e performance improvement in the author's benchmarks across varied GEMM sizes.

## Use case

Online/offline inference serving of FP8-quantized LLMs where every decode/prefill token flows through the quantized linear layers.

## Resources

GPU compute (tensor-core FP8 GEMM throughput); the padding workaround also wasted HBM and compute on padded rows.

## Type of users

Users serving FP8-quantized models on Hopper (SM90) or Lovelace (SM89) GPUs with recent CUDA toolkits.

## Kind of inputs

FP8 dtype activations/weights at typical serving batch dimensions, especially small batch sizes where `torch._scaled_mm` underperforms and needed batch_dim_padding.

## Mechanism with the issue

In `Fp8LinearMethod.apply` (vllm/model_executor/layers/quantization/fp8.py), the GEMM went through `torch._scaled_mm` with `batch_dim_padding=17`. That kernel is not as fast as the CUTLASS fp8 GEMM_DQ for these shapes, and the padding to 17 rows is a workaround that adds overhead instead of using a kernel efficient at small batch.

# Solution

## Fix in words

Adds `cutlass_fp8_supported()` which checks device capability and CUDA version (SM90 needs CUDA >12.0, SM89 needs CUDA >12.4) and caches the result on the linear method at init. In `apply`, when `bias is None` and CUTLASS is supported, it quantizes without batch padding and calls `ops.cutlass_scaled_mm_dq` for the fused GEMM_DQ; otherwise it falls back to the original `torch._scaled_mm` path with batch_dim_padding=17. Also renames the `cutlass_scaled_mm_dq` wrapper args from `a_scales`/`b_scales` to `scale_a`/`scale_b` for a consistent keyword interface. Effect: eligible FP8 layers run the faster CUTLASS kernel, giving 5-15% e2e gains while preserving correctness where bias or unsupported hardware require the old path.

## Author's solution

Switch FP8 layers from `torch._scaled_mm` to vLLM's CUTLASS fp8 kernels when supported, based on observing 5-15% e2e improvement on `neuralmagic/Meta-Llama-3-8B-Instruct-FP8`. Benchmarks across GEMM sizes and e2e comparisons are linked in the PR for justification.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/_custom_ops.py`
- `vllm/model_executor/layers/quantization/fp8.py`

### Diff
```diff
diff --git a/vllm/_custom_ops.py b/vllm/_custom_ops.py
index 8a6f6d96d81f..4980a03c3b2d 100644
--- a/vllm/_custom_ops.py
+++ b/vllm/_custom_ops.py
@@ -177,7 +177,7 @@ def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
 
 # cutlass
 def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
-                         a_scales: torch.Tensor, b_scales: torch.Tensor,
+                         scale_a: torch.Tensor, scale_b: torch.Tensor,
                          out_dtype: Type[torch.dtype]) -> torch.Tensor:
     assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
     assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16)
@@ -186,7 +186,7 @@ def cutlass_scaled_mm_dq(a: torch.Tensor, b: torch.Tensor,
     n = b.shape[1]
     out = torch.empty((m, n), dtype=out_dtype, device=a.device)
 
-    vllm_ops.cutlass_scaled_mm_dq(out, a, b, a_scales, b_scales)
+    vllm_ops.cutlass_scaled_mm_dq(out, a, b, scale_a, scale_b)
 
     return out
 
diff --git a/vllm/model_executor/layers/quantization/fp8.py b/vllm/model_executor/layers/quantization/fp8.py
index bf3a59e3d709..136a64623d7f 100644
--- a/vllm/model_executor/layers/quantization/fp8.py
+++ b/vllm/model_executor/layers/quantization/fp8.py
@@ -17,6 +17,24 @@
 logger = init_logger(__name__)
 
 
+def cutlass_fp8_supported() -> bool:
+    capability = torch.cuda.get_device_capability()
+    capability = capability[0] * 10 + capability[1]
+    version = torch.version.cuda
+    version = version[0] * 10 + version[1]
+
+    # CUTLASS FP8 kernels need at least
+    #   CUDA 12.0 on SM90 systems (Hopper)
+    #   CUDA 12.4 on SM89 systems (Lovelace)
+    gpu_is_supported = False
+    if capability >= 900:
+        gpu_is_supported = version > 120
+    elif capability >= 890:
+        gpu_is_supported = version > 124
+
+    return gpu_is_supported
+
+
 class Fp8Config(QuantizationConfig):
     """Config class for FP8."""
 
@@ -92,6 +110,7 @@ class Fp8LinearMethod(LinearMethodBase):
 
     def __init__(self, quant_config: Fp8Config):
         self.quant_config = quant_config
+        self.cutlass_fp8_supported = cutlass_fp8_supported()
 
     def _create_scale_param(
         self,
@@ -233,25 +252,40 @@ def apply(self,
               layer: torch.nn.Module,
               x: torch.Tensor,
               bias: Optional[torch.Tensor] = None) -> torch.Tensor:
+
         # ops.scaled_fp8_quant supports both dynamic and static quant.
         #   If dynamic, layer.act_scale is None and x_scale computed from x.
         #   If static,  layer.act_scale is scalar and x_scale set to act_scale.
-        qinput, x_scale = ops.scaled_fp8_quant(x,
-                                               layer.act_scale,
-                                               batch_dim_padding=17)
-
-        # Fused GEMM_DQ -- note we padded the input above because
-        # torch._scaled_mm is more performant for matrices with
-        # batch dimension > 16. Note that this could change
-        # in the future.
-        output, _ = torch._scaled_mm(
-            qinput,
-            layer.weight,
-            out_dtype=x.dtype,
-            scale_a=x_scale,
-            scale_b=layer.weight_scale,
-            bias=bias,
-        )
+
+        if bias is None and self.cutlass_fp8_supported:
+            qinput, x_scale = ops.scaled_fp8_quant(x, layer.act_scale)
+
+            # Fused GEMM_DQ
+            output = ops.cutlass_scaled_mm_dq(
+                qinput,
+                layer.weight,
+                out_dtype=x.dtype,
+                scale_a=x_scale,
+                scale_b=layer.weight_scale,
+            )
+
+        else:
+            qinput, x_scale = ops.scaled_fp8_quant(x,
+                                                   layer.act_scale,
+                                                   batch_dim_padding=17)
+
+            # Fused GEMM_DQ -- note we padded the input above because
+            # torch._scaled_mm is more performant for matrices with
+            # batch dimension > 16. Note that this could change
+            # in the future.
+            output, _ = torch._scaled_mm(
+                qinput,
+                layer.weight,
+                out_dtype=x.dtype,
+                scale_a=x_scale,
+                scale_b=layer.weight_scale,
+                bias=bias,
+            )
 
         return torch.narrow(output, 0, 0, x.shape[0])
 

```
