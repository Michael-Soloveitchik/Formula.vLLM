# PR #24783: [Perf] Fix DeepGEMM Contiguous Layout Issue, 5.5% Throughput Improvement

- field: quantization  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2025-09-12T21:41:33Z  merged: 2025-09-14T15:20:17Z
- url: https://github.com/vllm-project/vllm/pull/24783
- labels: performance, ready, deepseek

# Issue

## Problem

In FP8 weight post-load processing for DeepGEMM, `get_col_major_tma_aligned_tensor` returns a correctly TMA-aligned column-major tensor, but the code immediately calls `.contiguous()` on it. `.contiguous()` re-materializes the tensor in row-major layout, undoing the column-major alignment the function just produced. This forces DeepGEMM to operate on a layout it does not expect, adding runtime overhead.

## Example

Serving Qwen/Qwen3-30B-A3B-FP8 with `VLLM_USE_DEEP_GEMM=1` and expert parallelism, throughput was 41.92 req/s on main vs 44.26 req/s after the fix (~5.5% gain) at input-len 1000, output-len 100.

## Use case

FP8-quantized MoE inference served through the DeepGEMM grouped-GEMM backend, where weight scale tensors must be column-major TMA-aligned for the kernel.

## Resources

GPU compute and HBM bandwidth — the wrong memory layout triggers extra layout conversion / inefficient kernel access at runtime per forward pass.

## Type of users

Users serving FP8-quantized MoE models (e.g. Qwen3-30B-A3B-FP8) on the DeepGEMM path, often with expert parallelism on NVIDIA GPUs supporting TMA.

## Kind of inputs

FP8 checkpoints with col-major weight scale inverse tensors (`w13_weight_scale_inv`, `w2_weight_scale_inv`) that get routed through DeepGEMM; any batch/seq triggers it since overhead is per-layer per-forward.

## Mechanism with the issue

In `Fp8MoEMethod.process_weights_after_loading` (vllm/model_executor/layers/quantization/fp8.py), two code paths wrap `get_col_major_tma_aligned_tensor(...)` with `.contiguous()`. The aligned function yields a non-contiguous col-major view by design; `.contiguous()` copies it into default row-major contiguous storage, discarding the alignment and making DeepGEMM pay repeated layout-fixup cost during execution.

# Solution

## Fix in words

The diff removes the trailing `.contiguous()` call from four assignments in `process_weights_after_loading` (both `w13_weight_scale_inv` and `w2_weight_scale_inv`, in two separate method bodies). Before: `get_col_major_tma_aligned_tensor(x).contiguous()`. After: `get_col_major_tma_aligned_tensor(x)`. This preserves the column-major TMA-aligned layout DeepGEMM expects instead of converting back to row-major, eliminating per-run layout overhead and yielding ~5.5% throughput improvement with identical accuracy.

## Author's solution

Author identified that `.contiguous()` was unexpectedly converting the col-major TMA-aligned tensor from `get_col_major_tma_aligned_tensor` back to row-major, causing runtime perf overhead. The PR fixes this by dropping the `.contiguous()` call, validated by throughput gain (41.92→44.26 req/s) and unchanged gsm8k accuracy.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/quantization/fp8.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/quantization/fp8.py b/vllm/model_executor/layers/quantization/fp8.py
index 3d94626e5d8c..49ff87df93c3 100644
--- a/vllm/model_executor/layers/quantization/fp8.py
+++ b/vllm/model_executor/layers/quantization/fp8.py
@@ -772,10 +772,10 @@ def process_weights_after_loading(self, layer: Module) -> None:
             if self.allow_deep_gemm and not is_deep_gemm_e8m0_used():
                 if _is_col_major(layer.w13_weight_scale_inv):
                     layer.w13_weight_scale_inv = \
-                        get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv).contiguous()
+                        get_col_major_tma_aligned_tensor(layer.w13_weight_scale_inv)
                 if _is_col_major(layer.w2_weight_scale_inv):
                     layer.w2_weight_scale_inv = \
-                        get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv).contiguous()
+                        get_col_major_tma_aligned_tensor(layer.w2_weight_scale_inv)
 
         # If checkpoint is fp16, quantize in place.
         elif not self.quant_config.is_checkpoint_fp8_serialized:
@@ -923,10 +923,10 @@ def process_weights_after_loading(self, layer: Module) -> None:
             # Ensure column-major TMA alignment expected by DeepGEMM.
             if _is_col_major(layer.w13_weight_scale_inv):
                 layer.w13_weight_scale_inv = get_col_major_tma_aligned_tensor(
-                    layer.w13_weight_scale_inv).contiguous()
+                    layer.w13_weight_scale_inv)
             if _is_col_major(layer.w2_weight_scale_inv):
                 layer.w2_weight_scale_inv = get_col_major_tma_aligned_tensor(
-                    layer.w2_weight_scale_inv).contiguous()
+                    layer.w2_weight_scale_inv)
 
     def select_gemm_impl(
         self,

```
