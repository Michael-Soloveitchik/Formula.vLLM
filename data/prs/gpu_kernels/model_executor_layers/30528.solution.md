# PR #30528: [Perf] Set split_k to 1 for triton_kernels

- field: gpu_kernels  | area: model_executor_layers
- author: xyang16
- state: MERGED  created: 2025-12-12T01:15:38Z  merged: 2025-12-12T19:07:57Z
- url: https://github.com/vllm-project/vllm/pull/30528
- labels: performance, ready, gpt-oss, nvidia

# Issue

## Problem

triton_kernels computes a split_k factor for grouped MoE matmul that exceeds 1 at small batch sizes. When split_k > 1, the triton_kernels matmul_ogs path forces torch.float32 accumulation dtype instead of bf16, degrading performance. split_k also provides no benefit for small matmuls, so the net effect is slower decode at low concurrency.

## Example

Serving openai/gpt-oss-120b (MXFP4) on H200 (sm90, 132 SMs). At batch size 16, enabling split_k=1 raised e2e output token throughput ~38%. At bs=8, output token throughput went 575 -> 799 tok/s and benchmark duration 346s -> 249s.

## Use case

Low-concurrency online serving of MoE / MXFP4 models (gpt-oss) on Hopper, where per-step decode has small m dimension.

## Resources

GPU compute efficiency. fp32 accumulation path wastes tensor-core throughput vs bf16; split_k adds extra reduction work with no gain at small shapes.

## Type of users

Users serving MXFP4-quantized MoE models via triton_kernels on Hopper (sm90) GPUs at small batch size / low request concurrency.

## Kind of inputs

Small m (batch): split_k > 1 kicks in roughly when m < 32. Big batch already yields split_k=1 (grid_size large -> split_k = max(n_sms//grid_size, 1) = 1).

## Mechanism with the issue

triton_kernels opt_flags computes split_k = max(n_sms // grid_size, 1) (opt_flags_nvidia.py). Small m -> small grid_size -> split_k > 1. matmul_ogs.py then selects float32 dtype when split_k > 1. vLLM's _swizzle_mxfp4 in mxfp4_utils.py only set constraints for sm100 (Blackwell), leaving Hopper to use the slow computed split_k.

# Solution

## Fix in words

In _swizzle_mxfp4 (mxfp4_utils.py), the single `is_device_capability(100)` branch is replaced with an `is_cuda()` guard containing a cap-specific split. For sm90 (Hopper) it now calls opt_flags.update_opt_flags_constraints({"split_k": 1}), forcing split_k=1 so the kernel keeps bf16 accumulation instead of falling back to fp32. The sm100 (Blackwell) branch is preserved unchanged (is_persistent=True, epilogue_subtile=1), which already implies split_k=1. Net: small-batch Hopper matmuls use bf16 and skip pointless split-k reduction, improving decode throughput ~38% at bs=16.

## Author's solution

Set split_k to 1 for triton_kernels on Hopper (sm90) by adding an opt_flags constraint. For large batch size split_k is already 1, so no change there; for small batch size this avoids split_k > 1, which otherwise triggers fp32 precision and is meaningless for small matmul. Blackwell (sm100) already gets split_k=1 via is_persistent=True, so only Hopper needs the explicit constraint.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py b/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
index d0c8b3d1a309..7a351afb3c41 100644
--- a/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
+++ b/vllm/model_executor/layers/quantization/utils/mxfp4_utils.py
@@ -57,12 +57,18 @@ def _swizzle_mxfp4(quant_tensor, scale, num_warps):
                 mx_axis=1, num_warps=num_warps
             )
         )
-    if current_platform.is_cuda() and current_platform.is_device_capability(100):
-        constraints = {
-            "is_persistent": True,
-            "epilogue_subtile": 1,
-        }
-        opt_flags.update_opt_flags_constraints(constraints)
+    if current_platform.is_cuda():
+        if current_platform.is_device_capability(90):
+            constraints = {
+                "split_k": 1,
+            }
+            opt_flags.update_opt_flags_constraints(constraints)
+        elif current_platform.is_device_capability(100):
+            constraints = {
+                "is_persistent": True,
+                "epilogue_subtile": 1,
+            }
+            opt_flags.update_opt_flags_constraints(constraints)
     # transpose the tensor so that the quantization axis is on dim1
     quant_tensor = quant_tensor.transpose(-2, -1)
     scale = scale.transpose(-2, -1)

```
