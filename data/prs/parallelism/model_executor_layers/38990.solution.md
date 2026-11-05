# PR #38990: [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap

- field: parallelism  | area: model_executor_layers
- author: voipmonitor
- state: MERGED  created: 2026-04-04T17:59:02Z  merged: 2026-04-05T14:28:32Z
- url: https://github.com/vllm-project/vllm/pull/38990
- labels: bug, ready, ready-run-all-tests

# Issue

## Problem

MoE shared-expert scheduling picks the EXTERNAL order unconditionally on TP-only configs, forcing shared experts to run sequentially before routed experts on the main CUDA stream. The aux-stream MULTI_STREAM_OVERLAPPED path, which runs shared experts in parallel on a secondary stream, is never reached even when an aux stream exists. Result: 6-8% decode throughput regression introduced by #35153.

## Example

GLM-5 MoE on 8×RTX PRO 6000 Blackwell, TP=8, C=1, no MTP: 36.8 tok/s at ctx=0 before vs 41.8 tok/s after (+14%); 34.8→37.8 at 16k, 33.8→36.8 at 32k.

## Use case

Serving large MoE models (DeepSeek-V3, GLM-5) on multi-GPU tensor-parallel deployments without expert parallel (EP/EPLB).

## Resources

GPU compute utilization — serial execution leaves the aux stream idle and fails to overlap shared-expert compute with routed-expert compute.

## Type of users

Users serving MoE models on multi-GPU TP-only setups (no EP, no flashinfer two-sided kernels).

## Kind of inputs

Decode-phase workloads across context lengths (0–32k) on MoE models with shared experts under TP-only parallel config.

## Mechanism with the issue



# Solution

## Fix in words

Diff renames `_has_external_experts` to `_use_external_experts` and inverts its logic. Before: returned `True` unless EPLB-non-default or flashinfer-two-sided active, so TP-only always got `True` → EXTERNAL. After: returns `True` only when EPLB-non-default OR flashinfer-two-sided, and `False` early when `_use_dp_chunking`. In `_determine_shared_experts_order`, guard simplifies from `_has_external_experts and not _use_dp_chunking` to `_use_external_experts`. Effect: TP-only configs now fall through to MULTI_STREAM_OVERLAPPED, restoring aux-stream shared-expert overlap and 0.18.x-level throughput.

## Author's solution

Author reorders the checks so MULTI_STREAM_OVERLAPPED is evaluated first when an aux stream is available, falling back to EXTERNAL only when no aux stream exists. Root cause stated: the `_has_external_experts` early return in `_determine_shared_experts_order` fires before the aux-stream check, returning `True` for TP-only (neither EPLB nor flashinfer active) and unconditionally selecting EXTERNAL.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/runner/shared_experts.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/runner/shared_experts.py b/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
index 6d2189cb49b3..bb8645c8f619 100644
--- a/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
+++ b/vllm/model_executor/layers/fused_moe/runner/shared_experts.py
@@ -93,24 +93,24 @@ def __init__(
                 )
 
     @property
-    def _has_external_experts(self) -> bool:
+    def _use_external_experts(self) -> bool:
+        if self._use_dp_chunking:
+            return False
+
         # Disable shared expert overlap if:
         #   - we are using eplb with non-default backend, because of correctness issues
         #   - we are using flashinfer with DP, since there nothing to gain
         backend = self._moe_config.moe_parallel_config.all2all_backend
-        return not (
-            (
-                self._moe_config.moe_parallel_config.enable_eplb
-                and backend != "allgather_reducescatter"
-            )
-            or self._moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
-        )
+        return (
+            self._moe_config.moe_parallel_config.enable_eplb
+            and backend != "allgather_reducescatter"
+        ) or self._moe_config.moe_parallel_config.use_fi_nvl_two_sided_kernels
 
     def _determine_shared_experts_order(
         self,
         hidden_states: torch.Tensor,
     ) -> SharedExpertsOrder:
-        if self._has_external_experts and not self._use_dp_chunking:
+        if self._use_external_experts:
             return SharedExpertsOrder.EXTERNAL
 
         if self._quant_method.mk_owns_shared_expert:

```
