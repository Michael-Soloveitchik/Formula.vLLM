# PR #28164: [Feature] Enable TP + EP `shared_experts` overlap with router, 3.7% E2E performance improvement

- field: parallelism  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2025-11-05T21:57:47Z  merged: 2025-11-06T01:21:09Z
- url: https://github.com/vllm-project/vllm/pull/28164
- labels: ready

# Issue

## Problem

In TP + EP MoE layers, the shared-expert computation was serialized with the routed-expert (router) path instead of overlapping with it. The `SharedFusedMoE` overlap optimization was explicitly disabled whenever expert parallelism (EP) was enabled, via the `not (self.use_ep or ...)` guard. So on EP configs, shared experts ran as a separate blocking step rather than hiding behind the router's compute/communication.

## Example

DeepSeek-R1 served with `-tp 8 --enable-expert-parallel` on B200, small batch (random-input-len 4, output 1024, max-concurrency 32): without overlap mean TTFT 135ms; with overlap enabled the PR reports 24% faster time-to-first-token and 3.7% E2E gain.

## Use case

Low-latency serving of large MoE models (DeepSeek-R1) with tensor parallel + expert parallel on multi-GPU nodes, especially at small batch sizes where kernel serialization dominates.

## Resources

GPU compute and inter-GPU collective bandwidth (NVLink/all-reduce). At small batch, exposed shared-expert kernel launches and TP all-reduce latency are the bottleneck.

## Type of users

Users serving MoE models with combined TP + EP (DeepSeek-class) on multi-GPU hardware (B200), targeting low latency / small batch.

## Kind of inputs

Small batch size, short input, decode-heavy workloads where per-step latency (TTFT/TPOT) matters more than throughput.

## Mechanism with the issue

`SharedFusedMoE.__init__` set `use_overlapped` to False when `self.use_ep` was True, blocking the overlapped forward path. The overlapped `forward` branch also only reduced shared-expert outputs in the non-overlapped code path, so EP runs could not both overlap and still correctly TP-all-reduce the shared output.

# Solution

## Fix in words

Diff removes the `self.use_ep` term from the `use_overlapped` disable condition in `SharedFusedMoE.__init__`, so TP + EP now takes the overlapped path where shared experts run concurrently with the router. In the overlapped `forward` branch, adds a new post-computation guard that all-reduces `shared_out` when `reduce_results`, TP world size > 1, and `must_reduce_shared_expert_outputs()` hold — ensuring correctness that the non-overlapped path already had. Switches `self.tp_size` to `get_tensor_model_parallel_world_size()` for the reduce check. Also fixes a docstring typo (`all all_reduce` → `all_reduce`). Effect: shared-expert compute hides behind router work, cutting TTFT 24% and E2E 3.7% at small batch.

## Author's solution

Follow-up to PR #26440: enable TP + EP to overlap shared experts with the router, yielding gains at small batch size — 3.7% E2E improvement and 24% faster TTFT. Validated accuracy on DeepSeek-R1 (gsm8k 0.9515) and perf on B200. Noted follow-up TODOs: reuse of stream for optimization, and sweeping batch sizes since larger batches on local node may not benefit as much.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/layer.py b/vllm/model_executor/layers/fused_moe/layer.py
index 0a8c2f311f5c..1236116386c9 100644
--- a/vllm/model_executor/layers/fused_moe/layer.py
+++ b/vllm/model_executor/layers/fused_moe/layer.py
@@ -1178,7 +1178,7 @@ class FusedMoE(CustomOp):
         hidden_size: Input hidden state size of the transformer
         intermediate_size: Intermediate size of the experts
         params_dtype: Data type for the parameters.
-        reduce_results: Whether to all all_reduce on the output of the layer
+        reduce_results: Whether to all_reduce on the output of the layer
         renormalize: Whether to renormalize the logits in the fused_moe kernel
         quant_config: Quantization configure.
         enable_eplb: Whether to enable expert parallelism load balancer.
diff --git a/vllm/model_executor/layers/fused_moe/shared_fused_moe.py b/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
index 2db733b765ce..6b4a0b8cf073 100644
--- a/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -3,7 +3,10 @@
 
 import torch
 
-from vllm.distributed import tensor_model_parallel_all_reduce
+from vllm.distributed import (
+    get_tensor_model_parallel_world_size,
+    tensor_model_parallel_all_reduce,
+)
 from vllm.model_executor.layers.fused_moe.layer import FusedMoE
 
 
@@ -25,16 +28,13 @@ def __init__(
         super().__init__(**kwargs)
         self._shared_experts = shared_experts
 
-        # Disable shared expert overlap if EP is disabled or we are not using
+        # Disable shared expert overlap if we are not using
         # flashinfer + DP since there is nothing to be gained in this case.
         # Disabling the overlap optimization also prevents the shared experts
         # from being hidden from torch.compile.
         self.use_overlapped = (
             use_overlapped
-            and not (
-                self.use_ep
-                or (self.use_flashinfer_cutlass_kernels and self.dp_size > 1)
-            )
+            and not (self.use_flashinfer_cutlass_kernels and self.dp_size > 1)
             and self._shared_experts is not None
         )
 
@@ -65,7 +65,7 @@ def forward(
                 # should have been created with reduce_results=False.
                 if (
                     self.reduce_results
-                    and self.tp_size > 1
+                    and get_tensor_model_parallel_world_size() > 1
                     and self.must_reduce_shared_expert_outputs()
                 ):
                     shared_out = tensor_model_parallel_all_reduce(shared_out)
@@ -81,4 +81,12 @@ def forward(
                 hidden_states=hidden_states,
                 router_logits=router_logits,
             )
+            # ensure early TP reduction of shared expert outputs when required
+            if (
+                shared_out is not None
+                and self.reduce_results
+                and get_tensor_model_parallel_world_size() > 1
+                and self.must_reduce_shared_expert_outputs()
+            ):
+                shared_out = tensor_model_parallel_all_reduce(shared_out)
         return shared_out, fused_out

```
