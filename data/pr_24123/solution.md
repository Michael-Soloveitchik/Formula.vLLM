# PR #24123: [Compilation][WideEP] Enable Piecewise CUDAGraph for DeepEPHT

- field: gpu_kernels  | area: vllm_config
- author: yewentao256
- state: MERGED  created: 2025-09-02T20:36:10Z  merged: 2025-09-09T14:21:11Z
- url: https://github.com/vllm-project/vllm/pull/24123
- labels: performance, ready, deepseek

# Issue

## Problem

DeepEP high-throughput (HT) all-to-all MoE kernels were treated as entirely CUDA-graph-incompatible. Whenever the HT backend ran with data_parallel_size > 1, vLLM forced cudagraph_mode to NONE and set enforce_eager=True, disabling CUDA graphs for the whole model. This left the entire decode path running eagerly, paying per-op kernel-launch and CPU dispatch overhead even for the compute portions that are perfectly graph-safe.

## Example

Serving DeepSeek-R1 with --data-parallel-size 8 --enable-expert-parallel and the HT backend: decode ran fully eager, giving ~40.67 req/s and ~121.5 ms TPOT. Enabling piecewise graphs raised it to ~54.10 req/s and ~89.5 ms TPOT (~33% E2E decode improvement).

## Use case

Large-scale MoE serving with expert parallelism and wide EP across many DP ranks, where DeepEP HT kernels give the best dispatch/combine throughput for decode.

## Resources

Wasted GPU kernel-launch and CPU dispatch overhead per decode step; the all-to-all MoE dispatch/combine also stresses inter-GPU NVLink/network bandwidth.

## Type of users

Users serving large MoE models (e.g. DeepSeek-R1) multi-GPU with expert parallelism / wide EP and the deepep_high_throughput all-to-all backend.

## Kind of inputs

Decode-heavy workloads: small per-step batches, many concurrent requests (e.g. 1024 prompts, output-len 200), running under DP>1 with EP enabled.

## Mechanism with the issue

In vllm/platforms/cuda.py check_and_update_config, the HT+DP>1 branch hard-disabled graphs (cudagraph_mode=NONE, enforce_eager=True) because the MoE communication ops cannot be captured. But non-MoE compute (attention etc.) can be graphed; forcing full eager threw away that benefit. Separately, set_splitting_ops_for_v1 assigned self.splitting_ops = self._attention_ops by reference, risking mutation of the shared class-level list.

# Solution

## Fix in words

In cuda.py, the HT+DP>1 guard no longer blanket-disables graphs: instead of forcing NONE/enforce_eager, it now accepts NONE and PIECEWISE and sets cudagraph_mode=PIECEWISE, so compute gets graphed while MoE comm stays out. In compilation.py set_splitting_ops_for_v1, when VLLM_ALL2ALL_BACKEND==deepep_high_throughput it appends vllm.moe_forward and vllm.moe_forward_shared to splitting_ops, forcing the fx graph to split at those ops so dispatch/combine run outside CUDA graphs. It also changes self.splitting_ops = self._attention_ops to list(self._attention_ops), copying instead of aliasing the class-level list to avoid mutating it via reference. Net effect: piecewise CUDA graphs capture graph-safe compute while MoE all-to-all remains eager, yielding ~33% decode throughput gain with no accuracy loss.

## Author's solution

Enable Piecewise CUDA Graph for DeepEP High Throughput so compute is captured while MoE communication stays outside graphs. Validated with DeepSeek-R1 under DP=8 + expert parallelism: gsm8k accuracy unchanged (~0.95) and ~33% E2E decode improvement (TPOT 121.5→89.5 ms, throughput 40.67→54.10 req/s), with prefill roughly unchanged.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/config/compilation.py`
- `vllm/platforms/cuda.py`

### Diff
```diff
diff --git a/vllm/config/compilation.py b/vllm/config/compilation.py
index 677fb069bc07..09600e96a1c6 100644
--- a/vllm/config/compilation.py
+++ b/vllm/config/compilation.py
@@ -546,7 +546,8 @@ def set_splitting_ops_for_v1(self):
             # full cudagraph outside the fx graph. This reduces some cpu
             # overhead when the runtime batch_size is not cudagraph captured.
             # see https://github.com/vllm-project/vllm/pull/20059 for details.
-            self.splitting_ops = self._attention_ops
+            # make a copy to avoid mutating the class-level list via reference.
+            self.splitting_ops = list(self._attention_ops)
         elif len(self.splitting_ops) == 0:
             logger.warning_once("Using piecewise compilation with empty "
                                 "splitting_ops.")
@@ -561,6 +562,18 @@ def set_splitting_ops_for_v1(self):
                 self.cudagraph_mode = CUDAGraphMode.FULL
             self.splitting_ops = []
 
+        if envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput":
+            # exclude MoE dispatch/combine from capture by ensuring
+            # piecewise splitting includes them, so communication remains
+            # outside CUDA graphs while compute can still be graphed.
+            moe_ops = [
+                "vllm.moe_forward",
+                "vllm.moe_forward_shared",
+            ]
+            for op in moe_ops:
+                if op not in self.splitting_ops:
+                    self.splitting_ops.append(op)
+
     def splitting_ops_contain_attention(self) -> bool:
         return self.splitting_ops is not None and all(
             op in self.splitting_ops for op in self._attention_ops)
diff --git a/vllm/platforms/cuda.py b/vllm/platforms/cuda.py
index 1b0a298352cb..fc1a399d6f43 100644
--- a/vllm/platforms/cuda.py
+++ b/vllm/platforms/cuda.py
@@ -183,16 +183,14 @@ def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
         compilation_config = vllm_config.compilation_config
         if (envs.VLLM_ALL2ALL_BACKEND == "deepep_high_throughput"
                 and parallel_config.data_parallel_size > 1
-                and compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
+                and compilation_config.cudagraph_mode
+                not in [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE]):
             logger.info(
-                "Data Parallel: disabling cudagraphs since DP "
-                "with DeepEP high-throughput kernels are not CUDA Graph "
-                "compatible. The DeepEP low-latency kernels are CUDA Graph "
-                "compatible. Set the all_to_all backend to deepep_low_latency "
-                "to use those kernels instead.")
-            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
-            if model_config is not None:
-                model_config.enforce_eager = True
+                "Data Parallel with DeepEP high-throughput: using PIECEWISE "
+                "CUDA graphs and excluding MoE ops from capture. Set "
+                "VLLM_ALL2ALL_BACKEND=deepep_low_latency if you need MoE "
+                "graphs captured as well.")
+            compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
 
     @classmethod
     def get_current_memory_usage(cls,

```
