# PR #3985: [Core][Model] Use torch.compile to accelerate layernorm in commandr

- field: scheduling  | area: model_executor_models
- author: youkaichao
- state: MERGED  created: 2024-04-11T00:35:43Z  merged: 2024-04-11T01:48:26Z
- url: https://github.com/vllm-project/vllm/pull/3985
- labels: 

# Issue

## Problem

The commandr `LayerNorm.forward` runs a sequence of elementwise and reduction ops (cast to float32, mean, variance, rsqrt, weight scale, cast back) in eager mode. Each op launches a separate CUDA kernel, adding per-kernel-launch overhead on every layernorm call. Naively adding `@torch.compile` to the bound `forward` method forces torch to trace `self`, and because `self` changes across instances/invocations, it triggers repeated recompilation instead of a single cached compiled graph.

## Example

Command-R (CohereForAI/c4ai-command-r-v01) at tp=4, batch-size=1: eager avg latency 2.436s vs compiled-standalone 2.314s. At default batch-size=8: 2.900s without vs 2.734s with the PR.

## Use case

Serving the Cohere Command-R model under tensor parallelism (tp=4) for low-latency single-request and small-batch inference.

## Resources

GPU compute and kernel-launch overhead; fusing the elementwise/reduction chain into one compiled kernel cuts launch count and intermediate HBM traffic.

## Type of users

Users serving Command-R / CohereForAI models across multi-GPU tensor-parallel setups where many small layernorm calls dominate per-layer overhead.

## Kind of inputs

Small batch sizes (1-8) and typical decode-length hidden-state tensors in bf16/fp16 where per-op kernel launch cost is a large fraction of total time.

## Mechanism with the issue

`LayerNorm.forward` in `vllm/model_executor/models/commandr.py` computes layernorm as multiple unfused torch ops. Decorating the method directly would make torch.compile trace `self` as an input; since `self` mutates, the guard fails and torch recompiles repeatedly, killing the benefit.

# Solution

## Fix in words

Diff extracts the pure layernorm math into a module-level standalone function `layer_norm_func(hidden_states, weight, variance_epsilon)` decorated with `@torch.compile`. The function takes only tensors and the epsilon scalar—no `self`—so torch compiles once and reuses the cached graph across all calls. `LayerNorm.forward` now just calls `layer_norm_func(hidden_states, self.weight, self.variance_epsilon)` and returns `(hidden_states, residuals)`. Compiler fuses the cast/mean/variance/rsqrt/scale chain into fewer kernels, giving ~5% latency and throughput gains.

## Author's solution

Author notes that decorating `forward` directly makes pytorch trace `self`, and since `self` changes it recompiles every time. Because `self` does not affect the computation, they extract the layernorm into a standalone function that depends only on tensor inputs, so torch compiles it once. Benchmarks show ~5% latency and throughput improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/commandr.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/commandr.py b/vllm/model_executor/models/commandr.py
index aa27f0a96c74..aa9b28b676e0 100644
--- a/vllm/model_executor/models/commandr.py
+++ b/vllm/model_executor/models/commandr.py
@@ -48,6 +48,18 @@
 from vllm.sequence import SamplerOutput
 
 
+@torch.compile
+def layer_norm_func(hidden_states, weight, variance_epsilon):
+    input_dtype = hidden_states.dtype
+    hidden_states = hidden_states.to(torch.float32)
+    mean = hidden_states.mean(-1, keepdim=True)
+    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
+    hidden_states = (hidden_states - mean) * torch.rsqrt(variance +
+                                                         variance_epsilon)
+    hidden_states = weight.to(torch.float32) * hidden_states
+    return hidden_states.to(input_dtype)
+
+
 class LayerNorm(nn.Module):
 
     def __init__(self, param_shape=None, eps=1e-5):
@@ -57,14 +69,9 @@ def __init__(self, param_shape=None, eps=1e-5):
         set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})
 
     def forward(self, hidden_states, residuals=None):
-        input_dtype = hidden_states.dtype
-        hidden_states = hidden_states.to(torch.float32)
-        mean = hidden_states.mean(-1, keepdim=True)
-        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
-        hidden_states = (hidden_states -
-                         mean) * torch.rsqrt(variance + self.variance_epsilon)
-        hidden_states = self.weight.to(torch.float32) * hidden_states
-        return hidden_states.to(input_dtype), residuals
+        hidden_states = layer_norm_func(hidden_states, self.weight,
+                                        self.variance_epsilon)
+        return hidden_states, residuals
 
     def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
         tp_rank = get_tensor_model_parallel_rank()

```
