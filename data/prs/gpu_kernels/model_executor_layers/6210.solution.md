# PR #6210: [Kernel] reloading fused_moe config on the last chunk

- field: gpu_kernels  | area: model_executor_layers
- author: avshalomman
- state: MERGED  created: 2024-07-08T09:38:23Z  merged: 2024-07-08T15:00:39Z
- url: https://github.com/vllm-project/vllm/pull/6210
- labels: 

# Issue

## Problem

In the fused MoE chunking path, the Triton kernel configuration (block sizes, num_warps, num_stages) is selected once based on M = min(num_tokens, CHUNK_SIZE) and then reused for every chunk. When the total token count is not a multiple of the chunk size, the final chunk is smaller than CHUNK_SIZE, so it runs with a config tuned for a larger token count. This mismatched config is suboptimal for the small last chunk and leaves performance on the table.

## Example

If CHUNK_SIZE is 32K and a batch produces 40K tokens, the kernel runs one 32K chunk plus a 8K last chunk, but both use the config picked for M=32K. The 8K chunk would run faster with a config tuned for 8K tokens.

## Use case

MoE model serving where expert FFN batches exceed VLLM_FUSED_MOE_CHUNK_SIZE and get split into chunks during prefill/decode of batched requests.

## Resources

GPU compute efficiency / Triton kernel occupancy — wrong block/warp config underutilizes the SMs for the small tail chunk.

## Type of users

Users serving Mixture-of-Experts models (including FP8-quantized variants) with token batches large enough to trigger fused_moe chunking.

## Kind of inputs

Batches where total token count across experts is not an exact multiple of the chunk size, producing a smaller trailing chunk (any dtype, fp8 or not).

## Mechanism with the issue

In fused_experts (vllm/model_executor/layers/fused_moe/fused_moe.py), config was computed once from the first chunk's M before the chunk loop. The per-chunk loop truncated intermediate caches for tokens_in_chunk but kept the original config, so the last (smaller) chunk invoked the Triton kernel with a config sized for a full chunk.

# Solution

## Fix in words

Diff extracts the config-selection logic into a new helper try_get_optimal_moe_config, which given M either uses override_config, looks up the nearest tuned config from the loaded config map, or falls back to get_default_config. fused_experts now binds the fixed args (weight shapes, top_k, dtype, override) via functools.partial into get_config_func and calls get_config_func(M) for the initial config. Inside the chunk loop, when the current chunk is the smaller last chunk (tokens_in_chunk < chunk size), it calls get_config_func(tokens_in_chunk) to reload a config matched to the actual last-chunk size. Effect: the final chunk runs with a kernel config tuned to its real token count, yielding ~5% average speedup.

## Author's solution

Follow-up to the fused_moe chunking mechanism from PR #6029. Author observed that when the last chunk is smaller than the chunk size, reloading the kernel configuration according to the last chunk size helps, giving ~5% average speedup in their experiments.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe.py b/vllm/model_executor/layers/fused_moe/fused_moe.py
index 99a5c7d78a67..a29622b7d25c 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -332,6 +332,31 @@ def get_default_config(
     return config
 
 
+def try_get_optimal_moe_config(
+    w1_shape: Tuple[int, ...],
+    w2_shape: Tuple[int, ...],
+    top_k: int,
+    dtype: Optional[str],
+    M: int,
+    override_config: Optional[Dict[str, Any]] = None,
+):
+    if override_config:
+        config = override_config
+    else:
+        # First try to load optimal config from the file
+        E, _, N = w2_shape
+        configs = get_moe_configs(E, N, dtype)
+
+        if configs:
+            # If an optimal configuration map has been found, look up the
+            # optimal config
+            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
+        else:
+            # Else use the default config
+            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype)
+    return config
+
+
 def fused_topk(
     hidden_states: torch.Tensor,
     gating_output: torch.Tensor,
@@ -428,22 +453,16 @@ def fused_experts(hidden_states: torch.Tensor,
     CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
     M = min(num_tokens, CHUNK_SIZE)
 
-    if override_config:
-        config = override_config
-    else:
-        # First try to load optimal config from the file
-        configs = get_moe_configs(E, w2.shape[2],
-                                  "float8" if use_fp8 else None)
+    get_config_func = functools.partial(
+        try_get_optimal_moe_config,
+        w1.shape,
+        w2.shape,
+        topk_ids.shape[1],
+        "float8" if use_fp8 else None,
+        override_config=override_config,
+    )
 
-        if configs:
-            # If an optimal configuration map has been found, look up the
-            # optimal config
-            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
-        else:
-            # Else use the default config
-            config = get_default_config(M, E, N, w1.shape[2],
-                                        topk_ids.shape[1],
-                                        "float8" if use_fp8 else None)
+    config = get_config_func(M)
 
     intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                       device=hidden_states.device,
@@ -478,6 +497,8 @@ def fused_experts(hidden_states: torch.Tensor,
             intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
             intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
             intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
+            # reload config to get better performance on the last chunk
+            config = get_config_func(tokens_in_chunk)
 
         curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
         curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

```
