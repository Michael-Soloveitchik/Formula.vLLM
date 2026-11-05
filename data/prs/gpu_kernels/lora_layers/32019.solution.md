# PR #32019: [LoRA][Perf] Improve FusedMoE LoRA performance for small rank

- field: gpu_kernels  | area: lora_layers
- author: xyang16
- state: MERGED  created: 2026-01-09T09:52:17Z  merged: 2026-01-10T19:04:19Z
- url: https://github.com/vllm-project/vllm/pull/32019
- labels: ready

# Issue

## Problem

The FusedMoE LoRA Triton kernels used fixed tiling (default BLOCK_SIZE_N=64 for shrink, BLOCK_SIZE_K=32 for expand) regardless of LoRA rank. For shrink the LoRA output dim N equals the rank, and offs_bn is computed with a modulo, so when rank < BLOCK_SIZE_N the column offsets wrap and the same LoRA weight columns get loaded repeatedly. This forces redundant HBM loads and wasted tl.dot FLOPs on duplicated data. Same waste on expand, where contraction dim K equals rank but BLOCK_SIZE_K stays at 32.

## Example

For openai/gpt-oss-20b with max-lora-rank 8, shrink tiled N over a 64-wide block while rank is only 8, loading the 8 rank columns ~8x. Profiling showed prefill shrink _fused_moe_lora_kernel ~63µs and expand ~49µs before the fix.

## Use case

Serving an MoE model with a LoRA adapter applied to fused expert matmuls during prefill and decode.

## Resources

GPU compute (wasted tl.dot MACs on duplicated tiles) and HBM memory bandwidth (redundant LoRA weight loads).

## Type of users

Users serving MoE models with LoRA adapters, especially small LoRA ranks (e.g. rank 8) where rank << default block size.

## Kind of inputs

Small LoRA rank (N or K smaller than default block size), fused-MoE expert weights, prefill and decode batches.

## Mechanism with the issue

In the fused_moe_lora shrink kernel, offs_bn = ... % N (fused_moe_lora_op.py:136); when N < BLOCK_SIZE_N the index wraps so the weight-load region (lines 151-157) re-reads the same columns, inflating data movement and the dot. Block sizes came from try_get_optimal_moe_config / the static get_lora_op_configs defaults, neither of which considered LoRA rank.

# Solution

## Fix in words

Adds rank-aware config wrapper try_get_optimal_moe_lora_config in lora/layers/utils.py that calls normal try_get_optimal_moe_config then clamps block sizes by rank: shrink ops set BLOCK_SIZE_N = min(BLOCK_SIZE_N, next_power_of_2(rank)); expand ops set BLOCK_SIZE_K = max(16, min(BLOCK_SIZE_K, next_power_of_2(rank))). FusedMoEWithLoRA._get_lora_moe_configs now calls this wrapper with rank and distinct op_type strings for shrink vs expand, instead of calling try_get_optimal_moe_config twice with identical args. Static default table in lora/ops/triton_ops/utils.py is split so shrink defaults use block_n=min(64, next_power_of_2(rank)) and expand defaults use block_k=max(16, min(32, next_power_of_2(rank))). Effect: when rank < block size the kernel no longer tiles past real rank, eliminating repeated LoRA-weight loads and redundant dot work.

## Author's solution

BLOCK_SIZE_N defaults to 64; for shrink (rank=N) a rank smaller than BLOCK_SIZE_N makes offs_bn repeat via the modulo, loading LoRA weights repeatedly and adding tl.dot work, so set BLOCK_SIZE_N to N when N is smaller. Similarly for expand (rank=K), set BLOCK_SIZE_K to K to save data loading and computation. Yields ~9% e2e output-token-throughput gain on openai/gpt-oss-20b.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/layers/fused_moe.py`
- `vllm/lora/layers/utils.py`
- `vllm/lora/ops/triton_ops/utils.py`

### Diff
```diff
diff --git a/vllm/lora/layers/fused_moe.py b/vllm/lora/layers/fused_moe.py
index f0bcca9158ff..9e78b6164909 100644
--- a/vllm/lora/layers/fused_moe.py
+++ b/vllm/lora/layers/fused_moe.py
@@ -24,7 +24,6 @@
 )
 from vllm.model_executor.layers.fused_moe.fused_moe import (
     TritonExperts,
-    try_get_optimal_moe_config,
 )
 from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
     FusedMoEModularMethod,
@@ -39,7 +38,7 @@
     MoEPrepareAndFinalizeNoEP,
 )
 
-from .utils import _get_lora_device
+from .utils import _get_lora_device, try_get_optimal_moe_lora_config
 
 
 class FusedMoEWithLoRA(BaseLayerWithLoRA):
@@ -103,15 +102,21 @@ def _get_lora_moe_configs(
             )
         else:  # fall back to the default config
             get_config_func = functools.partial(
-                try_get_optimal_moe_config,
-                layer.w13_weight.size(),
-                layer.w2_weight.size(),
-                top_k,
-                config_dtype,
+                try_get_optimal_moe_lora_config,
+                w1_shape=layer.w13_weight.size(),
+                w2_shape=layer.w2_weight.size(),
+                rank=rank,
+                top_k=top_k,
+                dtype=config_dtype,
+                M=M,
                 block_shape=layer.quant_method.moe_quant_config.block_shape,
             )
-            shrink_config = get_config_func(M)
-            expand_config = get_config_func(M)
+            shrink_config = get_config_func(
+                op_type=f"fused_moe_lora_{op_prefix}_shrink"
+            )
+            expand_config = get_config_func(
+                op_type=f"fused_moe_lora_{op_prefix}_expand"
+            )
         shrink_config = self._normalize_keys(shrink_config)
         expand_config = self._normalize_keys(expand_config)
         return shrink_config, expand_config
diff --git a/vllm/lora/layers/utils.py b/vllm/lora/layers/utils.py
index 7dd0df2e3942..26d1a53f013a 100644
--- a/vllm/lora/layers/utils.py
+++ b/vllm/lora/layers/utils.py
@@ -7,6 +7,9 @@
 import torch
 import torch.nn as nn
 
+from vllm.model_executor.layers.fused_moe.fused_moe import try_get_optimal_moe_config
+from vllm.utils.math_utils import next_power_of_2
+
 
 class LoRAMappingType(Enum):
     LANGUAGE = 1
@@ -80,3 +83,33 @@ def dec(*args, **kwargs):
         )
 
     return dec
+
+
+def try_get_optimal_moe_lora_config(
+    op_type: str,
+    w1_shape: tuple[int, ...],
+    w2_shape: tuple[int, ...],
+    rank: int,
+    top_k: int,
+    dtype: str | None,
+    M: int,
+    block_shape: list[int] | None = None,
+) -> dict[str, int | None]:
+    config = try_get_optimal_moe_config(
+        w1_shape, w2_shape, top_k, dtype, M, block_shape
+    ).copy()
+    if op_type in [
+        "fused_moe_lora_w13_shrink",
+        "fused_moe_lora_w2_shrink",
+    ]:
+        config["BLOCK_SIZE_N"] = min(
+            config.get("BLOCK_SIZE_N", 64), next_power_of_2(rank)
+        )
+    elif op_type in [
+        "fused_moe_lora_w13_expand",
+        "fused_moe_lora_w2_expand",
+    ]:
+        config["BLOCK_SIZE_K"] = max(
+            16, min(config.get("BLOCK_SIZE_K", 32), next_power_of_2(rank))
+        )
+    return config
diff --git a/vllm/lora/ops/triton_ops/utils.py b/vllm/lora/ops/triton_ops/utils.py
index ed9e916455e5..66703a36a6b1 100644
--- a/vllm/lora/ops/triton_ops/utils.py
+++ b/vllm/lora/ops/triton_ops/utils.py
@@ -13,6 +13,7 @@
 from vllm.logger import init_logger
 from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
 from vllm.platforms import current_platform
+from vllm.utils.math_utils import next_power_of_2
 
 logger = init_logger(__name__)
 is_batch_invariant = vllm_is_batch_invariant()
@@ -223,14 +224,25 @@ def get_lora_op_configs(
     # The default config for fused_moe_lora ops
     elif op_type in [
         "fused_moe_lora_w13_shrink",
-        "fused_moe_lora_w13_expand",
         "fused_moe_lora_w2_shrink",
+    ]:
+        default = {
+            "block_m": 64,
+            "block_n": min(64, next_power_of_2(rank)),
+            "block_k": 32,
+            "num_warps": 4,
+            "num_stages": 3,
+            "group_size_m": 8,
+            "split_k": 1,
+        }
+    elif op_type in [
+        "fused_moe_lora_w13_expand",
         "fused_moe_lora_w2_expand",
     ]:
         default = {
             "block_m": 64,
             "block_n": 64,
-            "block_k": 32,
+            "block_k": max(16, min(32, next_power_of_2(rank))),
             "num_warps": 4,
             "num_stages": 3,
             "group_size_m": 8,

```
