# PR #37695: [Perf] Use torch compile to fuse pack topk in trtllm moe

- field: parallelism  | area: model_executor_layers
- author: wzhao18
- state: MERGED  created: 2026-03-20T15:21:34Z  merged: 2026-03-27T23:30:46Z
- url: https://github.com/vllm-project/vllm/pull/37695
- labels: performance, ready, nvidia

# Issue

## Problem

In the trtllm FP8 and NVFP4 MoE experts, packing topk_ids and topk_weights into the single int32 tensor the kernel expects was done with an inline, eager-mode elementwise expression: `(topk_ids << 16) | topk_weights.to(bfloat16).view(int16)`. Running eager launches several small CUDA kernels (cast, shift, view, bitwise-or) per MoE forward, each with its own launch overhead. For a routing-step op this small, kernel-launch and dispatch overhead dominate the actual compute.

## Example

Minimax M2.5 at TP=2, concurrency 64, 1K input / 1K output: the unfused eager pack op costs enough per-step overhead that fusing it yields ~2% end-to-end speedup across both FP8 and NVFP4 checkpoints.

## Use case

High-throughput online serving of a large MoE model where the pack-topk op runs every decode step on the critical path, so small per-step overheads accumulate across many tokens and concurrent requests.

## Resources

GPU kernel-launch / dispatch overhead and SM occupancy — the op is tiny so compute is trivial but multiple separate kernel launches waste GPU scheduling time.

## Type of users

Users serving MoE models with FP8 or NVFP4 quantization on multi-GPU tensor-parallel setups using the trtllm/flashinfer fused-MoE path.

## Kind of inputs

Decode-step routing tensors (topk_ids int, topk_weights float) sized per batch × top-k; benefit scales with high concurrency and sustained token generation (1K/1K, concurrency 64).

## Mechanism with the issue



# Solution

## Fix in words

The diff extracts the inline pack expression into one shared helper `trtllm_moe_pack_topk_ids_weights` in `fused_moe/utils.py`, decorated with `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)`. Both `trtllm_fp8_moe.py` and `trtllm_nvfp4_moe.py` drop their duplicated eager `(topk_ids << 16) | topk_weights.to(bfloat16).view(int16)` and call the compiled helper instead. torch.compile fuses the cast, shift, view, and bitwise-or into a single kernel, cutting per-step launch overhead. Net effect: ~2% throughput gain with identical packing semantics and no accuracy change (gsm8k matches).

## Author's solution

Author fused the pack topk_ids + topk_weights op in trtllm MoE using torch.compile. Benchmarking Minimax M2.5 at TP=2, concurrency 64, 1K/1K showed ~2% speedup across both FP8 and NVFP4, with gsm8k accuracy preserved.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py`
- `vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py`
- `vllm/model_executor/layers/fused_moe/utils.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py b/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py
index 671435a88c06..9a6f67b421f9 100644
--- a/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py
+++ b/vllm/model_executor/layers/fused_moe/experts/trtllm_fp8_moe.py
@@ -15,6 +15,7 @@
 from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
     TopKWeightAndReduceNoOP,
 )
+from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
 from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
     activation_to_flashinfer_int,
 )
@@ -152,11 +153,8 @@ def apply(
         import flashinfer
         from flashinfer.fused_moe import Fp8QuantizationType
 
-        # Pack topk_ids and topk_weights into single tensor
-        # Format: (expert_id << 16) | (weight_bf16.view(int16))
-        packed_topk_ids = (topk_ids << 16) | topk_weights.to(torch.bfloat16).view(
-            torch.int16
-        )
+        # Pack topk ids and weights into format expected by the kernel.
+        packed_topk_ids = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
 
         # trtllm_fp8_block_scale_routed_moe does not support autotuning
         # so skip this kernel during dummy run for autotuning.
diff --git a/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py b/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py
index 7960bdf44792..84beb6abb553 100644
--- a/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py
+++ b/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py
@@ -15,6 +15,7 @@
 from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
     TopKWeightAndReduceNoOP,
 )
+from vllm.model_executor.layers.fused_moe.utils import trtllm_moe_pack_topk_ids_weights
 from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
     activation_to_flashinfer_int,
 )
@@ -183,9 +184,7 @@ def apply(
         assert self.quant_config.w2_scale is not None
 
         # Pack topk ids and weights into format expected by the kernel.
-        packed_tensor = (topk_ids.to(torch.int32) << 16) | topk_weights.to(
-            torch.bfloat16
-        ).view(torch.int16)
+        packed_tensor = trtllm_moe_pack_topk_ids_weights(topk_ids, topk_weights)
 
         # trtllm_fp4_block_scale_routed_moe does not support autotuning
         # so skip this kernel during dummy run for autotuning.
diff --git a/vllm/model_executor/layers/fused_moe/utils.py b/vllm/model_executor/layers/fused_moe/utils.py
index ba4494f6cdc3..c576b0a25c28 100644
--- a/vllm/model_executor/layers/fused_moe/utils.py
+++ b/vllm/model_executor/layers/fused_moe/utils.py
@@ -323,3 +323,16 @@ def normalize_batched_scales_shape(
 @functools.cache
 def disable_inplace() -> bool:
     return is_torch_equal_or_newer("2.9")
+
+
+@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
+def trtllm_moe_pack_topk_ids_weights(
+    topk_ids: torch.Tensor, topk_weights: torch.Tensor
+) -> torch.Tensor:
+    """
+    Pack topk_ids and topk_weights into a single int32 tensor.
+    Format: (expert_id << 16) | weight_bf16.view(int16)
+    """
+    return (topk_ids.to(torch.int32) << 16) | topk_weights.to(torch.bfloat16).view(
+        torch.int16
+    )

```
