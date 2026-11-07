# PR #30243: [LoRA]  Reduce the loading time of MoE LoRA

- field: parallelism  | area: lora_models.py
- author: jeejeelee
- state: MERGED  created: 2025-12-08T07:41:22Z  merged: 2025-12-08T13:29:47Z
- url: https://github.com/vllm-project/vllm/pull/30243
- labels: ready

# Issue

## Problem

When loading LoRA adapters onto CPU, `from_lora_tensors` called `pin_memory()` on every `lora_a` and `lora_b` tensor individually, immediately after each tensor was moved to the device. For MoE models with a very large number of LoRA weights, this per-tensor pinning incurred significant cumulative overhead and dominated disk loading time. Additionally, pinning before weight merging/packing was wasteful because the packing step could invalidate the pinned allocation.

## Example

An MoE LoRA adapter with many expert weight tensors: each tiny tensor triggers a separate pinned-memory allocation during load, and subsequent `pack_moe` merging discards those pinned buffers, so the pinning work is redundant. The PR reports ~2x reduction in disk loading time after the change.

## Use case

Loading LoRA adapters from disk for serving, specifically when staging weights in pinned CPU memory for fast async H2D transfer to GPU.

## Resources

Disk/load time and CPU — pinned-memory allocation (cudaHostAlloc) is expensive and was paid per-tensor, plus wasted when packing freed the buffers.

## Type of users

Users serving MoE models with LoRA adapters, where the adapter contains many per-expert LoRA weight tensors.

## Kind of inputs

LoRA weight dicts with a large count of tensors (MoE experts), loaded onto CPU with pin_memory available.

## Mechanism with the issue

`from_lora_tensors` computed `pin_memory` once then called `.pin_memory()` inline on each `lora_a`/`lora_b` as tensors were assigned. For MoE, the sheer number of tensors made this per-call allocation slow, and because pinning happened before `_create_merged_loras_inplace` packed/merged weights (e.g. `pack_moe`), the packing invalidated the pinned allocations — doing the work twice.

# Solution

## Fix in words

The diff removes all `pin_memory()` calls from `from_lora_tensors` — tensors are now just moved to device/dtype without pinning during load. Pinning is relocated to `_create_merged_loras_inplace`, executed AFTER all LoRA weights are optimized and merged/packed. It detects device from the first LoRA's `lora_a` (handling both list form for MoE and single-tensor form), and if the device is CPU with pin_memory available, iterates once over the merged weights pinning each `lora_a`/`lora_b` (indexing into lists for MoE). Running pinning once on the final merged tensors avoids redundant per-tensor allocations and avoids pinning buffers that packing would later invalidate, cutting load time ~2x.

## Author's solution

Following up on PR #29322, merge all LoRA weights before running the `pin_memory` logic. Doing pin_memory after merging greatly reduces the overhead caused by pinning, reducing LoRA disk loading time by about 2x.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/models.py`

### Diff
```diff
diff --git a/vllm/lora/models.py b/vllm/lora/models.py
index ada30da60044..567ffce4e75f 100644
--- a/vllm/lora/models.py
+++ b/vllm/lora/models.py
@@ -115,7 +115,7 @@ def from_lora_tensors(
         weights_mapper: WeightsMapper | None = None,
     ) -> "LoRAModel":
         """Create a LoRAModel from a dictionary of tensors."""
-        pin_memory = str(device) == "cpu" and is_pin_memory_available()
+
         loras: dict[str, LoRALayerWeights] = {}
         for tensor_name, tensor in tensors.items():
             if is_base_embeddding_weights(tensor_name):
@@ -139,14 +139,8 @@ def from_lora_tensors(
                         f" with the base model's vocabulary size({model_vocab_size})."
                     )
                 loras[module_name].lora_a = tensor.to(device=device, dtype=dtype)
-                if pin_memory:
-                    loras[module_name].lora_a = loras[module_name].lora_a.pin_memory()
             else:
                 loras[module_name].lora_b = tensor.to(device=device, dtype=dtype)
-
-                if pin_memory:
-                    loras[module_name].lora_b = loras[module_name].lora_b.pin_memory()
-
         return cls(lora_model_id, peft_helper.r, loras)
 
     @classmethod
@@ -742,6 +736,32 @@ def _create_merged_loras_inplace(self, lora_model: LoRAModel) -> None:
         for lora in lora_model.loras.values():
             lora.optimize()
 
+        first_lora: LoRALayerWeights = next(iter(lora_model.loras.values()))
+        assert first_lora.lora_a is not None
+        if isinstance(first_lora.lora_a, list):
+            lora_device = next(iter(first_lora.lora_a))
+        else:
+            lora_device = first_lora.lora_a.device
+        # Execute pin_memory after LoRA weight merging, mainly because:
+        # 1. Some MoE models have a large number of LoRA weights. If we
+        # perform # pin_memory immediately after loading weights, the
+        # overhead is significant.
+        # 2. The weight packing above (e.g., pack_moe) may invalidate the
+        # pin_memory allocation, so we execute it after packing.
+
+        pin_memory = str(lora_device) == "cpu" and is_pin_memory_available()
+        if pin_memory:
+            for lora in lora_model.loras.values():
+                if isinstance(lora.lora_a, list):
+                    for index in range(len(lora.lora_a)):
+                        if lora.lora_a[index] is None:
+                            continue
+                        lora.lora_a[index] = lora.lora_a[index].pin_memory()
+                        lora.lora_b[index] = lora.lora_b[index].pin_memory()
+                else:
+                    lora.lora_a = lora.lora_a.pin_memory()
+                    lora.lora_b = lora.lora_b.pin_memory()
+
     def _get_lora_layer_weights(
         self, lora_model: LoRAModel, module_name: str
     ) -> LoRALayerWeights | None:

```
