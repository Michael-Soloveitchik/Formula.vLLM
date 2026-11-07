# PR #9948: [Bugfix] Fix Phi-3 BNB quantization with tensor parallel (#9948)

# Issue

## Original PR description (captured source)


FIX #9937 (*link existing issues this PR will resolve*)
- #9937 



## Full discussion

See [9948.md](./9948.md) and [9948.json](./9948.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/9948
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/9948.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `568fdaf40ea23eeea7231b51c816ab22733fcb5f9166c7970cd9173feec5d116`

Existing semantic notes in [9948.sem.json](./9948.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/layers/linear.py`
- `vllm/model_executor/model_loader/loader.py`

### Diff

```diff
diff --git a/vllm/model_executor/layers/linear.py b/vllm/model_executor/layers/linear.py
index 2471c160d66b..46ef11e7d02c 100644
--- a/vllm/model_executor/layers/linear.py
+++ b/vllm/model_executor/layers/linear.py
@@ -1,3 +1,4 @@
+import itertools
 from abc import abstractmethod
 from typing import Dict, List, Optional, Tuple
 
@@ -41,12 +42,12 @@ def adjust_marlin_shard(param, shard_size, shard_offset):
 
 
 def adjust_bitsandbytes_4bit_shard(param: Parameter,
-                                   qkv_offsets: Dict[str, Tuple[int, int]],
+                                   shard_offsets: Dict[str, Tuple[int, int]],
                                    loaded_shard_id: str) -> Tuple[int, int]:
     """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""
 
-    total, _ = qkv_offsets["total"]
-    orig_offset, orig_size = qkv_offsets[loaded_shard_id]
+    total, _ = shard_offsets["total"]
+    orig_offset, orig_size = shard_offsets[loaded_shard_id]
 
     quantized_total = param.data.shape[0]
     quantized_offset = orig_offset * quantized_total // total
@@ -499,9 +500,17 @@ def weight_loader(self,
                     # Special case for Marlin.
                     shard_size, shard_offset = adjust_marlin_shard(
                         param, shard_size, shard_offset)
+
                 if use_bitsandbytes_4bit:
-                    shard_size = loaded_weight.shape[output_dim] // 2
-                    shard_offset = shard_size * shard_id
+                    index = list(itertools.accumulate([0] + self.output_sizes))
+                    orig_offsets = {
+                        str(i): (index[i], size)
+                        for i, size in enumerate(self.output_sizes)
+                    }
+                    orig_offsets["total"] = (self.output_size, 0)
+                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
+                        param, orig_offsets, str(shard_id))
+
                 loaded_weight_shard = loaded_weight.narrow(
                     output_dim, shard_offset, shard_size)
                 self.weight_loader(param, loaded_weight_shard, shard_id)
diff --git a/vllm/model_executor/model_loader/loader.py b/vllm/model_executor/model_loader/loader.py
index 936c2fe41537..34e086016226 100644
--- a/vllm/model_executor/model_loader/loader.py
+++ b/vllm/model_executor/model_loader/loader.py
@@ -5,6 +5,7 @@
 import fnmatch
 import glob
 import inspect
+import itertools
 import json
 import math
 import os
@@ -27,7 +28,9 @@
                               get_tensor_model_parallel_world_size)
 from vllm.envs import VLLM_USE_MODELSCOPE
 from vllm.logger import init_logger
-from vllm.model_executor.layers.linear import (ReplicatedLinear,
+from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
+                                               QKVParallelLinear,
+                                               ReplicatedLinear,
                                                RowParallelLinear)
 from vllm.model_executor.layers.quantization.base_config import (
     QuantizeMethodBase)
@@ -936,6 +939,34 @@ def _unquantized_generator(self, hf_weights_files, use_safetensors,
                     end_index = total_size // tp_size * (tp_rank + 1)
                     weight_sub_tensor = weight_tensor[...,
                                                       start_index:end_index]
+                # Weights have fused on disk. In this case, we assume that the
+                # weight and module use same name.
+                elif any(
+                        weight_name.startswith(module)
+                        for module in self.maybe_fused_weights_modules):
+                    # special case for fused weights
+                    # get the size of each shard weight tensor
+                    total_shard_sizes = next(
+                        (sizes for module, sizes in
+                         self.maybe_fused_weights_modules.items()
+                         if weight_name.startswith(module)))
+                    total_size = weight_tensor.size(0)
+                    assert total_size == sum(total_shard_sizes)
+                    # get the start/end index of each shard weight tensor
+                    total_start_index = list(
+                        itertools.accumulate([0] + total_shard_sizes))[:-1]
+                    shard_weights_index = [
+                        (idx + size // tp_size * tp_rank,
+                         idx + size // tp_size * (tp_rank + 1))
+                        for idx, size in zip(total_start_index,
+                                             total_shard_sizes)
+                    ]
+                    # slice and reorder the weight tensor
+                    weight_tensor = [
+                        weight_tensor[start_index:end_index, ...]
+                        for start_index, end_index in shard_weights_index
+                    ]
+                    weight_sub_tensor = torch.cat(weight_tensor, dim=0)
                 # Shard by row
                 else:
                     total_size = weight_tensor.size(0)
@@ -985,12 +1016,22 @@ def _load_weights(self, model_config: ModelConfig,
             else:
                 self.target_modules = self.default_target_modules
 
+        # Modules whose weights might have fused on disk
+        # we need their output_sizes to make shard in flight correctly with TP
+        self.maybe_fused_weights_modules: Dict[str, List[int]] = {}
+
         for name, module in model.named_modules():
             # Some modules like `ReplicatedLinear` should not have their weights
             # sharded. The reason for implementing it this way is to avoid new
             # static variable in the model implementation.
             if isinstance(module, (ReplicatedLinear, )):
                 self.unsharded_weights_modules.append(name)
+            # `QKVParallelLinear` and `MergedColumnParallelLinear` might have
+            # fused weights on disk. We need to use the output sizes of these
+            # modules to shard the weights correctly.
+            elif isinstance(module,
+                            (QKVParallelLinear, MergedColumnParallelLinear)):
+                self.maybe_fused_weights_modules[name] = module.output_sizes
             # In TP, these weights are partitioned along the column
             # dimension (dim=-1)
             elif isinstance(module, (RowParallelLinear, )):
```
