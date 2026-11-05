# PR #19939: [PERF] Speedup of MRoPE prepare inputs

- field: attention  | area: model_executor_layers
- author: vadiklyutiy
- state: MERGED  created: 2025-06-21T17:54:51Z  merged: 2025-06-24T06:01:26Z
- url: https://github.com/vllm-project/vllm/pull/19939
- labels: ready, v1, qwen

# Issue

## Problem

`MRotaryEmbedding.get_next_input_positions_tensor` dominated the cost of `_prepare_inputs` on the CPU hot path. Each call built a fresh `torch.arange` tensor and `.expand(3, -1)`'d it, then copied the result into the preallocated `mrope_positions_cpu` buffer. The extra tensor allocation, the 3-row expand, and the tensor-to-tensor copy added per-request CPU overhead on every scheduler step.

## Example

Serving Qwen2.5-VL-3B-Instruct at 50 rps with one 512x512 image per request, `_prepare_inputs` averaged 3.869 ms; the arange/expand/copy for MRoPE positions was a large share of that time.

## Use case

Online serving of multimodal models that use M-RoPE (e.g. Qwen2.5-VL) under high request load, where per-step input preparation runs on the critical path.

## Resources

CPU time and kernel/allocation overhead in the input-prep path (not GPU compute); torch tensor allocation plus host-to-host copy per request.

## Type of users

Users serving Qwen2.5-VL and other M-RoPE vision-language models with high concurrency where CPU-side `_prepare_inputs` becomes a bottleneck.

## Kind of inputs

Many concurrent decode/completion requests with multimodal (image) inputs, each needing per-token 3D mrope position computation every scheduler step.

## Mechanism with the issue

In `gpu_model_runner._calc_mrope_positions`, every request called `get_next_input_positions_tensor`, which did `torch.arange(...).expand(3, -1)` to allocate a new tensor, then assigned it into `self.mrope_positions_cpu[:, dst_start:dst_end]`. The torch allocation, the broadcast expand across 3 rows, and the subsequent tensor copy made the hot CPU path slow under load.

# Solution

## Fix in words

The diff rewrites `get_next_input_positions_tensor` to write in-place into a caller-supplied numpy array instead of returning a new torch tensor. A `np.arange(..., dtype=out.dtype)` of length `num_new_tokens` is computed once and broadcast-assigned into `out[:, out_offset:out_offset+num_new_tokens]`, filling all 3 rows directly. `gpu_model_runner` adds `self.mrope_positions_np = self.mrope_positions_cpu.numpy()` (a zero-copy numpy view of the pinned CPU buffer) and passes it plus `out_offset=dst_start` into the function. This eliminates the per-request torch tensor allocation, the `.expand(3, -1)`, and the tensor-to-tensor copy, writing numpy values straight into the shared buffer — cutting `_prepare_inputs` from 3.869 ms to 1.487 ms (2.6x).

## Author's solution

Author found `MRotaryEmbedding.get_next_input_positions_tensor` was taking a lot of time and used numpy to speed it up, combining parts of PRs #16881 and #17617 to minimize changes while getting enough speedup. Measured with nvtx around `_prepare_inputs`: 3.869 ms before vs 1.487 ms after (2.6x), with ~1.5% end-to-end improvement on Qwen2.5-VL-3B-Instruct under high load.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/rotary_embedding.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/rotary_embedding.py b/vllm/model_executor/layers/rotary_embedding.py
index 9de2338968a1..b7bb2affc4fa 100644
--- a/vllm/model_executor/layers/rotary_embedding.py
+++ b/vllm/model_executor/layers/rotary_embedding.py
@@ -26,6 +26,7 @@
 import math
 from typing import Any, Optional, Union
 
+import numpy as np
 import torch
 import torch.nn as nn
 from transformers import PretrainedConfig
@@ -1458,15 +1459,14 @@ def get_next_input_positions(
         ]
 
     @staticmethod
-    def get_next_input_positions_tensor(
-        mrope_position_delta: int,
-        context_len: int,
-        seq_len: int,
-    ) -> torch.Tensor:
-        return torch.arange(
-            mrope_position_delta + context_len,
-            mrope_position_delta + seq_len,
-        ).expand(3, -1)
+    def get_next_input_positions_tensor(out: np.ndarray, out_offset: int,
+                                        mrope_position_delta: int,
+                                        context_len: int, num_new_tokens: int):
+
+        values = np.arange(mrope_position_delta + context_len,
+                           mrope_position_delta + context_len + num_new_tokens,
+                           dtype=out.dtype)
+        out[:, out_offset:out_offset + num_new_tokens] = values
 
     @classmethod
     def omni_get_updates_use_audio_in_video(
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index f96fb64342c9..c3b04a3f2da1 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -261,6 +261,7 @@ def __init__(
                 dtype=torch.int64,
                 device="cpu",
                 pin_memory=self.pin_memory)
+            self.mrope_positions_np = self.mrope_positions_cpu.numpy()
 
         # Only relevant for models using ALiBi (e.g, MPT)
         self.use_alibi = check_use_alibi(model_config)
@@ -888,15 +889,13 @@ def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
                 dst_start = mrope_pos_ptr
                 dst_end = mrope_pos_ptr + completion_part_len
 
-                self.mrope_positions_cpu[:, dst_start:dst_end] = \
-                    MRotaryEmbedding.get_next_input_positions_tensor(
-                        req.mrope_position_delta,
-                        context_len=num_computed_tokens +
-                        prompt_part_len,
-                        seq_len=num_computed_tokens +
-                        prompt_part_len +
-                        completion_part_len,
-                    )
+                MRotaryEmbedding.get_next_input_positions_tensor(
+                    out=self.mrope_positions_np,
+                    out_offset=dst_start,
+                    mrope_position_delta=req.mrope_position_delta,
+                    context_len=num_computed_tokens + prompt_part_len,
+                    num_new_tokens=completion_part_len,
+                )
 
                 mrope_pos_ptr += completion_part_len
 

```
