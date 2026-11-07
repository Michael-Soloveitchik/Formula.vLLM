# PR #19169: [Core] Batch multi modal input using pinned memory

- field: scheduling  | area: multimodal_inputs.py
- author: lgeiger
- state: MERGED  created: 2025-06-04T23:20:27Z  merged: 2025-06-10T05:44:59Z
- url: https://github.com/vllm-project/vllm/pull/19169
- labels: ready, v1, multi-modality

# Issue

## Problem

When batching multi-modal inputs for the encoder, `MultiModalKwargs.batch` stacked per-item tensors with a plain `torch.stack`, allocating the output in pageable (non-pinned) host memory. Copying pageable memory to GPU forces an extra staging step through a pinned bounce buffer inside the CUDA driver, so the CPU->GPU transfer of batched multimodal features is slower than it needs to be. Since #18862 the batched tensors are copied to GPU without any extra casting, so the stack buffer could be placed directly in pinned memory.

## Example

Gemma 3 4B with multiple images per prompt: batched image feature tensors stacked each step get copied host->device every forward pass. Author measured ~0.8-1.7% throughput gain on an internal multi-image workload once stacking used pinned memory.

## Use case

Online/offline serving of vision-language models where each scheduler step batches image (or other modality) inputs and ships them to the GPU encoder.

## Resources

PCIe/host-to-device copy bandwidth and CPU time for the stack/copy; pageable transfers underutilize PCIe vs pinned.

## Type of users

Users serving multimodal models (e.g. Gemma 3, VisionArena-style chat) especially with multiple images per prompt where mm input volume per step is non-trivial.

## Kind of inputs

Batched multi-modal feature tensors (images/video), many items per batch with matching inner shapes so they stack into one big tensor copied to GPU.

## Mechanism with the issue

`MultiModalKwargs._try_stack` built the stacked tensor via `torch.stack(tensors_)`, which allocates output in regular pageable host memory. That buffer then feeds the host->device copy in `_execute_mm_encoder`, incurring the driver's implicit pinned-staging copy each step.

# Solution

## Fix in words

Before: `_try_stack` returned `torch.stack(tensors_)` into default pageable memory. After: it pre-allocates the output with `torch.empty(len(tensors_), *tensors_[0].shape, dtype=..., device=..., pin_memory=pin_memory)` and calls `torch.stack(tensors_, out=outputs)`, so the stacked buffer lives in pinned host memory when requested. A `pin_memory` flag is threaded through `_try_stack` (recursively) and `MultiModalKwargs.batch`, and `gpu_model_runner` passes `pin_memory=self.pin_memory` at both the encoder batch call and the profile-run dummy batch. Net effect: faster CPU->GPU copy of batched mm inputs and possibly slightly faster stack.

## Author's solution

Since #18862 batched tensors are copied to GPU without extra casting, so the author stacks the inputs directly into pinned memory for faster CPU->GPU copy and maybe slightly faster torch.stack. Reported ~0.8-1.7% throughput improvement on an internal multi-image Gemma 3 4B workload and ~0.3% (noisy) on lmarena-ai/VisionArena-Chat.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/multimodal/inputs.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/multimodal/inputs.py b/vllm/multimodal/inputs.py
index 0bf5b1cf1c6c..5cb720381d94 100644
--- a/vllm/multimodal/inputs.py
+++ b/vllm/multimodal/inputs.py
@@ -680,7 +680,8 @@ def modalities(self):
         return self._items_by_modality.keys()
 
     @staticmethod
-    def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
+    def _try_stack(nested_tensors: NestedTensors,
+                   pin_memory: bool = False) -> NestedTensors:
         """
         Stack the inner dimensions that have the same shape in
         a nested list of tensors.
@@ -697,7 +698,9 @@ def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
         if isinstance(nested_tensors, (int, float)):
             return torch.tensor(nested_tensors)
 
-        stacked = [MultiModalKwargs._try_stack(t) for t in nested_tensors]
+        stacked = [
+            MultiModalKwargs._try_stack(t, pin_memory) for t in nested_tensors
+        ]
         if not is_list_of(stacked, torch.Tensor, check="all"):
             # Only tensors (not lists) can be stacked.
             return stacked
@@ -713,10 +716,16 @@ def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
             # The tensors have incompatible shapes and can't be stacked.
             return tensors_
 
-        return torch.stack(tensors_)
+        outputs = torch.empty(len(tensors_),
+                              *tensors_[0].shape,
+                              dtype=tensors_[0].dtype,
+                              device=tensors_[0].device,
+                              pin_memory=pin_memory)
+        return torch.stack(tensors_, out=outputs)
 
     @staticmethod
-    def batch(inputs_list: list["MultiModalKwargs"]) -> BatchedTensorInputs:
+    def batch(inputs_list: list["MultiModalKwargs"],
+              pin_memory: bool = False) -> BatchedTensorInputs:
         """
         Batch multiple inputs together into a dictionary.
 
@@ -738,7 +747,7 @@ def batch(inputs_list: list["MultiModalKwargs"]) -> BatchedTensorInputs:
                 item_lists[k].append(v)
 
         return {
-            k: MultiModalKwargs._try_stack(item_list)
+            k: MultiModalKwargs._try_stack(item_list, pin_memory)
             for k, item_list in item_lists.items()
         }
 
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 4a67e37781bf..eab84b86b77d 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -954,7 +954,8 @@ def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
 
         encoder_outputs = []
         for grouped_mm_inputs in grouped_mm_inputs_list:
-            batched_mm_inputs = MultiModalKwargs.batch(grouped_mm_inputs)
+            batched_mm_inputs = MultiModalKwargs.batch(
+                grouped_mm_inputs, pin_memory=self.pin_memory)
             batched_mm_inputs = MultiModalKwargs.as_kwargs(
                 batched_mm_inputs,
                 device=self.device,
@@ -1952,7 +1953,8 @@ def profile_run(self) -> None:
             ).multi_modal_data
 
             batched_dummy_mm_inputs = MultiModalKwargs.batch(
-                [dummy_mm_kwargs] * max_num_mm_items)
+                [dummy_mm_kwargs] * max_num_mm_items,
+                pin_memory=self.pin_memory)
             batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
                 batched_dummy_mm_inputs,
                 device=self.device,

```
