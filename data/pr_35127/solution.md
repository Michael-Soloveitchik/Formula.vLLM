# PR #35127: [Perf] Optimize pooling model redundant copy, 1.8% throughput improvement

- field: scheduling  | area: v1_worker
- author: yewentao256
- state: MERGED  created: 2026-02-23T18:13:45Z  merged: 2026-02-24T12:13:12Z
- url: https://github.com/vllm-project/vllm/pull/35127
- labels: ready, v1

# Issue

## Problem

In the pooling-model path of the v1 GPU model runner, the entire raw pooler output tensor was copied device-to-host on every step, even though only the subset of requests that finished in that step actually need their embeddings returned. This wastes PCIe/NVLink transfer bandwidth and CPU time copying rows that are immediately discarded. The redundant full copy added measurable overhead to each decode/pooling step.

## Example

With chunked prefill enabled (--max-num-batched-tokens 512, --max-model-len 4096) running Qwen/Qwen3-Embedding-0.6B, most steps finish only a fraction of the in-flight requests, yet the code copied the full batch's pooler output to CPU before masking out the unfinished rows.

## Use case

Online embedding/reranking serving where a pooling model processes many concurrent requests with chunked prefill or prefix caching, so requests finish at different steps.

## Resources

Host-device transfer bandwidth (PCIe/NVLink) and CPU cycles spent on the device-to-host copy; secondarily GPU time on the copy kernel.

## Type of users

Users serving pooling/embedding/reranking models (e.g. Qwen3-Embedding) under vLLM v1, especially with chunked prefill and prefix caching enabled.

## Kind of inputs

Large batches of many requests (e.g. 2000 prompts, random input len ~2048) where only a subset of requests finish per step, so finished_mask is partially true.

## Mechanism with the issue

In `_pool` and `AsyncGPUPoolingModelRunnerOutput.__init__` in `vllm/v1/worker/gpu_model_runner.py`, `json_map_leaves` copied every leaf of `raw_pooler_output` to CPU, then a list comprehension kept `out if include else None` using `finished_mask`. So the full batch was transferred and the unfinished rows were thrown away afterward, making the copy size scale with batch size instead of finished count.

# Solution

## Fix in words

The diff adds a helper `_copy_pooler_output_to_cpu(raw_pooler_output, finished_mask)` that copies only finished rows. For a tensor output: if zero finished it returns all `None`, if all finished it copies the whole tensor, else it builds an index tensor of finished indices and `index_select`s just those rows before the `.to('cpu', non_blocking=True)` copy, scattering them back into a `[None]*num_reqs` list. For a list output it copies only entries where `finished_mask` is true. Both `_pool` and the async pooling output path now call this helper instead of `json_map_leaves` full-copy + post-mask, cutting transfer volume to the finished subset. Also adds batch-size validation against the mask and removes the now-unused `json_map_leaves` import. Result: ~1.8% throughput improvement.

## Author's solution

Author noted the pooling output was being fully copied device-to-host and then filtered to the finished items. Their fix copies only the finished items to CPU instead of the whole output, avoiding the redundant copy, yielding ~1.8% throughput gain. Correctness covered by existing pooling/chunked-prefill/prefix-cache and MTEB unit tests.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index baef3fdc47c4..0013ec3d7cf3 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -95,7 +95,6 @@
 from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
 from vllm.tracing import instrument
 from vllm.utils import length_from_prompt_token_ids_or_embeds
-from vllm.utils.jsontree import json_map_leaves
 from vllm.utils.math_utils import cdiv, round_up
 from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
 from vllm.utils.nvtx_pytorch_hooks import PytHooks
@@ -268,6 +267,51 @@ def get_output(self) -> ModelRunnerOutput:
         return output
 
 
+def _copy_pooler_output_to_cpu(
+    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
+) -> list[torch.Tensor | None]:
+    num_reqs = len(finished_mask)
+
+    if isinstance(raw_pooler_output, torch.Tensor):
+        if raw_pooler_output.shape[0] != num_reqs:
+            raise ValueError(
+                "Pooler output batch size does not match finished mask size: "
+                f"{raw_pooler_output.shape[0]} != {num_reqs}."
+            )
+
+        num_finished = sum(finished_mask)
+        if num_finished == 0:
+            return [None] * num_reqs
+        if num_finished == num_reqs:
+            return list(raw_pooler_output.to("cpu", non_blocking=True))
+
+        # partial finished
+        finished_indices = [i for i, include in enumerate(finished_mask) if include]
+        index_tensor = torch.tensor(
+            finished_indices, device=raw_pooler_output.device, dtype=torch.long
+        )
+        finished_outputs = raw_pooler_output.index_select(0, index_tensor).to(
+            "cpu", non_blocking=True
+        )
+        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
+        for i, out in zip(finished_indices, finished_outputs):
+            partial_pooler_output[i] = out
+        return partial_pooler_output
+
+    assert isinstance(raw_pooler_output, list)
+    if len(raw_pooler_output) != num_reqs:
+        raise ValueError(
+            "Pooler output batch size does not match finished mask size: "
+            f"{len(raw_pooler_output)} != {num_reqs}."
+        )
+
+    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
+    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
+        if include and out is not None:
+            pooler_output[i] = out.to("cpu", non_blocking=True)
+    return pooler_output
+
+
 class AsyncGPUPoolingModelRunnerOutput(AsyncModelRunnerOutput):
     def __init__(
         self,
@@ -289,15 +333,11 @@ def __init__(
         default_stream = torch.cuda.current_stream()
         with torch.cuda.stream(async_output_copy_stream):
             async_output_copy_stream.wait_stream(default_stream)
-            raw_pooler_output_cpu = json_map_leaves(
-                lambda x: None if x is None else x.to("cpu", non_blocking=True),
-                self._raw_pooler_output,
+            self._model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
+                raw_pooler_output=self._raw_pooler_output,
+                finished_mask=finished_mask,
             )
             self.async_copy_ready_event.record()
-            self._model_runner_output.pooler_output = [
-                out if include else None
-                for out, include in zip(raw_pooler_output_cpu, finished_mask)
-            ]
 
     def get_output(self) -> ModelRunnerOutput:
         """Copy the device tensors to the host and return a ModelRunnerOutput.
@@ -2705,14 +2745,10 @@ def _pool(
                 async_output_copy_stream=self.async_output_copy_stream,
             )
 
-        raw_pooler_output = json_map_leaves(
-            lambda x: None if x is None else x.to("cpu", non_blocking=True),
-            raw_pooler_output,
+        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
+            raw_pooler_output=raw_pooler_output,
+            finished_mask=finished_mask,
         )
-        model_runner_output.pooler_output = [
-            out if include else None
-            for out, include in zip(raw_pooler_output, finished_mask)
-        ]
         self._sync_device()
 
         return model_runner_output

```
