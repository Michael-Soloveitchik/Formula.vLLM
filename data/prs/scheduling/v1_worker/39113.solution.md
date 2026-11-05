# PR #39113: [Perf] Optimize redundant sync for pooling model, 3.7% Throughput Improvement

- field: scheduling  | area: v1_worker
- author: yewentao256
- state: MERGED  created: 2026-04-06T21:06:14Z  merged: 2026-04-09T06:12:23Z
- url: https://github.com/vllm-project/vllm/pull/39113
- labels: ready, v1

# Issue

## Problem

The pooling output path in the v1 GPU model runner forced a full device synchronization (`self._sync_device()` → `torch.accelerator.synchronize()`) after copying pooler results to CPU, unless async scheduling was explicitly enabled. This blocking sync stalled the host until all queued GPU work drained, serializing the output copy against subsequent steps. The async, stream/event-based copy wrapper was gated behind `use_async_scheduling`, so CUDA runs without that flag ate the redundant sync.

## Example

Serving `BAAI/bge-base-en-v1.5` embeddings with `--random-input-len 32 --random-batch-size 1 --num-prompts 4000 --max-concurrency 64`, the blocking sync cost ~2ms mean E2EL per request (58.28ms → 56.03ms) and ~3.7% throughput (1050 → 1089 req/s).

## Use case

High-QPS embedding / reranking serving of pooling (encoder) models where many small requests are batched and output latency per step dominates.

## Resources

Wastes GPU/host overlap by forcing a CPU-side stall on a full device synchronize (kernel-launch/dispatch pipeline bubble), rather than stressing raw compute.

## Type of users

Users serving pooling/embedding/cross-encoder models on CUDA-alike GPUs without async scheduling enabled.

## Kind of inputs

Short-sequence, small-batch, high-concurrency embedding requests where per-step sync overhead is a large fraction of total latency.

## Mechanism with the issue

In `_pool` in `vllm/v1/worker/gpu_model_runner.py`, the non-async branch called `_copy_pooler_output_to_cpu(...)` then `self._sync_device()` unconditionally, and `AsyncGPUPoolingModelRunnerOutput` was only returned when `self.use_async_scheduling`. So any CUDA run without async scheduling incurred a blocking full-device synchronize instead of using the non-blocking CUDA stream/event copy wrapper.

# Solution

## Fix in words

The diff inverts the branch condition in `_pool`: instead of gating the async wrapper on `use_async_scheduling`, it now returns `AsyncGPUPoolingModelRunnerOutput` (stream/event-based non-blocking copy) for all CUDA-alike platforms, and only falls back to the synchronous `_copy_pooler_output_to_cpu(...)` + `self._sync_device()` path for CPU/XPU runners that cannot use the CUDA stream/event wrapper. It adds `_get_or_create_async_output_copy_stream()` to lazily allocate `self.async_output_copy_stream` on first use, and routes both the pooling path and the sampled-token-ids path through it. Effect: the redundant blocking device sync is removed on CUDA for pooling models, letting the output copy overlap with the next step, yielding ~3.7% throughput and ~2ms lower mean E2EL.

## Author's solution

Optimize the redundant `_sync_device` for pooling models (part of issue #35631). Validated accuracy via the MTEB embedding and cross-encoder correctness unit tests, and measured ~3.7% throughput improvement on `BAAI/bge-base-en-v1.5` embedding serving via `vllm bench serve`.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index a9a4497a3e67..189587eb55cd 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -1046,6 +1046,13 @@ def _init_device_properties(self) -> None:
     def _sync_device(self) -> None:
         torch.accelerator.synchronize()
 
+    def _get_or_create_async_output_copy_stream(self) -> torch.cuda.Stream:
+        stream = self.async_output_copy_stream
+        if stream is None:
+            stream = torch.cuda.Stream()
+            self.async_output_copy_stream = stream
+        return stream
+
     def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
         """Update the cached states and the persistent batch with the scheduler
         output.
@@ -3152,21 +3159,21 @@ def _pool(
             model_runner_output.pooler_output = [None] * num_reqs
             return model_runner_output
 
-        if self.use_async_scheduling:
-            return AsyncGPUPoolingModelRunnerOutput(
-                model_runner_output=model_runner_output,
+        if not current_platform.is_cuda_alike():
+            # cpu/xpu runners cannot use the CUDA stream/event-based wrapper.
+            model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
                 raw_pooler_output=raw_pooler_output,
                 finished_mask=finished_mask,
-                async_output_copy_stream=self.async_output_copy_stream,
             )
+            self._sync_device()
+            return model_runner_output
 
-        model_runner_output.pooler_output = _copy_pooler_output_to_cpu(
+        return AsyncGPUPoolingModelRunnerOutput(
+            model_runner_output=model_runner_output,
             raw_pooler_output=raw_pooler_output,
             finished_mask=finished_mask,
+            async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
         )
-        self._sync_device()
-
-        return model_runner_output
 
     def _pad_for_sequence_parallelism(self, num_scheduled_tokens: int) -> int:
         # Pad tokens to multiple of tensor_parallel_size when
@@ -4333,7 +4340,7 @@ def propose_draft_token_ids(sampled_token_ids):
                 sampled_token_ids=sampler_output.sampled_token_ids,
                 logprobs_tensors=sampler_output.logprobs_tensors,
                 invalid_req_indices=invalid_req_indices,
-                async_output_copy_stream=self.async_output_copy_stream,
+                async_output_copy_stream=self._get_or_create_async_output_copy_stream(),
                 vocab_size=self.input_batch.vocab_size,
             )
         with record_function_or_nullcontext(

```
