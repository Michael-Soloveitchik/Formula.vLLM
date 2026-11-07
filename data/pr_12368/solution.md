# PR #12368: [Core] add and implement `VLLM_LOGITS_PROCESSOR_THREADS`

- field: decoding_tpot  | area: vllm_envs.py
- author: akeshet
- state: MERGED  created: 2025-01-23T20:42:45Z  merged: 2025-02-05T02:46:26Z
- url: https://github.com/vllm-project/vllm/pull/12368
- labels: ready

# Issue

## Problem

vLLM applied custom logits processors sequentially, one sequence at a time, on a single Python thread. When a batch holds many sequences and each logits processor does real work (launching extra CUDA kernels or CPU-bound compute), the serial loop serializes all that work. GPU sits underutilized and inter-token latency (ITL/TPOT) grows linearly with batch size.

## Example

Large batch where each sequence runs a custom logits processor that launches its own CUDA kernel: the per-row loop dispatches kernels back-to-back instead of overlapping, so kernel-launch and compute stall the sampling step.

## Use case

Online serving with large batches and user-supplied logits processors (structured/guided decoding, custom biasing) where processor cost is non-trivial per sequence.

## Resources

GPU compute utilization and kernel-launch overlap; also CPU when processors do GIL-releasing CPU-bound work. Serial path leaves both idle.

## Type of users

Users serving with custom/heavy logits processors, especially at high batch concurrency where per-sequence processor cost dominates the decode step.

## Kind of inputs

Large batches (many sequences/seq_groups per step) combined with logits processors that either launch CUDA kernels or do significant CPU work without holding the GIL.

## Mechanism with the issue

`_apply_logits_processors` in logits_processor.py looped over each seq_group and each logits_row, calling every processor inline and writing `logits[logits_row_idx]` before moving on. No parallelism across sequences, so independent per-row processor work could not overlap.

# Solution

## Fix in words

Diff adds env var `VLLM_LOGITS_PROCESSOR_THREADS` (envs.py) and, when set, builds a module-level `ThreadPoolExecutor` of that size in logits_processor.py. Per-row processing extracted into `_apply_logits_processors_single_seq`. Before: each row processed inline in the loop. After: if the threadpool exists, each row's processing is `submit`ted as a future and collected into a list; the loop then gathers `future.result()` and writes `logits[logits_row_idx]`. This runs processor calls across sequences concurrently, overlapping their CUDA kernel launches / GIL-free CPU work, raising GPU utilization and cutting ITL on large batches. Without the env var, behavior stays serial/unchanged.

## Author's solution

Author adds `VLLM_LOGITS_PROCESSOR_THREADS`; when set, vLLM uses a threadpool of that size to multithread logits-processor calls across sequences in a batch. Intended to increase GPU utilization and decrease ITL when batches are large and processors launch extra CUDA kernels or do significant CPU-bound work without holding the GIL.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/envs.py`
- `vllm/model_executor/layers/logits_processor.py`

### Diff
```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index 8627caec7790..8a5888c93197 100644
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -29,6 +29,7 @@
     VLLM_LOGGING_LEVEL: str = "INFO"
     VLLM_LOGGING_PREFIX: str = ""
     VLLM_LOGGING_CONFIG_PATH: Optional[str] = None
+    VLLM_LOGITS_PROCESSOR_THREADS: Optional[int] = None
     VLLM_TRACE_FUNCTION: int = 0
     VLLM_ATTENTION_BACKEND: Optional[str] = None
     VLLM_USE_FLASHINFER_SAMPLER: Optional[bool] = None
@@ -275,6 +276,14 @@ def maybe_convert_int(value: Optional[str]) -> Optional[int]:
     "VLLM_LOGGING_PREFIX":
     lambda: os.getenv("VLLM_LOGGING_PREFIX", ""),
 
+    # if set, vllm will call logits processors in a thread pool with this many
+    # threads. This is useful when using custom logits processors that either
+    # (a) launch additional CUDA kernels or (b) do significant CPU-bound work
+    # while not holding the python GIL, or both.
+    "VLLM_LOGITS_PROCESSOR_THREADS":
+    lambda: int(os.getenv("VLLM_LOGITS_PROCESSOR_THREADS", "0"))
+    if "VLLM_LOGITS_PROCESSOR_THREADS" in os.environ else None,
+
     # Trace function calls
     # If set to 1, vllm will trace function calls
     # Useful for debugging
diff --git a/vllm/model_executor/layers/logits_processor.py b/vllm/model_executor/layers/logits_processor.py
index 42decde1d0f7..1454df04d2dd 100644
--- a/vllm/model_executor/layers/logits_processor.py
+++ b/vllm/model_executor/layers/logits_processor.py
@@ -1,5 +1,6 @@
 """A layer that compute logits from hidden_stats."""
 import inspect
+from concurrent.futures import ThreadPoolExecutor
 from typing import Optional
 
 import torch
@@ -14,6 +15,11 @@
 from vllm.model_executor.sampling_metadata import SamplingMetadata
 from vllm.platforms import current_platform
 
+_logits_processor_threadpool: Optional[ThreadPoolExecutor] = None
+if envs.VLLM_LOGITS_PROCESSOR_THREADS is not None:
+    _logits_processor_threadpool = ThreadPoolExecutor(
+        envs.VLLM_LOGITS_PROCESSOR_THREADS)
+
 
 class LogitsProcessor(nn.Module):
     """Process logits and apply logits processors from sampling metadata.
@@ -134,6 +140,7 @@ def _apply_logits_processors(
 ) -> torch.Tensor:
     found_logits_processors = False
     logits_processed = 0
+    logits_row_ids_and_logits_row_futures = []
     for seq_group in sampling_metadata.seq_groups:
         seq_ids = seq_group.seq_ids
         sampling_params = seq_group.sampling_params
@@ -147,22 +154,39 @@ def _apply_logits_processors(
                 past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                 prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids
 
-                for logits_processor in logits_processors:
-                    parameters = inspect.signature(logits_processor).parameters
-                    if len(parameters) == 3:
-                        logits_row = logits_processor(prompt_tokens_ids,
-                                                      past_tokens_ids,
-                                                      logits_row)
-                    else:
-                        logits_row = logits_processor(past_tokens_ids,
-                                                      logits_row)
-
-                logits[logits_row_idx] = logits_row
+                if _logits_processor_threadpool is not None:
+                    logits_row_ids_and_logits_row_futures.append(
+                        (logits_row_idx,
+                         _logits_processor_threadpool.submit(
+                             _apply_logits_processors_single_seq, logits_row,
+                             logits_processors, past_tokens_ids,
+                             prompt_tokens_ids)))
+                else:
+                    logits[logits_row_idx] = \
+                        _apply_logits_processors_single_seq(
+                            logits_row, logits_processors, past_tokens_ids,
+                            prompt_tokens_ids)
 
         logits_processed += len(seq_group.sample_indices) + len(
             seq_group.prompt_logprob_indices)
 
+    for logits_row_idx, future in logits_row_ids_and_logits_row_futures:
+        logits[logits_row_idx] = future.result()
+
     if found_logits_processors:
         # verifies that no rows in logits were missed unexpectedly
         assert logits_processed == logits.shape[0]
     return logits
+
+
+def _apply_logits_processors_single_seq(logits_row, logits_processors,
+                                        past_tokens_ids,
+                                        prompt_tokens_ids) -> torch.Tensor:
+    for logits_processor in logits_processors:
+        parameters = inspect.signature(logits_processor).parameters
+        if len(parameters) == 3:
+            logits_row = logits_processor(prompt_tokens_ids, past_tokens_ids,
+                                          logits_row)
+        else:
+            logits_row = logits_processor(past_tokens_ids, logits_row)
+    return logits_row

```
