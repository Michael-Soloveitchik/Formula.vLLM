# PR #40891: [Core] Avoid using extra thread in `UniProcExecutor`

- field: scheduling  | area: v1_executor
- author: njhill
- state: MERGED  created: 2026-04-26T00:37:17Z  merged: 2026-05-07T22:33:00Z
- url: https://github.com/vllm-project/vllm/pull/40891
- labels: ready, v1

# Issue

## Problem

When `UniProcExecutor` ran with `max_concurrent_batches > 1` (async scheduling enabled), it spawned a dedicated single-worker `ThreadPoolExecutor` just to call `AsyncModelRunnerOutput.get_output()`. Every async batch output was handed off to this extra thread, incurring thread-handoff, scheduling, and synchronization overhead on the critical output path. In a single-process (uniproc, TP=1) deployment there is no real concurrency benefit from a separate thread — the work is still serialized, so the thread hop was pure overhead.

## Example

On a single NVIDIA GB200 running `Qwen/Qwen3-0.6B` with `--tensor-parallel-size 1 --distributed-executor-backend uni`, 256 in / 2048 out, concurrency 512 (1024 prompts): output throughput rose from ~17222 to ~18974 tok/s (+10.17%) and P90 TTFT dropped ~15% once the extra thread was removed.

## Use case

Online serving with the uniproc executor and async scheduling (overlapping model execution with output processing), the common path for single-GPU / single-process vLLM deployments.

## Resources

CPU — specifically thread-scheduling, context-switch, and inter-thread synchronization overhead on the per-batch output critical path; no GPU resource was the bottleneck here.

## Type of users

Users serving on a single process / single GPU with `--distributed-executor-backend uni` (TP=1), especially low-latency / high-concurrency workloads with small models where per-batch overhead dominates.

## Kind of inputs

High request concurrency (e.g. 512) with many small/medium requests and async scheduling active, where per-batch output handoff happens frequently and overhead accumulates.

## Mechanism with the issue

In `collective_rpc`, when `result` was an `AsyncModelRunnerOutput`, the code submitted `result.get_output` (or a list-wrapping closure) to `self.async_output_thread`, a `ThreadPoolExecutor(max_workers=1)`. The actual `get_output()` work still ran serially on that one worker thread, so the only effect was paying thread-dispatch and future-synchronization cost per batch.

# Solution

## Fix in words

The diff deletes the `async_output_thread` `ThreadPoolExecutor` and its construction (previously created when `max_concurrent_batches > 1`), removing the `ThreadPoolExecutor` import. It adds a lazy `AsyncOutputFuture(Future)` subclass that stores the `AsyncModelRunnerOutput` and `single_value` flag and only calls `get_output()` the first time `.result()` is invoked, caching via `set_result`/`set_exception` and wrapping in a list when `single_value` is false. In `collective_rpc`, an `AsyncModelRunnerOutput` result now returns `AsyncOutputFuture(result, single_value)` directly instead of submitting to the thread. This keeps the async-future interface callers expect while computing the output inline on the caller's thread, eliminating thread-handoff overhead and improving throughput and latency.

## Author's solution

Crediting @zhuohan123's idea, the author avoids the extra thread in `UniProcExecutor`, citing clear performance benefit especially in low-latency / high-concurrency cases, backed by GB200 benchmarks on Qwen3-0.6B showing up to +10.17% output throughput and ~15% lower P90 TTFT at concurrency 512.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/executor/uniproc_executor.py`

### Diff
```diff
diff --git a/vllm/v1/executor/uniproc_executor.py b/vllm/v1/executor/uniproc_executor.py
index d006946079e7..92e668406f98 100644
--- a/vllm/v1/executor/uniproc_executor.py
+++ b/vllm/v1/executor/uniproc_executor.py
@@ -2,7 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 import os
 from collections.abc import Callable
-from concurrent.futures import Future, ThreadPoolExecutor
+from concurrent.futures import Future
 from functools import cached_property
 from multiprocessing import Lock
 from typing import Any
@@ -23,6 +23,25 @@
 logger = init_logger(__name__)
 
 
+class AsyncOutputFuture(Future):
+    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
+        self.async_output = async_output
+        self.single_value = single_value
+        super().__init__()
+
+    def result(self, timeout=None):
+        if timeout is not None:
+            raise RuntimeError("timeout not implemented")
+
+        if not super().done():
+            try:
+                output = self.async_output.get_output()
+                self.set_result(output if self.single_value else [output])
+            except Exception as e:
+                self.set_exception(e)
+        return super().result()
+
+
 class UniProcExecutor(Executor):
     def _init_executor(self) -> None:
         """Initialize the worker and load the model."""
@@ -37,12 +56,6 @@ def _init_executor(self) -> None:
             shared_worker_lock=Lock(),
         )
 
-        self.async_output_thread: ThreadPoolExecutor | None = None
-        if self.max_concurrent_batches > 1:
-            self.async_output_thread = ThreadPoolExecutor(
-                max_workers=1, thread_name_prefix="WorkerAsyncOutput"
-            )
-
         self.driver_worker.init_worker(all_kwargs=[kwargs])
         self.driver_worker.init_device()
 
@@ -83,15 +96,7 @@ def collective_rpc(  # type: ignore[override]
         try:
             result = run_method(self.driver_worker, method, args, kwargs)
             if isinstance(result, AsyncModelRunnerOutput):
-                if (async_thread := self.async_output_thread) is not None:
-                    if single_value:
-                        return async_thread.submit(result.get_output)
-
-                    def get_output_list() -> list[Any]:
-                        return [result.get_output()]
-
-                    return async_thread.submit(get_output_list)
-                result = result.get_output()
+                return AsyncOutputFuture(result, single_value)
             future = Future[Any]()
             future.set_result(result if single_value else [result])
         except Exception as e:

```
