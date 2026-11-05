# PR #33612: [Perf] Optimize spec decoding + async scheduling, 1.5% Throughput improvement

- field: scheduling  | area: v1_core
- author: yewentao256
- state: MERGED  created: 2026-02-02T22:52:37Z  merged: 2026-02-04T14:34:32Z
- url: https://github.com/vllm-project/vllm/pull/33612
- labels: ready, v1

# Issue

## Problem

On every scheduling step, `_update_after_schedule` allocates a fresh list `[-1] * self.num_spec_tokens` for each running request's `request.spec_token_ids`. This per-request, per-step allocation is pure waste because the content is always identical placeholder values that get overwritten in the worker process. With many concurrent requests the repeated list construction adds CPU and allocator pressure to the hot scheduling loop.

## Example

Serving Llama-3.2-1B with EAGLE3 speculative decoding (`num_speculative_tokens=2`), `--max-num-seqs 128`, async scheduling, TP=2: each of the up to 128 running requests rebuilds a 2-element `[-1,-1]` list every step, hundreds of times per second.

## Use case

Online serving with async scheduling and speculative decoding enabled, where the scheduler runs continuously under high request concurrency.

## Resources

CPU time and Python memory allocator churn in the single-threaded scheduler hot path; no GPU resource involved.

## Type of users

Users running speculative decoding (EAGLE/EAGLE3 draft models) with async scheduling, typically multi-GPU TP setups serving at high `max-num-seqs`.

## Kind of inputs

Large batches (high `num-prompts`/`max-num-seqs`, e.g. 128), `request-rate inf`, with a fixed small `num_speculative_tokens` so the placeholder list is reconstructed identically every step.

## Mechanism with the issue

`AsyncScheduler._update_after_schedule` set `request.spec_token_ids = [-1] * self.num_spec_tokens` inline per request per step. The list value is constant placeholder (-1) data later filled by the worker, so allocating a new list each time is redundant work in the scheduler loop. `Scheduler.schedule` and `_preempt_request` mutated this list in-place (`del`, `.clear()`), which assumed a per-request owned list.

# Solution

## Fix in words

`AsyncScheduler.__init__` now builds one reusable read-only placeholder list `self._spec_token_placeholders = [-1] * self.num_spec_tokens`, and `_update_after_schedule` assigns that shared list instead of allocating a new one each step. Because the list is now shared, `schedule` no longer mutates it in place: instead of `del request.spec_token_ids[num_scheduled_spec_tokens:]` it slices a copy only when trimming is needed (`spec_token_ids[:num_scheduled_spec_tokens]`) before storing into `scheduled_spec_decode_tokens`. `_preempt_request` likewise stops calling `.clear()` on the shared list and instead rebinds `request.spec_token_ids = []` (guarded by a non-empty check). Net effect: eliminates per-request per-step list allocation, yielding ~1.5% throughput gain with no behavioral change.

## Author's solution

Author observed that recreating `request.spec_token_ids = [-1] * self.num_spec_tokens` on every `_update_after_schedule` call wastes resources, and proposed replacing it with a single reusable placeholder (`num_spec_token_placeholders`) rather than reallocating each time.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/sched/async_scheduler.py`
- `vllm/v1/core/sched/scheduler.py`

### Diff
```diff
diff --git a/vllm/v1/core/sched/async_scheduler.py b/vllm/v1/core/sched/async_scheduler.py
index 3c66a23208ec..23c610f3bde4 100644
--- a/vllm/v1/core/sched/async_scheduler.py
+++ b/vllm/v1/core/sched/async_scheduler.py
@@ -10,6 +10,11 @@
 
 
 class AsyncScheduler(Scheduler):
+    def __init__(self, *args, **kwargs) -> None:
+        super().__init__(*args, **kwargs)
+        # reusable read-only placeholder list for speculative decoding.
+        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
+
     def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
         super()._update_after_schedule(scheduler_output)
         has_structured_output_requests = False
@@ -31,9 +36,9 @@ def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
                 # The request will generate a new token plus num_spec_tokens
                 # in this scheduling step.
                 request.num_output_placeholders += 1 + cur_num_spec_tokens
-                # Add placeholders for the new tokens in spec_token_ids.
+                # Add placeholders for the new draft/spec tokens.
                 # We will update the actual spec token ids in the worker process.
-                request.spec_token_ids = [-1] * self.num_spec_tokens
+                request.spec_token_ids = self._spec_token_placeholders
 
         scheduler_output.has_structured_output_requests = has_structured_output_requests
         scheduler_output.pending_structured_output_tokens = (
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index 3f7ac9374e15..2681caf46409 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -487,9 +487,11 @@ def schedule(self) -> SchedulerOutput:
                     - request.num_output_placeholders
                 )
                 if num_scheduled_spec_tokens > 0:
-                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
-                    del request.spec_token_ids[num_scheduled_spec_tokens:]
-                    scheduled_spec_decode_tokens[request_id] = request.spec_token_ids
+                    spec_token_ids = request.spec_token_ids
+                    if len(spec_token_ids) > num_scheduled_spec_tokens:
+                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
+                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
+
                 # New spec tokens will be set in `update_draft_token_ids` before the
                 # next step when applicable.
                 request.spec_token_ids = []
@@ -887,7 +889,8 @@ def _preempt_request(self, request: Request, timestamp: float) -> None:
         self.encoder_cache_manager.free(request)
         request.status = RequestStatus.PREEMPTED
         request.num_computed_tokens = 0
-        request.spec_token_ids.clear()
+        if request.spec_token_ids:
+            request.spec_token_ids = []
         request.num_preemptions += 1
         if self.log_stats:
             request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

```
