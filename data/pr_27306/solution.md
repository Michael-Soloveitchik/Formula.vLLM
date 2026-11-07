# PR #27306: [Feature]: Add corrupted request metric to V1 metrics system. (#27306)

# Issue

## Original PR description (captured source)

## 
Signed off: sneh.lata@nutanix.com
## Purpose

The purpose of this PR is to fix: https://github.com/vllm-project/vllm/issues/27301 and https://github.com/vllm-project/vllm/issues/27298. This PR implements a new Prometheus metric `vllm:num_requests_corrupted` to track and monitor corrupted requests in vLLM. The feature provides visibility into request corruption issues that can occur during inference, enabling better monitoring and debugging capabilities.

- New Prometheus gauge metric `vllm:num_requests_corrupted`
- Console logging of corrupted request counts per logging interval.
- Enabled via the env variable  `VLLM_COMPUTE_NANS_IN_LOGITS` flag for performance control, thus it won't impact any functionality unless enabled; the default value is set to False.
- Updated documentation and updated dashboard examples config as well.

**Performance Considerations:**
- Corrupted request counting is optional and disabled by default
- When enabled, it adds no computational overhead


## Test Plan
- Not needed as it is a by-default disabled metric.

## Test Result
- N/A

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [27306.md](./27306.md) and [27306.json](./27306.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/27306
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27306.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `1941e7c73e7ebaaebdd0e587c5af379ce6b3d88f845e92286e7e0564685b3deb`

Existing semantic notes in [27306.sem.json](./27306.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/v1/metrics/test_stats.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/metrics/loggers.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/request.py`

### Diff

```diff
diff --git a/tests/v1/metrics/test_stats.py b/tests/v1/metrics/test_stats.py
index 67a2d1739b6b..b12e97a875f8 100644
--- a/tests/v1/metrics/test_stats.py
+++ b/tests/v1/metrics/test_stats.py
@@ -18,6 +18,7 @@ def test_iteration_stats_repr():
         "time_to_first_tokens_iter=[], "
         "inter_token_latencies_iter=[], "
         "waiting_lora_adapters={}, "
-        "running_lora_adapters={})"
+        "running_lora_adapters={}, "
+        "num_corrupted_reqs=0)"
     )
     assert repr(iteration_stats) == expected_repr
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index 97341c762b99..f558306e3b2f 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -1023,6 +1023,7 @@ def update_from_output(
                         kv_transfer_params=kv_transfer_params,
                         trace_headers=request.trace_headers,
                         num_cached_tokens=request.num_cached_tokens,
+                        num_nans_in_logits=request.num_nans_in_logits,
                     )
                 )
             else:
@@ -1259,7 +1260,6 @@ def make_stats(
             prefix_cache_stats=prefix_cache_stats,
             connector_prefix_cache_stats=connector_prefix_cache_stats,
             spec_decoding_stats=spec_decoding_stats,
-            num_corrupted_reqs=sum(req.is_output_corrupted for req in self.running),
             kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,
         )
 
diff --git a/vllm/v1/engine/__init__.py b/vllm/v1/engine/__init__.py
index e2c1ed7b561c..058a4bcaecb5 100644
--- a/vllm/v1/engine/__init__.py
+++ b/vllm/v1/engine/__init__.py
@@ -122,6 +122,10 @@ class EngineCoreOutput(
     # The number of tokens with prefix cache hits.
     num_cached_tokens: int = 0
 
+    # The number of NaNs in logits.
+    # A value greater than 0 indicates that the output is corrupted.
+    num_nans_in_logits: int = 0
+
     @property
     def finished(self) -> bool:
         return self.finish_reason is not None
diff --git a/vllm/v1/metrics/loggers.py b/vllm/v1/metrics/loggers.py
index e85f85bfb0aa..eb113c74a22a 100644
--- a/vllm/v1/metrics/loggers.py
+++ b/vllm/v1/metrics/loggers.py
@@ -9,6 +9,7 @@
 
 from prometheus_client import Counter, Gauge, Histogram
 
+import vllm.envs as envs
 from vllm.config import SupportsMetricsInfo, VllmConfig
 from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
     KVConnectorLogging,
@@ -116,11 +117,13 @@ def _reset(self, now):
         # Tracked stats over current local logging interval.
         self.num_prompt_tokens: int = 0
         self.num_generation_tokens: int = 0
+        self.num_corrupted_reqs: int = 0
 
     def _track_iteration_stats(self, iteration_stats: IterationStats):
         # Save tracked stats for token counters.
         self.num_prompt_tokens += iteration_stats.num_prompt_tokens
         self.num_generation_tokens += iteration_stats.num_generation_tokens
+        self.num_corrupted_reqs += iteration_stats.num_corrupted_reqs
 
     def _get_throughput(self, tracked_stats: int, now: float) -> float:
         # Compute summary metrics for tracked stats
@@ -204,6 +207,10 @@ def log(self):
             self.last_scheduler_stats.kv_cache_usage * 100,
             self.prefix_caching_metrics.hit_rate * 100,
         ]
+
+        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
+            log_parts.append("Corrupted: %d reqs")
+            log_args.append(self.num_corrupted_reqs)
         if not self.connector_prefix_caching_metrics.empty:
             log_parts.append("External prefix cache hit rate: %.1f%%")
             log_args.append(self.connector_prefix_caching_metrics.hit_rate * 100)
@@ -275,9 +282,6 @@ def aggregate_scheduler_stats(self):
             self.last_scheduler_stats.num_running_reqs += (
                 last_scheduler_stats.num_running_reqs
             )
-            self.last_scheduler_stats.num_corrupted_reqs += (
-                last_scheduler_stats.num_corrupted_reqs
-            )
             self.last_scheduler_stats.kv_cache_usage += (
                 last_scheduler_stats.kv_cache_usage
             )
@@ -481,6 +485,19 @@ def __init__(
             gauge_kv_cache_usage, engine_indexes, model_name
         )
 
+        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
+            counter_corrupted_requests = self._counter_cls(
+                name="vllm:corrupted_requests",
+                documentation=(
+                    "Corrupted requests, in terms of total number of requests "
+                    "with NaNs in logits."
+                ),
+                labelnames=labelnames,
+            )
+            self.counter_corrupted_requests = make_per_engine(
+                counter_corrupted_requests, engine_indexes, model_name
+            )
+
         counter_prefix_cache_queries = self._counter_cls(
             name="vllm:prefix_cache_queries",
             documentation=(
@@ -933,7 +950,6 @@ def record(
             self.gauge_scheduler_waiting[engine_idx].set(
                 scheduler_stats.num_waiting_reqs
             )
-
             if self.show_hidden_metrics:
                 self.gauge_gpu_cache_usage[engine_idx].set(
                     scheduler_stats.kv_cache_usage
@@ -979,7 +995,10 @@ def record(
 
         if iteration_stats is None:
             return
-
+        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
+            self.counter_corrupted_requests[engine_idx].inc(
+                iteration_stats.num_corrupted_reqs
+            )
         self.counter_num_preempted_reqs[engine_idx].inc(
             iteration_stats.num_preempted_reqs
         )
diff --git a/vllm/v1/metrics/stats.py b/vllm/v1/metrics/stats.py
index 7868141d1b1d..c5f06a66e21e 100644
--- a/vllm/v1/metrics/stats.py
+++ b/vllm/v1/metrics/stats.py
@@ -6,6 +6,7 @@
 from dataclasses import dataclass, field
 from typing import TYPE_CHECKING, Any
 
+import vllm.envs as envs
 from vllm.v1.spec_decode.metrics import SpecDecodingStats
 
 if TYPE_CHECKING:
@@ -169,8 +170,6 @@ class SchedulerStats:
     spec_decoding_stats: SpecDecodingStats | None = None
     kv_connector_stats: dict[str, Any] | None = None
 
-    num_corrupted_reqs: int = 0
-
 
 @dataclass
 class LoRAStats:
@@ -196,6 +195,9 @@ class RequestStateStats:
     # first token latency
     first_token_latency: float = 0.0
 
+    # Track if this request is corrupted (NaNs in logits)
+    is_corrupted: bool = False
+
 
 @dataclass
 class FinishedRequestStats:
@@ -211,6 +213,7 @@ class FinishedRequestStats:
     inference_time: float = 0.0
     decode_time: float = 0.0
     mean_time_per_output_token: float = 0.0
+    is_corrupted: bool = False
 
 
 class IterationStats:
@@ -228,6 +231,7 @@ def __init__(self):
         self.inter_token_latencies_iter: list[float] = []
         self.waiting_lora_adapters: dict[str, int] = {}
         self.running_lora_adapters: dict[str, int] = {}
+        self.num_corrupted_reqs: int = 0
 
     def __repr__(self) -> str:
         field_to_value_str = ", ".join(f"{k}={v}" for k, v in vars(self).items())
@@ -258,6 +262,15 @@ def update_from_output(
 
         req_stats.num_generation_tokens += num_new_generation_tokens
 
+        # Track if this request is corrupted (only check once per request)
+        # Early exit if already marked as corrupted to avoid redundant checks
+        if (
+            envs.VLLM_COMPUTE_NANS_IN_LOGITS
+            and not req_stats.is_corrupted
+            and output.num_nans_in_logits > 0
+        ):
+            req_stats.is_corrupted = True
+
         # Process request-level engine core events
         if output.events is not None:
             self.update_from_events(
@@ -339,9 +352,14 @@ def update_from_finished_request(
             inference_time=inference_time,
             decode_time=decode_time,
             mean_time_per_output_token=mean_time_per_output_token,
+            is_corrupted=req_stats.is_corrupted,
         )
         self.finished_requests.append(finished_req)
 
+        # Count corrupted requests when they finish (only once per request)
+        if req_stats.is_corrupted:
+            self.num_corrupted_reqs += 1
+
 
 class LoRARequestStates:
     """Per-LoRA request state stats."""
diff --git a/vllm/v1/request.py b/vllm/v1/request.py
index 864b0eb7fa41..7a5f1183ed48 100644
--- a/vllm/v1/request.py
+++ b/vllm/v1/request.py
@@ -168,10 +168,6 @@ def append_output_token_ids(
     def use_structured_output(self) -> bool:
         return self.structured_output_request is not None
 
-    @property
-    def is_output_corrupted(self) -> bool:
-        return self.num_nans_in_logits > 0
-
     @property
     def num_tokens(self) -> int:
         return len(self._all_token_ids)
```
