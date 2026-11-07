# PR #8868: [Core] Avoid metrics log noise when idle (#8868)

# Issue

## Original PR description (captured source)


1a789ae7 [Core] Avoid metrics log noise when idle

commit 1a789ae76695bca801b55c9d4531e5438a3378c4
Author: Russell Bryant <rbryant@redhat.com>
Date:   Thu Sep 26 20:56:41 2024 +0000

    [Core] Avoid metrics log noise when idle
    
    Prior to this change, there was always a log message every 10 seconds,
    even when the system is idle. The noise when the metrics are all zeros
    doesn't seem very interesting, so this change makes it so the log
    message is only emitted while active. It will log a message whenever
    metrics are non-zero, or when returning to idle from non-idle.
    Repeated messages during idle now go to DEBUG instead of INFO.
    
    Signed-off-by: Russell Bryant <rbryant@redhat.com>


## Full discussion

See [8868.md](./8868.md) and [8868.json](./8868.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/8868
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/8868.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `afdeb95e9d4868f0906890261322568f530f36ba2b9c6caa8bd13d727ef31830`

Existing semantic notes in [8868.sem.json](./8868.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/engine/metrics.py`

### Diff

```diff
diff --git a/vllm/engine/metrics.py b/vllm/engine/metrics.py
index 47472c274ccb..5bfd6a9f4b38 100644
--- a/vllm/engine/metrics.py
+++ b/vllm/engine/metrics.py
@@ -421,6 +421,11 @@ def get_throughput(tracked_stats: List[int], now: float,
 class LoggingStatLogger(StatLoggerBase):
     """LoggingStatLogger is used in LLMEngine to log to Stdout."""
 
+    def __init__(self, *args, **kwargs) -> None:
+        super().__init__(*args, **kwargs)
+        self.last_prompt_throughput: Optional[float] = None
+        self.last_generation_throughput: Optional[float] = None
+
     def log(self, stats: Stats) -> None:
         """Called by LLMEngine.
            Logs to Stdout every self.local_interval seconds."""
@@ -445,8 +450,14 @@ def log(self, stats: Stats) -> None:
                 now=stats.now,
                 last_log=self.last_local_log)
 
-            # Log to stdout.
-            logger.info(
+            log_fn = logger.info
+            if not any((prompt_throughput, generation_throughput,
+                        self.last_prompt_throughput,
+                        self.last_generation_throughput)):
+                # Avoid log noise on an idle production system
+                log_fn = logger.debug
+
+            log_fn(
                 "Avg prompt throughput: %.1f tokens/s, "
                 "Avg generation throughput: %.1f tokens/s, "
                 "Running: %d reqs, Swapped: %d reqs, "
@@ -472,11 +483,16 @@ def log(self, stats: Stats) -> None:
                     self._format_spec_decode_metrics_str(
                         self.spec_decode_metrics))
 
-            # Reset tracked stats for next interval.
-            self.num_prompt_tokens = []
-            self.num_generation_tokens = []
-            self.last_local_log = stats.now
-            self.spec_decode_metrics = None
+            self._reset(stats, prompt_throughput, generation_throughput)
+
+    def _reset(self, stats, prompt_throughput, generation_throughput) -> None:
+        # Reset tracked stats for next interval.
+        self.num_prompt_tokens = []
+        self.num_generation_tokens = []
+        self.last_local_log = stats.now
+        self.spec_decode_metrics = None
+        self.last_prompt_throughput = prompt_throughput
+        self.last_generation_throughput = generation_throughput
 
     def _format_spec_decode_metrics_str(
             self, metrics: "SpecDecodeWorkerMetrics") -> str:
```
