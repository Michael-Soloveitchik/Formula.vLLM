# PR #11081: [torch.compile] allow tracking forward time (#11081)

# Issue

## Original PR description (captured source)

when benchmarking `torch.compile` performance, I always get this question: if the performance gain is not satisfactory, is it because `torch.compile` does not optimize the model well, or is it because of the scheduling overhead?

this pr adds the tracking for forward time, so that we can directly test the perf of `torch.compile` for certain sizes.

e.g.

```bash
$ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy
INFO 12-10 19:52:26 forward_context.py:88] Batchsize forward time stats (batchsize, count, median_time(ms)): [(1, 5054, 6.5), (32, 41, 7.51)]

$ VLLM_LOG_BATCHSIZE_INTERVAL=1.0 python3 benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-8B --batch-size 1 --load-format dummy -O "{'level': 3, 'candidate_compile_sizes': [1]}"
INFO 12-10 19:54:34 forward_context.py:88] Batchsize forward time stats (batchsize, count, median_time(ms)): [(1, 5049, 5.93), (32, 41, 7.35)]
```

then it is clear that the forward time improves from 6.5ms to 5.93ms, 8.8% improvement. And the end-to-end 7.7% improvement in latency in https://github.com/vllm-project/vllm/pull/11078 is shadowed a little bit in the end-to-end pipeline.

## Full discussion

See [11081.md](./11081.md) and [11081.json](./11081.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/11081
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/11081.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `0de4f3525ab12ca0548c871d917e6a30b75cd9f4db715de261b81fbd02c16688`

Existing semantic notes in [11081.sem.json](./11081.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/forward_context.py`

### Diff

```diff
diff --git a/vllm/forward_context.py b/vllm/forward_context.py
index cd136f43c0c5..7f56575279e9 100644
--- a/vllm/forward_context.py
+++ b/vllm/forward_context.py
@@ -1,9 +1,11 @@
 import time
-from collections import Counter
+from collections import defaultdict
 from contextlib import contextmanager
 from dataclasses import dataclass
 from typing import Any, Dict, Optional
 
+import torch
+
 import vllm.envs as envs
 from vllm.config import VllmConfig
 from vllm.logger import init_logger
@@ -11,9 +13,10 @@
 logger = init_logger(__name__)
 
 track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
-batchsize_counter: Counter = Counter()
 last_logging_time: float = 0
+forward_start_time: float = 0
 batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
+batchsize_forward_time: defaultdict = defaultdict(list)
 
 
 @dataclass
@@ -40,23 +43,10 @@ def set_forward_context(context: Any, vllm_config: VllmConfig):
     can be attention metadata, etc.
     Here we can inject common logic for every model forward pass.
     """
-    global track_batchsize, batchsize_counter
-    global last_logging_time, batchsize_logging_interval
-    if track_batchsize and context is not None:
-        if hasattr(context, "num_prefill_tokens"):
-            # for v0 attention backends
-            batchsize = context.num_prefill_tokens + context.num_decode_tokens
-        else:
-            # for v1 attention backends
-            batchsize = context.num_input_tokens
-        batchsize_counter[batchsize] += 1
-        if time.monotonic() - last_logging_time > batchsize_logging_interval:
-            last_logging_time = time.monotonic()
-            sorted_data = sorted(batchsize_counter.items(),
-                                 key=lambda x: x[1],
-                                 reverse=True)
-            logger.info("Batchsize distribution (batchsize, count): %s",
-                        sorted_data)
+    global forward_start_time
+    need_to_track_batchsize = track_batchsize and context is not None
+    if need_to_track_batchsize:
+        forward_start_time = time.perf_counter()
     global _forward_context
     prev_context = _forward_context
     _forward_context = ForwardContext(
@@ -66,4 +56,37 @@ def set_forward_context(context: Any, vllm_config: VllmConfig):
     try:
         yield
     finally:
+        global batchsize_counter
+        global last_logging_time, batchsize_logging_interval
+        if need_to_track_batchsize:
+            if hasattr(context, "num_prefill_tokens"):
+                # for v0 attention backends
+                batchsize = context.num_prefill_tokens + \
+                    context.num_decode_tokens
+            else:
+                # for v1 attention backends
+                batchsize = context.num_input_tokens
+            # we use synchronous scheduling right now,
+            # adding a sync point here should not affect
+            # scheduling of the next batch
+            torch.cuda.synchronize()
+            now = time.perf_counter()
+            # time measurement is in milliseconds
+            batchsize_forward_time[batchsize].append(
+                (now - forward_start_time) * 1000)
+            if now - last_logging_time > batchsize_logging_interval:
+                last_logging_time = now
+                forward_stats = []
+                for bs, times in batchsize_forward_time.items():
+                    if len(times) <= 1:
+                        # can be cudagraph / profiling run
+                        continue
+                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
+                    medium = round(medium, 2)
+                    forward_stats.append((bs, len(times), medium))
+                forward_stats.sort(key=lambda x: x[1], reverse=True)
+                if forward_stats:
+                    logger.info(("Batchsize forward time stats "
+                                 "(batchsize, count, median_time(ms)): %s"),
+                                forward_stats)
         _forward_context = prev_context
```
