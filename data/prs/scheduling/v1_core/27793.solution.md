# PR #27793: [Core][Observability] Add KV cache residency metrics (#27793)

# Issue

## Original PR description (captured source)

## Purpose

This PR introduces **three new Prometheus histograms** for fine-grained observability of KV cache residency behavior:

* `vllm:kv_block_lifetime_seconds` — total lifetime from allocation to free
* `vllm:kv_block_idle_before_evict_seconds` — idle duration before eviction
* `vllm:kv_block_reuse_gap_seconds` — time between consecutive reuses of the same block

These metrics help operators analyze KV cache efficiency, reuse patterns, and eviction timing beyond simple utilization rates.
Implementation uses **monotonic timestamps** for accuracy, **1% sampling** for minimal overhead (~48 bytes/block), and is fully thread-safe with **zero runtime cost when disabled**.

Two new runtime flags are introduced:

* `--kv-cache-metrics` – enable KV cache residency metrics
* `--kv-cache-metrics-sample` – control sampling ratio (default: 0.01)

---

## Test Plan

* **Unit tests**  
  * `pytest -v tests/v1/core/test_kv_cache_metrics.py`  
    * Deterministic fake clock for lifetime/idle/prefix calculations  
    * Sampling and concurrency smoke coverage  

---

* **Integration smoke (CPU)**  
  * `python -m vllm.entrypoints.openai.api_server --model gpt2 --host 127.0.0.1 --port 8000 --kv-cache-metrics --kv-cache-metrics-sample 1.0`  
  * Sent two `/v1/completions` requests  
  * `curl http://127.0.0.1:8000/metrics | grep 'vllm:kv_block_'` → lifetime/idle/reuse histograms present  

---

* **Manual LLM snapshot (CPU)**  
  * `LLM(model="gpt2", kv_cache_metrics=True, kv_cache_metrics_sample=1.0, disable_log_stats=False)`  
  * Generated a short prompt, ran `llm.llm_engine.do_log_stats()`  
  * Checked `get_prometheus_registry().collect()` — all four KV histograms registered  

---

## Test Result

```
pytest -v tests/v1/core/test_kv_cache_metrics.py
```
(passed locally)

---

`/metrics` excerpt after the CPU smoke run:
```
# TYPE vllm:kv_block_lifetime_seconds histogram
# TYPE vllm:kv_block_idle_before_evict_seconds histogram
# TYPE vllm:kv_block_reuse_gap_seconds histogram
```

---

PromQL sanity check (validated syntactically):
```promql
rate(vllm:kv_block_lifetime_seconds_sum[5m])
/
rate(vllm:kv_block_lifetime_seconds_count[5m])
```

Fixes #25725

## Full discussion

See [27793.md](./27793.md) and [27793.json](./27793.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/27793
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27793.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `84bd388c9a10b534c0821699417aaa4658d43dd72d81fdba1a6c70d4e7025641`

Existing semantic notes in [27793.sem.json](./27793.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `docs/design/metrics.md`
- `tests/v1/core/test_kv_cache_metrics.py`
- `vllm/config/observability.py`
- `vllm/engine/arg_utils.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_metrics.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/metrics/loggers.py`
- `vllm/v1/metrics/stats.py`

### Diff

```diff
diff --git a/docs/design/metrics.md b/docs/design/metrics.md
index 313c9aaebd26..59cb6ba46fe1 100644
--- a/docs/design/metrics.md
+++ b/docs/design/metrics.md
@@ -263,6 +263,29 @@ record:
 - End-to-end latency - the interval between frontend `arrival_time`
   and the frontend receiving the final token.
 
+### KV Cache Residency Metrics
+
+We also emit a set of histograms that describe how long sampled KV cache
+blocks stay resident and how often they are reused. Sampling
+(`--kv-cache-metrics-sample`) keeps the overhead tiny; when a block is
+chosen we record:
+
+- `lifetime` – allocation ⟶ eviction
+- `idle before eviction` – last touch ⟶ eviction
+- `reuse gaps` – the pauses between touches when the block gets reused
+
+Those map directly to the Prometheus metrics:
+
+- `vllm:kv_block_lifetime_seconds` – how long each sampled block exists.
+- `vllm:kv_block_idle_before_evict_seconds` – idle tail after the final access.
+- `vllm:kv_block_reuse_gap_seconds` – time between consecutive touches.
+
+The engine core only ships raw eviction events via `SchedulerStats`; the
+frontend drains them, turns them into Prometheus observations, and also
+exposes the same data through `LLM.get_metrics()` when logging is on.
+Looking at lifetime and idle time on one chart makes it easy to spot
+stranded cache or workloads that pin prompts for a long decode.
+
 ### Metrics Publishing - Logging
 
 The `LoggingStatLogger` metrics publisher outputs a log `INFO` message
diff --git a/tests/v1/core/test_kv_cache_metrics.py b/tests/v1/core/test_kv_cache_metrics.py
new file mode 100644
index 000000000000..9e16aa64ab6a
--- /dev/null
+++ b/tests/v1/core/test_kv_cache_metrics.py
@@ -0,0 +1,224 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+from unittest.mock import patch
+
+import pytest
+
+from vllm.v1.core.kv_cache_metrics import (
+    BlockMetricsState,
+    KVCacheMetricsCollector,
+)
+from vllm.v1.core.kv_cache_utils import KVCacheBlock
+
+
+class TestBlockMetricsState:
+    def test_init(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+            assert state.birth_time_ns == 1000000000
+            assert state.last_access_ns == 1000000000
+            assert len(state.access_history) == 0
+
+    def test_access_tracking(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+
+        with patch("time.monotonic_ns", return_value=2000000000):
+            state.record_access()
+
+        assert state.last_access_ns == 2000000000
+        assert list(state.access_history) == [2000000000]
+
+    def test_ring_buffer_wraps_at_4(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+
+        for i in range(5):
+            t = 1000000000 + (i + 1) * 1000000000
+            with patch("time.monotonic_ns", return_value=t):
+                state.record_access()
+
+        assert len(state.access_history) == 4
+        assert list(state.access_history) == [
+            3000000000,
+            4000000000,
+            5000000000,
+            6000000000,
+        ]
+
+    def test_lifetime(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+        with patch("time.monotonic_ns", return_value=6500000000):
+            assert abs(state.get_lifetime_seconds() - 5.5) < 0.001
+
+    def test_idle_time(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+        state.last_access_ns = 2000000000
+        with patch("time.monotonic_ns", return_value=5200000000):
+            assert abs(state.get_idle_time_seconds() - 3.2) < 0.001
+
+    def test_reuse_gaps(self):
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+
+        base = 1000000000
+        for offset in [0, 1.5, 3.0, 5.5]:
+            state.access_history.append(base + int(offset * 1e9))
+
+        gaps = state.get_reuse_gaps_seconds()
+        assert len(gaps) == 3
+        assert gaps[0] == 1.5 and gaps[1] == 1.5 and gaps[2] == 2.5
+
+    def test_ring_wrap_only_gives_3_gaps(self):
+        # 5 accesses in size-4 buffer = 3 gaps
+        with patch("time.monotonic_ns", return_value=1000000000):
+            state = BlockMetricsState()
+
+        for i in range(5):
+            state.access_history.append(1000000000 + i * 1000000000)
+
+        assert len(state.get_reuse_gaps_seconds()) == 3
+
+
+class TestKVCacheMetricsCollector:
+    def test_sample_rate_validation(self):
+        with pytest.raises(AssertionError):
+            KVCacheMetricsCollector(sample_rate=-0.1)
+        with pytest.raises(AssertionError):
+            KVCacheMetricsCollector(sample_rate=1.5)
+        with pytest.raises(AssertionError):
+            KVCacheMetricsCollector(sample_rate=0.0)
+
+    def test_sampling(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+        assert sum(1 for _ in range(100) if c.should_sample_block()) == 100
+
+        c = KVCacheMetricsCollector(sample_rate=0.5)
+        samples = sum(1 for _ in range(1000) if c.should_sample_block())
+        assert 400 < samples < 600
+
+    def test_alloc(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+
+        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
+        with patch("time.monotonic_ns", return_value=1000000000):
+            for block in blocks:
+                c.on_block_allocated(block)
+
+        assert len(c.block_metrics) == 5
+
+    def test_access(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+        block = KVCacheBlock(block_id=0)
+
+        with patch("time.monotonic_ns", return_value=1000000000):
+            c.on_block_allocated(block)
+
+        for i in range(3):
+            t = 1000000000 + (i + 1) * 1000000000
+            with patch("time.monotonic_ns", return_value=t):
+                c.on_block_accessed(block)
+
+        assert len(c.block_metrics[0].access_history) == 3
+
+    def test_evict_no_accesses(self):
+        # lifetime should equal idle if never accessed
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+
+        block = KVCacheBlock(block_id=0)
+        with patch("time.monotonic_ns", return_value=1000000000):
+            c.on_block_allocated(block)
+
+        with patch("time.monotonic_ns", return_value=6000000000):
+            c.on_block_evicted(block)
+
+        events = c.drain_events()
+        assert len(events) == 1
+        assert abs(events[0].lifetime_seconds - 5.0) < 0.001
+        assert abs(events[0].idle_seconds - 5.0) < 0.001
+
+    def test_evict(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+
+        block = KVCacheBlock(block_id=0)
+        with patch("time.monotonic_ns", return_value=1000000000):
+            c.on_block_allocated(block)
+
+        with patch("time.monotonic_ns", return_value=2000000000):
+            c.on_block_accessed(block)
+        with patch("time.monotonic_ns", return_value=3000000000):
+            c.on_block_accessed(block)
+
+        with patch("time.monotonic_ns", return_value=4000000000):
+            c.on_block_evicted(block)
+
+        events = c.drain_events()
+        assert len(events) == 1
+        sample = events[0]
+        assert abs(sample.lifetime_seconds - 3.0) < 0.001
+        assert abs(sample.idle_seconds - 1.0) < 0.001
+        assert sample.reuse_gaps_seconds == (1.0,)
+        assert 0 not in c.block_metrics
+
+    def test_reset(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+
+        with patch("time.monotonic_ns", return_value=1000000000):
+            for i in range(5):
+                c.on_block_allocated(KVCacheBlock(block_id=i))
+
+        assert len(c.block_metrics) == 5
+        c.reset()
+        assert len(c.block_metrics) == 0
+
+        with patch("time.monotonic_ns", return_value=2000000000):
+            c.on_block_allocated(KVCacheBlock(block_id=10))
+        assert 10 in c.block_metrics
+
+    def test_huge_time_jump(self):
+        c = KVCacheMetricsCollector(sample_rate=1.0)
+
+        block = KVCacheBlock(block_id=0)
+        with patch("time.monotonic_ns", return_value=1000000000):
+            c.on_block_allocated(block)
+
+        with patch("time.monotonic_ns", return_value=9999999999999999):
+            c.on_block_evicted(block)
+
+        events = c.drain_events()
+        assert len(events) == 1
+        assert events[0].lifetime_seconds > 0
+
+
+def test_kv_cache_metrics_collector_smoke() -> None:
+    """Simple smoke test for KVCacheMetricsCollector on CPU."""
+    collector = KVCacheMetricsCollector(sample_rate=1.0)
+    block = KVCacheBlock(block_id=123)
+
+    # Allocate at t = 1.0s.
+    with patch("time.monotonic_ns", return_value=1_000_000_000):
+        collector.on_block_allocated(block)
+
+    # Access at t = 2.0s and t = 3.0s.
+    with patch("time.monotonic_ns", return_value=2_000_000_000):
+        collector.on_block_accessed(block)
+    with patch("time.monotonic_ns", return_value=3_000_000_000):
+        collector.on_block_accessed(block)
+
+    # Evict at t = 4.0s.
+    with patch("time.monotonic_ns", return_value=4_000_000_000):
+        collector.on_block_evicted(block)
+
+    events = collector.drain_events()
+    assert len(events) == 1
+
+    event = events[0]
+    # Lifetime: 1.0s → 4.0s.
+    assert abs(event.lifetime_seconds - 3.0) < 1e-6
+    # Idle: last access at 3.0s, evicted at 4.0s.
+    assert abs(event.idle_seconds - 1.0) < 1e-6
+    # One reuse gap between the two accesses.
+    assert event.reuse_gaps_seconds == (1.0,)
diff --git a/vllm/config/observability.py b/vllm/config/observability.py
index ff35e12fe20e..656a5f8a9068 100644
--- a/vllm/config/observability.py
+++ b/vllm/config/observability.py
@@ -5,7 +5,7 @@
 from typing import Any, Literal, cast
 
 from packaging.version import parse
-from pydantic import field_validator, model_validator
+from pydantic import Field, field_validator, model_validator
 from pydantic.dataclasses import dataclass
 
 from vllm import version
@@ -47,6 +47,14 @@ def show_hidden_metrics(self) -> bool:
     Note that collecting detailed timing information for each request can be
     expensive."""
 
+    kv_cache_metrics: bool = False
+    """Enable KV cache residency metrics (lifetime, idle time, reuse gaps).
+    Uses sampling to minimize overhead.
+    Requires log stats to be enabled (i.e., --disable-log-stats not set)."""
+
+    kv_cache_metrics_sample: float = Field(default=0.01, gt=0, le=1)
+    """Sampling rate for KV cache metrics (0.0, 1.0]. Default 0.01 = 1% of blocks."""
+
     @cached_property
     def collect_model_forward_time(self) -> bool:
         """Whether to collect model forward time for the request."""
diff --git a/vllm/engine/arg_utils.py b/vllm/engine/arg_utils.py
index ad5a34c56161..72482b2b45a9 100644
--- a/vllm/engine/arg_utils.py
+++ b/vllm/engine/arg_utils.py
@@ -517,6 +517,10 @@ class EngineArgs:
     collect_detailed_traces: list[DetailedTraceModules] | None = (
         ObservabilityConfig.collect_detailed_traces
     )
+    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics
+    kv_cache_metrics_sample: float = get_field(
+        ObservabilityConfig, "kv_cache_metrics_sample"
+    )
     scheduling_policy: SchedulerPolicy = SchedulerConfig.policy
     scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls
 
@@ -1002,6 +1006,13 @@ def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
             "--collect-detailed-traces",
             **observability_kwargs["collect_detailed_traces"],
         )
+        observability_group.add_argument(
+            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
+        )
+        observability_group.add_argument(
+            "--kv-cache-metrics-sample",
+            **observability_kwargs["kv_cache_metrics_sample"],
+        )
 
         # Scheduler arguments
         scheduler_kwargs = get_kwargs(SchedulerConfig)
@@ -1687,6 +1698,8 @@ def create_engine_config(
             show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,
             otlp_traces_endpoint=self.otlp_traces_endpoint,
             collect_detailed_traces=self.collect_detailed_traces,
+            kv_cache_metrics=self.kv_cache_metrics,
+            kv_cache_metrics_sample=self.kv_cache_metrics_sample,
         )
 
         # Compilation config overrides
diff --git a/vllm/v1/core/block_pool.py b/vllm/v1/core/block_pool.py
index 8b0e8fd3a241..cfb2c02e00f1 100644
--- a/vllm/v1/core/block_pool.py
+++ b/vllm/v1/core/block_pool.py
@@ -11,6 +11,7 @@
     KVCacheEvent,
 )
 from vllm.logger import init_logger
+from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
 from vllm.v1.core.kv_cache_utils import (
     BlockHash,
     BlockHashList,
@@ -140,6 +141,7 @@ class BlockPool:
             where different KV cache groups have different block sizes, the
             actual block size can be a multiple of hash_block_size.
         enable_kv_cache_events: Whether to enable kv cache events.
+        metrics_collector: Optional metrics collector for tracking block residency.
     """
 
     def __init__(
@@ -148,6 +150,7 @@ def __init__(
         enable_caching: bool,
         hash_block_size: int,
         enable_kv_cache_events: bool = False,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ):
         assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
         self.num_gpu_blocks = num_gpu_blocks
@@ -174,6 +177,8 @@ def __init__(
         self.enable_kv_cache_events = enable_kv_cache_events
         self.kv_event_queue: list[KVCacheEvent] = []
 
+        self.metrics_collector = metrics_collector
+
     def get_cached_block(
         self, block_hash: BlockHash, kv_cache_group_ids: list[int]
     ) -> list[KVCacheBlock] | None:
@@ -308,10 +313,14 @@ def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
                 self._maybe_evict_cached_block(block)
                 assert block.ref_cnt == 0
                 block.ref_cnt += 1
+                if self.metrics_collector:
+                    self.metrics_collector.on_block_allocated(block)
         else:
             for block in ret:
                 assert block.ref_cnt == 0
                 block.ref_cnt += 1
+                if self.metrics_collector:
+                    self.metrics_collector.on_block_allocated(block)
         return ret
 
     def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
@@ -325,6 +334,10 @@ def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
         Returns:
             True if the block is evicted, False otherwise.
         """
+        # Clean up metrics tracking first to prevent leaks
+        if self.metrics_collector:
+            self.metrics_collector.on_block_evicted(block)
+
         block_hash = block.block_hash
         if block_hash is None:
             # The block doesn't have hash, eviction is not needed
@@ -365,6 +378,8 @@ def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
                 if block.ref_cnt == 0 and not block.is_null:
                     self.free_block_queue.remove(block)
                 block.ref_cnt += 1
+                if self.metrics_collector:
+                    self.metrics_collector.on_block_accessed(block)
 
     def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
         """Free a list of blocks. The blocks should be ordered by their
@@ -407,6 +422,9 @@ def reset_prefix_cache(self) -> bool:
         for block in self.blocks:
             block.reset_hash()
 
+        if self.metrics_collector:
+            self.metrics_collector.reset()
+
         logger.info("Successfully reset prefix cache")
 
         if self.enable_kv_cache_events:
diff --git a/vllm/v1/core/kv_cache_coordinator.py b/vllm/v1/core/kv_cache_coordinator.py
index fd1ec8e27fba..4b09b76c1c59 100644
--- a/vllm/v1/core/kv_cache_coordinator.py
+++ b/vllm/v1/core/kv_cache_coordinator.py
@@ -5,6 +5,7 @@
 from math import lcm
 
 from vllm.v1.core.block_pool import BlockPool
+from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
 from vllm.v1.core.kv_cache_utils import (
     BlockHash,
     BlockHashList,
@@ -39,6 +40,7 @@ def __init__(
         dcp_world_size: int,
         pcp_world_size: int,
         hash_block_size: int,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ):
         self.kv_cache_config = kv_cache_config
         self.max_model_len = max_model_len
@@ -49,6 +51,7 @@ def __init__(
             enable_caching,
             hash_block_size,
             enable_kv_cache_events,
+            metrics_collector,
         )
 
         # Needs special handling for find_longest_cache_hit if eagle is enabled
@@ -228,6 +231,7 @@ def __init__(
         dcp_world_size: int,
         pcp_world_size: int,
         hash_block_size: int,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ):
         super().__init__(
             kv_cache_config,
@@ -238,6 +242,7 @@ def __init__(
             dcp_world_size=dcp_world_size,
             pcp_world_size=pcp_world_size,
             hash_block_size=hash_block_size,
+            metrics_collector=metrics_collector,
         )
         self.num_single_type_manager = len(self.single_type_managers)
 
@@ -272,6 +277,7 @@ def __init__(
         dcp_world_size: int,
         pcp_world_size: int,
         hash_block_size: int,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ):
         super().__init__(
             kv_cache_config,
@@ -282,6 +288,7 @@ def __init__(
             dcp_world_size=dcp_world_size,
             pcp_world_size=pcp_world_size,
             hash_block_size=hash_block_size,
+            metrics_collector=metrics_collector,
         )
         self.kv_cache_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
         self.block_size = self.kv_cache_spec.block_size
@@ -338,6 +345,7 @@ def __init__(
         dcp_world_size: int,
         pcp_world_size: int,
         hash_block_size: int,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ):
         super().__init__(
             kv_cache_config,
@@ -348,6 +356,7 @@ def __init__(
             dcp_world_size=dcp_world_size,
             pcp_world_size=pcp_world_size,
             hash_block_size=hash_block_size,
+            metrics_collector=metrics_collector,
         )
         # hash_block_size: the block size used to compute block hashes.
         # The actual block size usually equals hash_block_size, but in cases where
@@ -523,6 +532,7 @@ def get_kv_cache_coordinator(
     dcp_world_size: int,
     pcp_world_size: int,
     hash_block_size: int,
+    metrics_collector: KVCacheMetricsCollector | None = None,
 ) -> KVCacheCoordinator:
     if not enable_caching:
         return KVCacheCoordinatorNoPrefixCache(
@@ -530,9 +540,10 @@ def get_kv_cache_coordinator(
             max_model_len,
             use_eagle,
             enable_kv_cache_events,
-            dcp_world_size,
-            pcp_world_size,
-            hash_block_size,
+            dcp_world_size=dcp_world_size,
+            pcp_world_size=pcp_world_size,
+            hash_block_size=hash_block_size,
+            metrics_collector=metrics_collector,
         )
     if len(kv_cache_config.kv_cache_groups) == 1:
         return UnitaryKVCacheCoordinator(
@@ -541,9 +552,10 @@ def get_kv_cache_coordinator(
             use_eagle,
             enable_caching,
             enable_kv_cache_events,
-            dcp_world_size,
-            pcp_world_size,
-            hash_block_size,
+            dcp_world_size=dcp_world_size,
+            pcp_world_size=pcp_world_size,
+            hash_block_size=hash_block_size,
+            metrics_collector=metrics_collector,
         )
     return HybridKVCacheCoordinator(
         kv_cache_config,
@@ -551,7 +563,8 @@ def get_kv_cache_coordinator(
         use_eagle,
         enable_caching,
         enable_kv_cache_events,
-        dcp_world_size,
-        pcp_world_size,
-        hash_block_size,
+        dcp_world_size=dcp_world_size,
+        pcp_world_size=pcp_world_size,
+        hash_block_size=hash_block_size,
+        metrics_collector=metrics_collector,
     )
diff --git a/vllm/v1/core/kv_cache_manager.py b/vllm/v1/core/kv_cache_manager.py
index b061e5cc831d..3823384881cd 100644
--- a/vllm/v1/core/kv_cache_manager.py
+++ b/vllm/v1/core/kv_cache_manager.py
@@ -9,6 +9,7 @@
 from vllm.distributed.kv_events import KVCacheEvent
 from vllm.logger import init_logger
 from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
+from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
 from vllm.v1.core.kv_cache_utils import KVCacheBlock
 from vllm.v1.kv_cache_interface import KVCacheConfig
 from vllm.v1.metrics.stats import PrefixCacheStats
@@ -102,12 +103,14 @@ def __init__(
         enable_kv_cache_events: bool = False,
         dcp_world_size: int = 1,
         pcp_world_size: int = 1,
+        metrics_collector: KVCacheMetricsCollector | None = None,
     ) -> None:
         self.max_model_len = max_model_len
 
         self.enable_caching = enable_caching
         self.use_eagle = use_eagle
         self.log_stats = log_stats
+        self.metrics_collector = metrics_collector
         # FIXME: make prefix cache stats conditional on log_stats. We still need
         # this comment because when the log stats is enabled there are still
         # potential configs we could expose in the future.
@@ -122,6 +125,7 @@ def __init__(
             dcp_world_size=dcp_world_size,
             pcp_world_size=pcp_world_size,
             hash_block_size=hash_block_size,
+            metrics_collector=self.metrics_collector,
         )
         self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
         self.block_pool = self.coordinator.block_pool
diff --git a/vllm/v1/core/kv_cache_metrics.py b/vllm/v1/core/kv_cache_metrics.py
new file mode 100644
index 000000000000..a6dbf5b1e403
--- /dev/null
+++ b/vllm/v1/core/kv_cache_metrics.py
@@ -0,0 +1,96 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""KV cache metrics tracking."""
+
+import random
+import time
+from collections import deque
+from typing import TYPE_CHECKING
+
+if TYPE_CHECKING:
+    from vllm.v1.core.kv_cache_utils import KVCacheBlock
+
+from vllm.v1.metrics.stats import KVCacheEvictionEvent
+
+
+class BlockMetricsState:
+    """Tracks lifecycle metrics for a single KV cache block."""
+
+    def __init__(self):
+        now_ns = time.monotonic_ns()
+        self.birth_time_ns = now_ns
+        self.last_access_ns = now_ns
+        # Bounded to prevent unbounded growth if a block is accessed many times.
+        self.access_history: deque[int] = deque(maxlen=4)
+
+    def record_access(self) -> None:
+        now_ns = time.monotonic_ns()
+        self.last_access_ns = now_ns
+        self.access_history.append(now_ns)
+
+    def get_lifetime_seconds(self) -> float:
+        now_ns = time.monotonic_ns()
+        return (now_ns - self.birth_time_ns) / 1e9
+
+    def get_idle_time_seconds(self) -> float:
+        now_ns = time.monotonic_ns()
+        return (now_ns - self.last_access_ns) / 1e9
+
+    def get_reuse_gaps_seconds(self) -> list[float]:
+        if len(self.access_history) < 2:
+            return []
+        history = list(self.access_history)
+        return [(history[i] - history[i - 1]) / 1e9 for i in range(1, len(history))]
+
+
+class KVCacheMetricsCollector:
+    """Collects KV cache residency metrics with sampling."""
+
+    def __init__(self, sample_rate: float = 0.01):
+        assert 0 < sample_rate <= 1.0, (
+            f"sample_rate must be in (0, 1.0], got {sample_rate}"
+        )
+        self.sample_rate = sample_rate
+
+        self.block_metrics: dict[int, BlockMetricsState] = {}
+
+        self._eviction_events: list[KVCacheEvictionEvent] = []
+
+    def should_sample_block(self) -> bool:
+        return random.random() < self.sample_rate
+
+    def on_block_allocated(self, block: "KVCacheBlock") -> None:
+        if self.should_sample_block():
+            self.block_metrics[block.block_id] = BlockMetricsState()
+
+    def on_block_accessed(self, block: "KVCacheBlock") -> None:
+        metrics = self.block_metrics.get(block.block_id)
+        if metrics:
+            metrics.record_access()
+
+    def on_block_evicted(self, block: "KVCacheBlock") -> None:
+        metrics = self.block_metrics.pop(block.block_id, None)
+        if not metrics:
+            return
+
+        lifetime = metrics.get_lifetime_seconds()
+        idle_time = metrics.get_idle_time_seconds()
+        reuse_gaps = tuple(metrics.get_reuse_gaps_seconds())
+
+        self._eviction_events.append(
+            KVCacheEvictionEvent(
+                lifetime_seconds=lifetime,
+                idle_seconds=idle_time,
+                reuse_gaps_seconds=reuse_gaps,
+            )
+        )
+
+    def reset(self) -> None:
+        """Clear all state on cache reset."""
+        self.block_metrics.clear()
+        self._eviction_events.clear()
+
+    def drain_events(self) -> list[KVCacheEvictionEvent]:
+        events = self._eviction_events
+        self._eviction_events = []
+        return events
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index e3ec8440a932..cf8741c6f868 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -29,6 +29,7 @@
     compute_encoder_budget,
 )
 from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
+from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
 from vllm.v1.core.sched.interface import SchedulerInterface
 from vllm.v1.core.sched.output import (
     CachedRequestData,
@@ -40,7 +41,10 @@
 from vllm.v1.core.sched.utils import check_stop, remove_all
 from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
 from vllm.v1.kv_cache_interface import KVCacheConfig
-from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
+from vllm.v1.metrics.stats import (
+    PrefixCacheStats,
+    SchedulerStats,
+)
 from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
 from vllm.v1.request import Request, RequestStatus
 from vllm.v1.spec_decode.metrics import SpecDecodingStats
@@ -69,6 +73,12 @@ def __init__(
         self.kv_events_config = vllm_config.kv_events_config
         self.parallel_config = vllm_config.parallel_config
         self.log_stats = log_stats
+        self.observability_config = vllm_config.observability_config
+        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
+        if self.observability_config.kv_cache_metrics:
+            self.kv_metrics_collector = KVCacheMetricsCollector(
+                self.observability_config.kv_cache_metrics_sample,
+            )
         self.structured_output_manager = structured_output_manager
         self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder
 
@@ -187,6 +197,7 @@ def __init__(
             dcp_world_size=self.dcp_world_size,
             pcp_world_size=self.pcp_world_size,
             hash_block_size=self.block_size,
+            metrics_collector=self.kv_metrics_collector,
         )
         self.use_pp = self.parallel_config.pipeline_parallel_size > 1
         self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
@@ -1348,14 +1359,24 @@ def make_stats(
         prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
         assert prefix_cache_stats is not None
         connector_prefix_cache_stats = self._make_connector_prefix_cache_stats()
+        eviction_events = (
+            self.kv_metrics_collector.drain_events()
+            if self.kv_metrics_collector is not None
+            else []
+        )
+        spec_stats = spec_decoding_stats
+        connector_stats_payload = (
+            kv_connector_stats.data if kv_connector_stats else None
+        )
         return SchedulerStats(
             num_running_reqs=len(self.running),
             num_waiting_reqs=len(self.waiting),
             kv_cache_usage=self.kv_cache_manager.usage,
             prefix_cache_stats=prefix_cache_stats,
             connector_prefix_cache_stats=connector_prefix_cache_stats,
-            spec_decoding_stats=spec_decoding_stats,
-            kv_connector_stats=kv_connector_stats.data if kv_connector_stats else None,
+            kv_cache_eviction_events=eviction_events,
+            spec_decoding_stats=spec_stats,
+            kv_connector_stats=connector_stats_payload,
         )
 
     def make_spec_decoding_stats(
diff --git a/vllm/v1/metrics/loggers.py b/vllm/v1/metrics/loggers.py
index bd18a152ffc0..8010fe81d78d 100644
--- a/vllm/v1/metrics/loggers.py
+++ b/vllm/v1/metrics/loggers.py
@@ -375,6 +375,9 @@ def __init__(
         # Use this flag to hide metrics that were deprecated in
         # a previous release and which will be removed future
         self.show_hidden_metrics = vllm_config.observability_config.show_hidden_metrics
+        self.kv_cache_metrics_enabled = (
+            vllm_config.observability_config.kv_cache_metrics
+        )
 
         labelnames = ["model_name", "engine"]
         model_name = vllm_config.model_config.served_model_name
@@ -853,6 +856,79 @@ def __init__(
             histogram_decode_time_request, engine_indexes, model_name
         )
 
+        #
+        # KV Cache residency metrics
+        #
+        if self.kv_cache_metrics_enabled:
+            kv_cache_residency_buckets = [
+                0.001,
+                0.002,
+                0.005,
+                0.01,
+                0.02,
+                0.05,
+                0.1,
+                0.2,
+                0.5,
+                1,
+                2,
+                5,
+                10,
+                20,
+                30,
+                60,
+                120,
+                300,
+                600,
+                1200,
+                1800,
+            ]
+
+            histogram_kv_block_lifetime = self._histogram_cls(
+                name="vllm:kv_block_lifetime_seconds",
+                documentation=(
+                    "Histogram of KV cache block lifetime from allocation to eviction. "
+                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
+                ),
+                buckets=kv_cache_residency_buckets,
+                labelnames=labelnames,
+            )
+            self.histogram_kv_block_lifetime = make_per_engine(
+                histogram_kv_block_lifetime, engine_indexes, model_name
+            )
+
+            histogram_kv_block_idle_before_evict = self._histogram_cls(
+                name="vllm:kv_block_idle_before_evict_seconds",
+                documentation=(
+                    "Histogram of idle time before KV cache block eviction. "
+                    "Sampled metrics (controlled by --kv-cache-metrics-sample)."
+                ),
+                buckets=kv_cache_residency_buckets,
+                labelnames=labelnames,
+            )
+            self.histogram_kv_block_idle_before_evict = make_per_engine(
+                histogram_kv_block_idle_before_evict, engine_indexes, model_name
+            )
+
+            histogram_kv_block_reuse_gap = self._histogram_cls(
+                name="vllm:kv_block_reuse_gap_seconds",
+                documentation=(
+                    "Histogram of time gaps between consecutive KV cache block "
+                    "accesses. Only the most recent accesses are recorded "
+                    "(ring buffer). Sampled metrics (controlled by "
+                    "--kv-cache-metrics-sample)."
+                ),
+                buckets=kv_cache_residency_buckets,
+                labelnames=labelnames,
+            )
+            self.histogram_kv_block_reuse_gap = make_per_engine(
+                histogram_kv_block_reuse_gap, engine_indexes, model_name
+            )
+        else:
+            self.histogram_kv_block_lifetime = {}
+            self.histogram_kv_block_idle_before_evict = {}
+            self.histogram_kv_block_reuse_gap = {}
+
         #
         # LoRA metrics
         #
@@ -944,6 +1020,20 @@ def record(
                     scheduler_stats.kv_connector_stats, engine_idx
                 )
 
+            if (
+                self.kv_cache_metrics_enabled
+                and scheduler_stats.kv_cache_eviction_events
+            ):
+                lifetime_hist = self.histogram_kv_block_lifetime[engine_idx]
+                idle_hist = self.histogram_kv_block_idle_before_evict[engine_idx]
+                reuse_hist = self.histogram_kv_block_reuse_gap[engine_idx]
+
+                for event in scheduler_stats.kv_cache_eviction_events:
+                    lifetime_hist.observe(event.lifetime_seconds)
+                    idle_hist.observe(event.idle_seconds)
+                    for gap in event.reuse_gaps_seconds:
+                        reuse_hist.observe(gap)
+
             if self.gauge_lora_info is not None:
                 running_lora_adapters = ",".join(
                     scheduler_stats.running_lora_adapters.keys()
diff --git a/vllm/v1/metrics/stats.py b/vllm/v1/metrics/stats.py
index 4e9db98db0bc..a3078eaa75dc 100644
--- a/vllm/v1/metrics/stats.py
+++ b/vllm/v1/metrics/stats.py
@@ -150,6 +150,15 @@ class MultiModalCacheStats(BaseCacheStats):
     """
 
 
+@dataclass
+class KVCacheEvictionEvent:
+    """Single KV cache block eviction sample."""
+
+    lifetime_seconds: float
+    idle_seconds: float
+    reuse_gaps_seconds: tuple[float, ...]
+
+
 @dataclass
 class SchedulerStats:
     """Stats associated with the scheduler."""
@@ -166,6 +175,8 @@ class SchedulerStats:
     prefix_cache_stats: PrefixCacheStats = field(default_factory=PrefixCacheStats)
     connector_prefix_cache_stats: PrefixCacheStats | None = None
 
+    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(default_factory=list)
+
     spec_decoding_stats: SpecDecodingStats | None = None
     kv_connector_stats: dict[str, Any] | None = None
 
```
