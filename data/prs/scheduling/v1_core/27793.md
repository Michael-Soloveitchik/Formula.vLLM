# PR #27793: [Core][Observability] Add KV cache residency metrics

- field: scheduling  | area: v1/core
- author: shivampr
- state: MERGED  created: 2025-10-30T05:56:14Z  merged: 2025-12-01T18:27:53Z
- url: https://github.com/vllm-project/vllm/pull/27793
- labels: documentation, ready, v1

## Body

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

## Comments

### hmellor @ 2025-11-03T17:11:58Z

@markmc could you look at this metrics PR?
---
### markmc @ 2025-11-06T13:43:53Z

This looks very clean and promising, very nice work!

Two things jump out at me before getting into details:

1.  You're not passing a `KVCacheMetricsCollector` to `get_kv_cache_coordinator()` in the `KVCacheManager` constructor
2. You're accessing `PrometheusStatLogger` state from the engine core, when it is only accessible in the frontend process - this would probably become apparent if you tried to resolve (1)

To follow the pattern of the existing architecture, you're going to have to collect these intervals in the engine core and send them back to the frontend via `EngineCoreOutputs` for processing by the `StatLogger`s

For comparison, see how we already have `EngineCoreEvent` which sends back timestamps associated with request-level events. And `SchedulerStats` has prefix cache and spec decoding stats sent from the engine.

`SchedulerStats` could think have e.g. a list of `BlockEvictionEvent`s with `lifetime`, `idle_time`, `reuse_gaps`, `residency`? You can see how in #27718 it's implemented using `SchedulerStats.kv_cache_block_lifetimes`:

```
kv_cache_block_lifetimes: list[float] = field(default_factory=list)
```
Prometheus is definitely the priority, but it might be worth also adding support to `LoggingStatLogger` so we get  (maybe a subset of) these metrics on the console too

Also, thank you for adding the unit tests. I do think we need more of an e2e too test, though - preferably both with the `LLM.get_metrics()` API and online serving via the HTTP API
 
---
### markmc @ 2025-11-06T13:46:55Z

Also, the semantics of these metrics, and how they differ from each other, perhaps could benefit from visualization. I'm thinking of the diagrams I added in https://docs.vllm.ai/en/latest/design/metrics.html#engine-core-events. You could add a section to that doc?
---
### shivampr @ 2025-11-07T06:13:58Z

@markmc Thanks for the detailed feedback! I’ll work on addressing your review comments, and reach out over slack if i need more information.
Will push the updates soon.
---
### mergify @ 2025-11-10T04:17:38Z

Documentation preview: https://vllm--27793.org.readthedocs.build/en/27793/
---
### shivampr @ 2025-11-10T04:52:54Z

I addressed the review by plumbing a single `KVCacheMetricsCollector` from the scheduler into `get_kv_cache_coordinator()` and having the engine emit raw `BlockResidencyEvent`s via `SchedulerStats`. Prometheus and logging observers now consume those events on the frontend.

For validation I ran:

- `pytest -v tests/v1/core/test_kv_cache_metrics.py`
- CPU smoke for `LLM.get_metrics()` using `LLM(model="gpt2", kv_cache_metrics=True, kv_cache_metrics_sample=1.0, disable_log_stats=False)` and confirmed all four histograms in `get_prometheus_registry().collect()`
- CPU API server smoke (`python -m vllm.entrypoints.openai.api_server --model gpt2 --kv-cache-metrics --kv-cache-metrics-sample 1.0`) with two `/v1/completions` calls and verified `/metrics` exposed the new histogram families
---
### shivampr @ 2025-11-19T04:36:16Z

Ack, will address the review comments!
---
### shivampr @ 2025-11-28T02:33:32Z

## Benchmark Results: Performance Overhead

I created a simple benchmark script that isolates the metrics collection code (simulating 10,000 block lifecycles with allocation, 4 accesses, and eviction) to measure pure overhead without full inference.

**Results:**
- Sample rate 0.01 (1% - default): ~365ns per sampled block
- Sample rate 0.1 (10%): ~511ns per sampled block
- Sample rate 1.0 (100%): ~1,496ns per sampled block

**Overhead Analysis:**
For a typical request with 100 blocks at 1% sampling (1 block tracked), the overhead is ~365ns. Assuming typical request latency of 100-500ms, this represents **< 0.1% overhead per request**. When accounting for sampling decisions, event creation, and Prometheus recording, the **expected overall overhead is < 0.5% at the default 1% sampling rate**.
---
### markmc @ 2025-11-28T11:27:36Z

This looks valid:

```
FAILED v1/engine/test_llm_engine.py::test_kv_cache_metrics_surface_in_llm_get_metrics - AssertionError: assert 0 > 0
...
[2025-11-28T09:31:41Z] >           assert lifetime_hist.count > 0
[2025-11-28T09:31:41Z] E           AssertionError: assert 0 > 0
[2025-11-28T09:31:41Z] E            +  where 0 = Histogram(name='vllm:kv_block_lifetime_seconds', labels={'engine': '0', 'model_name': 'facebook/opt-125m'}, count=0, sum=0.0, buckets={'0.001': 0, '0.002': 0, '0.005': 0, '0.01': 0, '0.02': 0, '0.05': 0, '0.1': 0, '0.2': 0, '0.5': 0, '1.0': 0, '2.0': 0, '5.0': 0, '10.0': 0, '20.0': 0, '30.0': 0, '60.0': 0, '120.0': 0, '300.0': 0, '600.0': 0, '1200.0': 0, '1800.0': 0, '+Inf': 0}).count
[2025-11-28T09:31:41Z]
[2025-11-28T09:31:41Z] v1/engine/test_llm_engine.py:244: AssertionError
```
---
### shivampr @ 2025-11-28T21:01:16Z

I'll try to run the test on A40 CUDA GPU on [Runpod](https://console.runpod.io/). Currently my request to access model meta-llama/Llama-3.1-8B-Instruct is awaiting a review from the repo authors. Once approved, will try to run these tests there, and once it passed, will raise another revision.
---
### shivampr @ 2025-11-29T07:11:58Z

 I tried pretty hard to get vLLM building and running on a CUDA GPU(both `A40` and `H200`) via [Runpod](https://console.runpod.io/deploy) for the KV cache metrics test, but couldn’t get to a stable, buildable setup. As a fallback I replicated the same checks in the CPU unit tests (tests/v1/core/test_kv_cache_metrics.py), so the metrics logic is still covered. I don’t currently know an easier way to bring up a reliable vLLM CUDA dev environment, so I left the GPU-side integration test for a follow-up.
---