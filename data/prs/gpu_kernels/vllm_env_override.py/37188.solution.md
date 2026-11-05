# PR #37188: [Performance] Enable Triton autotuning disk cache by default

- field: gpu_kernels  | area: vllm_env_override.py
- author: arpera
- state: MERGED  created: 2026-03-16T12:41:57Z  merged: 2026-03-19T21:36:28Z
- url: https://github.com/vllm-project/vllm/pull/37188
- labels: performance, ready

# Issue

## Problem

Triton's `@triton.autotune` decorator persists autotuning results to disk only when `TRITON_CACHE_AUTOTUNING` is enabled, but that env var defaults to disabled. As a result Triton ignores the on-disk autotuning cache and re-runs full kernel autotuning on every process start. Each restart re-sweeps all candidate configs (num_warps, num_ctas, num_stages) for every autotuned kernel before serving traffic.

## Example

On 8×B200 serving Qwen/Qwen3.5-397B-A17B-FP8 (dp=8, expert parallelism), restarting the server re-runs autotuning for kernels like `chunk_local_cumsum_scalar_kernel`, sweeping num_warps 1/2/4/8 each restart; the 2nd benchmark run took 121.49s vs 39.45s with caching enabled (req throughput 1.05 vs 3.24 req/s).

## Use case

Production vLLM serving where the server process is restarted (deploys, crashes, scaling, config changes) and the first inference requests after startup pay the autotuning cost.

## Resources

Kernel autotuning/compile time at startup — GPU time spent benchmarking candidate configs and CPU orchestration — wasted because the disk cache is ignored; the disk cache itself is underused.

## Type of users

Users serving large MoE / FP8-quantized models across multi-GPU setups (expert parallelism, data parallelism) on modern hardware (e.g. B200) where many Triton autotuned kernels are in the hot path.

## Kind of inputs

First inference request(s) after each server restart; the cost is per-process-start and independent of request content, though many autotuned kernels (FP8, MoE, cumsum) amplify it.

## Mechanism with the issue

Triton's autotune caching is gated by the `TRITON_CACHE_AUTOTUNING` environment variable; when unset it is treated as `False`, so autotuning results are neither written to nor read from `TRITON_CACHE_DIR`. Every new worker process therefore re-benchmarks all configs from scratch, adding large one-time latency to the first request after startup.

# Solution

## Fix in words

The diff adds `os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")` to `vllm/env_override.py`, right after the existing `torch._inductor.config.compile_threads = 1` setting. Before: the var was unset, defaulting Triton to disabled autotuning caching, so results were discarded each restart. After: autotuning results are persisted to `TRITON_CACHE_DIR` and reloaded on subsequent process starts, skipping re-autotuning. Because `setdefault` is used, users who explicitly set `TRITON_CACHE_AUTOTUNING=0` still opt out. Effect: 2nd-run benchmark duration dropped 121.49s→39.45s and token throughput rose 8,430→25,962 tok/s.

## Author's solution

Set `TRITON_CACHE_AUTOTUNING=1` via `os.environ.setdefault` so Triton persists autotuning results to `TRITON_CACHE_DIR` and reuses them across restarts, eliminating repeated autotuning latency on the first request. Using `setdefault` keeps it overridable, so users can opt out by explicitly setting `TRITON_CACHE_AUTOTUNING=0`.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/env_override.py`

### Diff
```diff
diff --git a/vllm/env_override.py b/vllm/env_override.py
index 181d000a68a7..5358568fc180 100644
--- a/vllm/env_override.py
+++ b/vllm/env_override.py
@@ -105,6 +105,14 @@ def _maybe_set_cuda_compatibility_path():
 # see https://github.com/vllm-project/vllm/issues/10619
 torch._inductor.config.compile_threads = 1
 
+# Enable Triton autotuning result caching to disk by default.
+# Without this, Triton re-runs autotuning on every process restart,
+# adding significant latency to the first inference request.
+# This writes autotuning results to TRITON_CACHE_DIR.
+# It can still be overridden by setting TRITON_CACHE_AUTOTUNING=0
+# in the environment.
+os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
+
 # ===================================================
 # torch 2.9 Inductor PythonWrapperCodegen monkeypatch
 # ===================================================

```
