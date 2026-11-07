# PR #39822: [Hybrid] Warmup Mamba2 SSD kernel

- field: gpu_kernels  | area: model_executor/layers
- author: tdoublep
- state: MERGED  created: 2026-04-14T19:12:41Z  merged: 2026-05-12T12:46:23Z
- url: https://github.com/vllm-project/vllm/pull/39822
- labels: ready

## Body

## Summary

Triton's auto-tuner for the Mamba2 SSD kernels currently runs lazily on the first inference request, causing a large latency spike. This PR adds a `_warmup_ssd_kernels()` method to `MambaMixer2` that triggers auto-tuning during vLLM's profile phase (before SSM cache allocation), shifting the cost into server startup.

- Runs a minimal `mamba_chunk_scan_combined_varlen` forward pass with dummy tensors during the V1 profile run
- Covers both `HAS_INITSTATES` constexpr code paths (with and without `initial_states`)
- Uses correct dtypes (including `ssm_state_dtype`) to match Triton's cache keys
- Each layer warms up once, and subsequent layers hit the Triton cache

## Benchmark Results

Measured on a single H100 80GB with `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (Mamba2 hybrid model), `max_model_len=512`, cold Triton cache (`TRITON_PRINT_AUTOTUNING=1` confirmed autotuning location):

| Metric | Baseline (main) | With Warmup | Change |
|--------|-----------------|-------------|--------|
| Model load time | 30.0s | 76.9s | +46.9s (autotuning shifted here) |
| First request latency | 31.343s | 2.890s | **-28.5s (91% reduction)** |
| Subsequent request avg | 0.083s | 0.083s | No change |
| First / subsequent ratio | 378x | 35x | **10.8x improvement** |

With `TRITON_PRINT_AUTOTUNING=1`, 58 kernel autotuning events occurred **after** model load on main (during the first request). With the warmup branch, **zero** autotuning events occurred after model load -- all SSD kernel autotuning completed during initialization.

The remaining ~2.9s first-request overhead (vs 0.08s subsequent) is clearly not from auto-tuning of any kernels, since `TRITON_PRINT_AUTOTUNING=1` produces no output during the first request. The current suspicion is that this residual cost comes from Triton JIT compilation (as opposed to auto-tuning), but this requires further investigation as a follow-up.

<details>
<summary>Benchmark script</summary>

```python
#!/usr/bin/env python3
"""Measure first-request vs subsequent-request latency for a Mamba2 model."""
import os
import shutil
import time

for cache_dir in [
    os.path.expanduser("~/.triton/cache"),
    os.path.expanduser("~/.cache/vllm/torch_compile_cache"),
]:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cleared {cache_dir}")

from vllm import LLM, SamplingParams

NUM_REQUESTS = 5
PROMPT = "The capital of France is"

print("Loading model...")
t0 = time.perf_counter()
llm = LLM(model="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
           max_model_len=512, trust_remote_code=True)
load_time = time.perf_counter() - t0
print(f"Model loaded in {load_time:.1f}s\n")

sampling_params = SamplingParams(temperature=0.0, max_tokens=20)

results = []
for i in range(NUM_REQUESTS):
    t_start = time.perf_counter()
    llm.generate([PROMPT], sampling_params)
    latency = time.perf_counter() - t_start
    results.append(latency)
    print(f"  Request {i+1}: {latency:.3f}s")

first = results[0]
rest_avg = sum(results[1:]) / len(results[1:])
print(f"\n  Model load time:    {load_time:.1f}s")
print(f"  First request:      {first:.3f}s")
print(f"  Subsequent avg:     {rest_avg:.3f}s")
print(f"  Ratio (1st / avg):  {first / rest_avg:.2f}x")
```

</details>

## Test plan

- [x] Verified autotuning moves from first request to model init via `TRITON_PRINT_AUTOTUNING=1`
- [x] Confirmed first-request latency drops from 31.3s to 2.9s
- [x] Confirmed subsequent request latency is unchanged
- [ ] Run existing Mamba2 model tests


## Comments

### tdoublep @ 2026-05-11T18:07:32Z

@tomeras91 Thanks for the review - I think I have addressed all feedback, please TAL. 
---