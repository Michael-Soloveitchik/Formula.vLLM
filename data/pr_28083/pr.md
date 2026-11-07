# PR #28083: [PERF] Decouple projections from GDN custom op. Attempt 2

- field: gpu_kernels  | area: vllm/config
- author: vadiklyutiy
- state: MERGED  created: 2025-11-05T00:50:08Z  merged: 2025-11-06T01:01:12Z
- url: https://github.com/vllm-project/vllm/pull/28083
- labels: ready, qwen

## Body

## Purpose
The second attempt for #27512

This PR is refactoring of GDN. 

The main goal is to allow wider using of `torch.compile`.

1. Separated forward pass of GDN attention into three distinct pieces: Input Projection, Core Attention, Output Projection. Before projections was in the GDN custom op and were not covered by `torch.compile`.
2. Added `RMSNormGated` class that implements torch native gated rmsnorm and use it for GDN. `torch.compile` creates a good code for `RMSNormGated` even better than custom triton kernel used before.


## Functional Test Result
`lm_eval`
Before
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8059|±  |0.0109|
```
After
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8544|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

## Perf Test Result
Server
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_MOE_FP16=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --enable-expert-parallel --no-enable-prefix-caching --async-scheduling --max_cudagraph_capture_size=2048
```

**Prefill** 
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 8192 --random-output 1 --max-concurrency 512 --num-prompt 512 --ignore-eos
```
Before: Total Token throughput (tok/s):          104098.78
After:    Total Token throughput (tok/s):          105270.70
Speedup: **1.1%**

**Decode1**
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 512 --num-prompt 512 --ignore-eos
```
Before: Output token throughput (tok/s):         19212.17
After: Output token throughput (tok/s):         22384.37
Speedup: **16.5%**


**Decode2** 
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 1024 --num-prompt 1024 --ignore-eos
```
Before: Output token throughput (tok/s):         28821.37
After: Output token throughput (tok/s):         30298.90
Speed up: **5.1%**


**Decode3**
Server
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_MOE_FP16=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --enable-expert-parallel --no-enable-prefix-caching --async-scheduling
```
(without increasing `--max_cudagraph_capture_size`)
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 1024 --num-prompt 1024 --ignore-eos
```

Before: Output token throughput (tok/s):         16586.93
After: Output token throughput (tok/s):         18953.92
Speed up: **14.3%**






## Comments

### vadiklyutiy @ 2025-11-05T15:46:43Z

> idea looks good, please make sure tests are good too.

sure, I am working on it 
---
### ProExpertProg @ 2025-11-05T17:23:59Z

Please fix DCO, also what was the reason for the revert originally? 
---
### vadiklyutiy @ 2025-11-05T19:35:57Z

> what was the reason for the revert originally?

#27512 had a bit wired CI fails. They didn't look as related to my changes: one test failed on CUDAART init, another out of memory. I couldn’t reproduce them locally, even with a configuration very close to the CI environment. I asked for any idea that can help. Simon, was a bit too kind, took into account that CI is a bit unstable last several day, and merged the PR. But several next commit's CI also failed on the same tests and we decided to revert.  


---
### vadiklyutiy @ 2025-11-05T20:05:55Z

picking here and there I found the reason of CI fails. It isn't root cause but good enough IMO. 

This is logs with fails 
https://buildkite.com/vllm/ci/builds/37607/steps/canvas

after applying https://github.com/vllm-project/vllm/pull/28083/commits/e7a926050bd13013a5bb5f691ea582ce7b63da5b
fails disappeared
https://buildkite.com/vllm/ci/builds/37696/steps/canvas

https://github.com/vllm-project/vllm/pull/28083/commits/e7a926050bd13013a5bb5f691ea582ce7b63da5b moved `import` from global scope to function scope. By some reason it caused fails. Hope it is a acceptable workaround. 


---
### youkaichao @ 2025-11-06T02:30:14Z

> [e7a9260](https://github.com/vllm-project/vllm/commit/e7a926050bd13013a5bb5f691ea582ce7b63da5b) moved `import` from global scope to function scope. By some reason it caused fails. Hope it is a acceptable workaround.

likely the code unconditionaly initializes cuda runtime, which we need to be careful about.
---