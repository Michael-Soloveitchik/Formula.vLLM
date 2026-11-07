# PR #26534: Move query quantization to attention layer for Flashinfer & Triton.

- field: gpu_kernels  | area: v1/attention
- author: adabeyta
- state: MERGED  created: 2025-10-09T21:10:40Z  merged: 2025-10-15T23:01:38Z
- url: https://github.com/vllm-project/vllm/pull/26534
- labels: ready, v1

## Body

### Purpose
Implements refactor of quantization to the attention layer for triton and flashinfer, resolves feature request [#25584](https://github.com/vllm-project/vllm/issues/25584)

### Test Plan
Spin up server:

Flashinfer:
```
VLLM_ATTENTION_BACKEND=FLASHINFER  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --no-enable-prefix-caching
```
  
Triton:
```
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --no-enable-prefix-caching
```

Benchmark:
```
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct  \
    --dataset-name sonnet \
    --dataset-path vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 1000 \
    --sonnet-output-len 200 \
    --port 8000 \
    --num-prompts 20 \
    --max-concurrency 1
```
### Accuracy
To ensure there is no accidental accuracy degradation we also run the following for Flashinfer & Triton with kv_cache_dtype in {auto,fp8} both on this PR and on mainline. We also run without enforce_eager=True for the FP8 variants
```
lm_eval \
  --model vllm \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,kv_cache_dtype=auto,tensor_parallel_size=1,enforce_eager=True \
  --tasks gsm8k \
  --batch_size 
```

### Test Results
```
PR + FlashInfer
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  16.81     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.19      
Output token throughput (tok/s):         237.97    
Peak output token throughput (tok/s):    240.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1323.60   
---------------Time to First Token----------------
Mean TTFT (ms):                          20.13     
Median TTFT (ms):                        19.97     
P99 TTFT (ms):                           23.20     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.12      
Median TPOT (ms):                        4.13      
P99 TPOT (ms):                           4.15      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.12      
Median ITL (ms):                         4.12      
P99 ITL (ms):                            4.61      
==================================================

PR + Triton
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  17.23     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.16      
Output token throughput (tok/s):         232.19    
Peak output token throughput (tok/s):    235.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1291.45   
---------------Time to First Token----------------
Mean TTFT (ms):                          24.86     
Median TTFT (ms):                        24.77     
P99 TTFT (ms):                           31.37     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.20      
Median TPOT (ms):                        4.20      
P99 TPOT (ms):                           4.24      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.20      
Median ITL (ms):                         4.20      
P99 ITL (ms):                            4.58      
==================================================

Main + FlashInfer
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  18.15     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.10      
Output token throughput (tok/s):         220.42    
Peak output token throughput (tok/s):    223.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1225.98   
---------------Time to First Token----------------
Mean TTFT (ms):                          24.18     
Median TTFT (ms):                        24.08     
P99 TTFT (ms):                           29.89     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.44      
Median TPOT (ms):                        4.44      
P99 TPOT (ms):                           4.46      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.44      
Median ITL (ms):                         4.43      
P99 ITL (ms):                            5.01      
==================================================

Main + Triton
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  17.42     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.15      
Output token throughput (tok/s):         229.59    
Peak output token throughput (tok/s):    234.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1276.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          27.61     
Median TTFT (ms):                        26.97     
P99 TTFT (ms):                           37.66     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.24      
Median TPOT (ms):                        4.24      
P99 TPOT (ms):                           4.32      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.24      
Median ITL (ms):                         4.23      
P99 ITL (ms):                            4.94      
==================================================
```

### Accuracy on GSM8k
This PR shows no accuracy variation compared to mainline vllm.
```
PR + Triton + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7817|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|

PR + FlashInfer + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7801|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7551|±  |0.0118|


Main + FlashInfer + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7854|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7635|±  |0.0118|


Main + Triton + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7817|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|


PR + Triton + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7771|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7521|±  |0.0119|

PR + FlashInfer + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7665|±  |0.0117|
|     |       |strict-match    |     5|exact_match|↑  |0.7483|±  |0.0120|

Main + FlashInfer + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7665|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7483|±  |0.0118|


Main + Triton + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7771|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7521|±  |0.0118|


PR + Triton + fp8 (w/ compile)
Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7695|±  |0.0116|
|     |       |strict-match    |     5|exact_match|↑  |0.7460|±  |0.0120|


PR + FlashInfer + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7756|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|

Main + FlashInfer + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7680|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7437|±  |0.0118|


Main + Triton + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7672|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7468|±  |0.0118| 
```

## Comments

### mergify @ 2025-10-09T21:11:20Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @adabeyta.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-10-09T22:02:44Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @adabeyta.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### pavanimajety @ 2025-10-14T20:50:52Z

@adabeyta Any analysis on why we are seeing lower toks/sec with enhanced fusion? Even without a custom kernel, the fact that rope + Quant can be jitted to a triton kernel should give us slightly higher perf, correct? 
---
### ProExpertProg @ 2025-10-15T00:24:06Z

@adabeyta test failure looks related, it's possible this change breaks the fusion test. Could you run locally to check? Also worth running a model E2E to make sure fusion happens E2E (e2e tests coming soon in #24604)
---
### adabeyta @ 2025-10-15T23:20:16Z

> @adabeyta Any analysis on why we are seeing lower toks/sec with enhanced fusion? Even without a custom kernel, the fact that rope + Quant can be jitted to a triton kernel should give us slightly higher perf, correct?

@pavanimajety Updated with new perf numbers. We're seeing better
  performance across both Triton and FlashInfer backends (up to 8% throughput
  improvements). The earlier regression was from an intermediate commit
  before the gating logic was added.
---
### pavanimajety @ 2025-10-16T00:39:01Z

Great work, thanks for the update! 
---