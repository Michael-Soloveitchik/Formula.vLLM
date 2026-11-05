# PR #29217: [Model][Qwen3VL] Tune Triton w8a8 block fp8 kernel for L40s

- field: quantization  | area: model_executor/layers
- author: lgeiger
- state: MERGED  created: 2025-11-22T01:24:02Z  merged: 2025-11-22T01:30:55Z
- url: https://github.com/vllm-project/vllm/pull/29217
- labels: qwen

## Body

## Purpose

This tunes the triton w8a8 block fp8 kernels on a single L40s GPU with the shapes used in `Qwen3-VL-32B-Instruct-FP8` which improves throughput by ~11.4% in the ShareGPT text-only benchmark.

## Test Plan

```
vllm serve Qwen/Qwen3-VL-32B-Instruct-FP8 --limit-mm-per-prompt.video 0 --limit-mm-per-prompt.image 0 --max-model-len 24000 --no-enable-prefix-caching

vllm bench serve --backend vllm --model Qwen/Qwen3-VL-32B-Instruct-FP8 --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000
```

## Test Result

**Before:**

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Failed requests:                         0
Benchmark duration (s):                  305.45
Total input tokens:                      217393
Total generated tokens:                  189963
Request throughput (req/s):              3.27
Output token throughput (tok/s):         621.91
Peak output token throughput (tok/s):    1514.00
Peak concurrent requests:                1000.00
Total Token throughput (tok/s):          1333.62
---------------Time to First Token----------------
Mean TTFT (ms):                          124986.70
Median TTFT (ms):                        123096.83
P99 TTFT (ms):                           257832.38
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          130.71
Median TPOT (ms):                        112.19
P99 TPOT (ms):                           554.13
---------------Inter-token Latency----------------
Mean ITL (ms):                           109.80
Median ITL (ms):                         73.75
P99 ITL (ms):                            523.20
==================================================
```

**After:**

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Failed requests:                         0
Benchmark duration (s):                  274.13
Total input tokens:                      217393
Total generated tokens:                  189963
Request throughput (req/s):              3.65
Output token throughput (tok/s):         692.98
Peak output token throughput (tok/s):    1439.00
Peak concurrent requests:                1000.00
Total Token throughput (tok/s):          1486.02
---------------Time to First Token----------------
Mean TTFT (ms):                          118620.50
Median TTFT (ms):                        121301.58
P99 TTFT (ms):                           239239.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          130.03
Median TPOT (ms):                        103.26
P99 TPOT (ms):                           531.54
---------------Inter-token Latency----------------
Mean ITL (ms):                           102.96
Median ITL (ms):                         65.82
P99 ITL (ms):                            489.05
==================================================
```


## Comments

(no discussion comments)