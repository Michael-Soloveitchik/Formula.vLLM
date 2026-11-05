# PR #25105: [PERF] Add `conv1d` metadata to GDN attn

- field: attention  | area: v1/attention
- author: vadiklyutiy
- state: MERGED  created: 2025-09-17T22:15:45Z  merged: 2025-09-18T14:27:50Z
- url: https://github.com/vllm-project/vllm/pull/25105
- labels: ready, v1, qwen

## Body

## Purpose
The on of the main gap in performance for Qwen3-next model is low utilisation of GPU in GDN attn. 
Low utilization caused by several GPU<->host memory transfers. 
These transfers caused by `causal_conv1d_fn`. 
Qwen3-next haven't supported this specific convolution metadata. The purpose of the metadata is avoid these memory transfer.

Add conv metadata (similar to mamba2). 

\+ corrected `tensor.tensor` -> `tensor.Tensor` types annotation.

## Test Result
H200, tp=4
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct   --endpoint /v1/completions --dataset-name random --random-input 32768   --random-output 1 --max-concurrency 8 --num-prompt 8
```

Before 
```
============ Serving Benchmark Result ============
Successful requests:                     8
Maximum request concurrency:             8
Benchmark duration (s):                  4.56
Total input tokens:                      262144
Total generated tokens:                  8
Request throughput (req/s):              1.76
Output token throughput (tok/s):         1.76
Total Token throughput (tok/s):          57510.95
---------------Time to First Token----------------
Mean TTFT (ms):                          2600.46
Median TTFT (ms):                        2601.40
P99 TTFT (ms):                           4511.95
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00
Median TPOT (ms):                        0.00
P99 TPOT (ms):                           0.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00
Median ITL (ms):                         0.00
P99 ITL (ms):                            0.00
==================================================
```

After
```
============ Serving Benchmark Result ============
Successful requests:                     8
Maximum request concurrency:             8
Benchmark duration (s):                  3.61
Total input tokens:                      262144
Total generated tokens:                  8
Request throughput (req/s):              2.22
Output token throughput (tok/s):         2.22
Total Token throughput (tok/s):          72623.02
---------------Time to First Token----------------
Mean TTFT (ms):                          2067.73
Median TTFT (ms):                        2068.62
P99 TTFT (ms):                           3570.84
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00
Median TPOT (ms):                        0.00
P99 TPOT (ms):                           0.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00
Median ITL (ms):                         0.00
P99 ITL (ms):                            0.00
==================================================
```

**Speedup is 26%**


## Comments

### vadiklyutiy @ 2025-09-18T08:05:45Z

Could someone please run the CI?

---