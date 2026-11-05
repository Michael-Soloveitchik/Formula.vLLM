# PR #28164: [Feature] Enable TP + EP `shared_experts` overlap with router, 3.7% E2E performance improvement

- field: parallelism  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-11-05T21:57:47Z  merged: 2025-11-06T01:21:09Z
- url: https://github.com/vllm-project/vllm/pull/28164
- labels: ready

## Body

## Purpose

A follow up PR for https://github.com/vllm-project/vllm/pull/26440

This PR we enable TP + EP to overlap shared experts with router, and get performance in small batch size.
We get 3.7% E2E performance improvement and 24% faster for time to first token.

<img width="3062" height="496" alt="image" src="https://github.com/user-attachments/assets/da9e6a50-8881-4b38-8074-98f3caea77df" />

Follow up TODOs:
- Optimization through reuse of stream
- The performance may be not that good in larger batch size in local node, we can sweep through the batch size and find the perf point and optimize it.

## Test

On B200

`vllm serve deepseek-ai/DeepSeek-R1 -tp 8  --port 9256 --enable-expert-parallel `

### Acc

```bash
lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=deepseek-ai/DeepSeek-R1,num_concurrent=1024" --tasks gsm8k
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9515|±  |0.0059|
|     |       |strict-match    |     5|exact_match|↑  |0.9515|±  |0.0059|
```

### Perf

`vllm bench serve --model deepseek-ai/DeepSeek-R1  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 4 --random-output-len 1024 --request-rate inf --num-prompts 160 --max-concurrency 32`

With overlap

```bash
============ Serving Benchmark Result ============
Successful requests:                     160       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  171.55    
Total input tokens:                      480       
Total generated tokens:                  123383    
Request throughput (req/s):              0.93      
Output token throughput (tok/s):         719.24    
Peak output token throughput (tok/s):    832.00    
Peak concurrent requests:                49.00     
Total Token throughput (tok/s):          722.04    
---------------Time to First Token----------------
Mean TTFT (ms):                          168.90    
Median TTFT (ms):                        87.14     
P99 TTFT (ms):                           420.81    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.90     
Median TPOT (ms):                        40.11     
P99 TPOT (ms):                           40.23     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.90     
Median ITL (ms):                         39.90     
P99 ITL (ms):                            46.70     
==================================================
```

Without (main):

```bash
============ Serving Benchmark Result ============
Successful requests:                     160       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  166.48    
Total input tokens:                      480       
Total generated tokens:                  124313    
Request throughput (req/s):              0.96      
Output token throughput (tok/s):         746.73    
Peak output token throughput (tok/s):    864.00    
Peak concurrent requests:                50.00     
Total Token throughput (tok/s):          749.62    
---------------Time to First Token----------------
Mean TTFT (ms):                          135.40    
Median TTFT (ms):                        83.12     
P99 TTFT (ms):                           286.72    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.42     
Median TPOT (ms):                        38.61     
P99 TPOT (ms):                           38.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           38.43     
Median ITL (ms):                         38.43     
P99 ITL (ms):                            44.55     
==================================================
```

## Comments

(no discussion comments)