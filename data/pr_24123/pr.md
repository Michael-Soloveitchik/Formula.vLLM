# PR #24123: [Compilation][WideEP] Enable Piecewise CUDAGraph for DeepEPHT

- field: scheduling  | area: vllm/config
- author: yewentao256
- state: MERGED  created: 2025-09-02T20:36:10Z  merged: 2025-09-09T14:21:11Z
- url: https://github.com/vllm-project/vllm/pull/24123
- labels: performance, ready, deepseek

## Body

## Purpose

Enable Piecewise Cuda Graph for DeepEP High Throughput

## Test

`vllm serve --model="deepseek-ai/DeepSeek-R1" --max-num-seqs 512 --trust-remote-code --data-parallel-size 8 --enable-expert-parallel --gpu-memory-utilization 0.9 --port 9256 --disable-log-requests --no-enable-prefix-caching -O '{"full_cuda_graph": true}' --cuda-graph-sizes 16 32 48 64 96 128 160 192 256 512`

### Acc

`lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=deepseek-ai/DeepSeek-R1,num_concurrent=256" --tasks gsm8k`

```bash
# Now
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.95|±  | 0.006|
|     |       |strict-match    |     5|exact_match|↑  | 0.95|±  | 0.006|
# Main

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9530|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.9515|±  |0.0059|
```

### Performance

**Around 33% E2E improvement for decode**

Decode:
`vllm bench serve --model deepseek-ai/DeepSeek-R1 --dataset-name random  --host 127.0.0.1 --port 9256 --random-input-len 1 --random-output-len 200 --request-rate inf --num-prompts 1024`

```bash
# Now
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  18.93     
Total input tokens:                      0         
Total generated tokens:                  204800    
Request throughput (req/s):              54.10     
Output token throughput (tok/s):         10819.65  
Total Token throughput (tok/s):          10819.65  
---------------Time to First Token----------------
Mean TTFT (ms):                          960.11    
Median TTFT (ms):                        975.87    
P99 TTFT (ms):                           1053.79   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          89.55     
Median TPOT (ms):                        89.51     
P99 TPOT (ms):                           89.73     
---------------Inter-token Latency----------------
Mean ITL (ms):                           89.56     
Median ITL (ms):                         89.47     
P99 ITL (ms):                            125.84    
==================================================

# Main
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  25.18     
Total input tokens:                      0         
Total generated tokens:                  204800    
Request throughput (req/s):              40.67     
Output token throughput (tok/s):         8133.06   
Total Token throughput (tok/s):          8133.06   
---------------Time to First Token----------------
Mean TTFT (ms):                          886.26    
Median TTFT (ms):                        918.52    
P99 TTFT (ms):                           945.95    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          121.50    
Median TPOT (ms):                        121.53    
P99 TPOT (ms):                           121.60    
---------------Inter-token Latency----------------
Mean ITL (ms):                           121.50    
Median ITL (ms):                         120.95    
P99 ITL (ms):                            157.64    
==================================================
```

For prefill seems the same.

Prefill:
`vllm bench serve --model deepseek-ai/DeepSeek-R1 --dataset-name random  --host 127.0.0.1 --port 9256 --random-input-len 200 --random-output-len 1 --request-rate inf --num-prompts 1024`

```bash
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  4.61      
Total input tokens:                      203537    
Total generated tokens:                  1024      
Request throughput (req/s):              221.93    
Output token throughput (tok/s):         221.93    
Total Token throughput (tok/s):          44334.43  
---------------Time to First Token----------------
Mean TTFT (ms):                          3226.53   
Median TTFT (ms):                        3161.64   
P99 TTFT (ms):                           4519.03   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
# main
============ Serving Benchmark Result ============
Successful requests:                     1024      
Benchmark duration (s):                  4.64      
Total input tokens:                      203537    
Total generated tokens:                  1024      
Request throughput (req/s):              220.54    
Output token throughput (tok/s):         220.54    
Total Token throughput (tok/s):          44056.42  
---------------Time to First Token----------------
Mean TTFT (ms):                          3143.05   
Median TTFT (ms):                        3450.70   
P99 TTFT (ms):                           4565.75   
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

## Comments

### ProExpertProg @ 2025-09-02T23:15:41Z

Also what's the performance for a prefill-only node? I assume some of the speedup is coming from speeding up decode?
---
### yewentao256 @ 2025-09-03T00:38:11Z

> Also what's the performance for a prefill-only node? I assume some of the speedup is coming from speeding up decode?

@ProExpertProg For prefill only, seems quite similar, added the comparison in the PR description
---
### yewentao256 @ 2025-09-04T17:50:32Z

@ProExpertProg CC
---
### yewentao256 @ 2025-09-09T14:18:22Z

@ProExpertProg @mgoin All unit tests pass, wondering if we could land this PR then?
---