# PR #30494: [Perf] Optimize deepgemm experts initialization, 3.9% TTFT improvement

- field: gpu_kernels  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-12-11T16:15:17Z  merged: 2025-12-11T22:28:34Z
- url: https://github.com/vllm-project/vllm/pull/30494
- labels: performance, ready, deepseek

## Body

## Purpose

Thanks for the context from @youkaichao !

Co-author: @li-jinpeng 

DeepGEMM uses negative values in m_indices (here expert_ids) to mark completely invalid / padded blocks that should be skipped. We always initialize expert_ids to -1 so any row that is not explicitly written by the scatter kernel will be treated as invalid and skipped by DeepGEMM's scheduler.

## Test

`export MODEL="Qwen/Qwen3-VL-235B-A22B-Thinking-FP8"`
`vllm serve $MODEL -tp 8 --port 9256  --enable-expert-parallel --enforce_eager`

### Perf

`vllm bench serve --model $MODEL  --dataset-name random --host 127.0.0.1  --random-input-len 4 --random-output-len 128 --request-rate inf --num-prompts 256 --port 9256`

```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Benchmark duration (s):                  46.09     
Total input tokens:                      1024      
Total generated tokens:                  32768     
Request throughput (req/s):              5.55      
Output token throughput (tok/s):         710.93    
Peak output token throughput (tok/s):    768.00    
Peak concurrent requests:                256.00    
Total token throughput (tok/s):          733.15    
---------------Time to First Token----------------
Mean TTFT (ms):                          736.08    
Median TTFT (ms):                        735.08    
P99 TTFT (ms):                           753.15    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          356.92    
Median TPOT (ms):                        356.90    
P99 TPOT (ms):                           356.95    
---------------Inter-token Latency----------------
Mean ITL (ms):                           356.92    
Median ITL (ms):                         339.90    
P99 ITL (ms):                            403.77    
==================================================

main
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Benchmark duration (s):                  46.35     
Total input tokens:                      1024      
Total generated tokens:                  32768     
Request throughput (req/s):              5.52      
Output token throughput (tok/s):         707.01    
Peak output token throughput (tok/s):    768.00    
Peak concurrent requests:                256.00    
Total token throughput (tok/s):          729.11    
---------------Time to First Token----------------
Mean TTFT (ms):                          765.44    
Median TTFT (ms):                        764.74    
P99 TTFT (ms):                           781.15    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          358.70    
Median TPOT (ms):                        358.69    
P99 TPOT (ms):                           358.73    
---------------Inter-token Latency----------------
Mean ITL (ms):                           358.70    
Median ITL (ms):                         341.41    
P99 ITL (ms):                            401.74    
==================================================
```

### Acc

`lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=$MODEL,num_concurrent=1024" --tasks gsm8k `

```bash
# now
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8681|±  |0.0093|
|     |       |strict-match    |     5|exact_match|↑  |0.8362|±  |0.0102|
# main
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8704|±  |0.0093|
|     |       |strict-match    |     5|exact_match|↑  |0.8385|±  |0.0101|
```

## Comments

### yewentao256 @ 2025-12-11T17:31:01Z

> thanks for the fix! maybe we should add @li-jinpeng as a co-author of the PR?

Definitely, is there any way we could add him?
---
### li-jinpeng @ 2025-12-11T18:07:23Z

👍
---