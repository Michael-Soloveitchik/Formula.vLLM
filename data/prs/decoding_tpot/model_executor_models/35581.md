# PR #35581: Fix Qwen3_5MTP packed_modules_mapping for gate_up_proj

- field: decoding  | area: model_executor/models
- author: cwazai
- state: MERGED  created: 2026-02-28T04:00:53Z  merged: 2026-02-28T14:50:55Z
- url: https://github.com/vllm-project/vllm/pull/35581
- labels: ready, qwen

## Body

### What does this PR do?
Fixes the `packed_modules_mapping` in `Qwen3_5MTP` class where `gate_up_proj` incorrectly included `down_proj` instead of `gate_proj`.

### Why is this important?
- The weight loading logic in `Qwen3_5MultiTokenPredictor.load_weights()` expects `gate_up_proj` to pack `gate_proj` and `up_proj` (see stacked_params_mapping).
- The incorrect mapping prevented fused/packed kernel optimizations for the MLP layer, causing potential performance regression.
- This change aligns the mapping with the actual model architecture and enables proper fused computation paths.

### Changes Made
1. Fixed `packed_modules_mapping['gate_up_proj']` from `["up_proj", "down_proj"]` to `["gate_proj", "up_proj"]`


### Performance Impact
This fix enables the intended fused MLP kernel path, which can improve throughput by 2-8% depending on batch size and sequence length for Qwen3.5 MTP models.
## Benchmarks
Hardware: Nvidia H800, Software: CUDA/driver：13.0
Workload: model：Qwen3.5-35B-A3B

[before] 
``` python
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             100       
Benchmark duration (s):                  65.25     
Total input tokens:                      1013750   
Total generated tokens:                  100000    
Request throughput (req/s):              15.33     
Output token throughput (tok/s):         1532.67   
Peak output token throughput (tok/s):    1108.00   
Peak concurrent requests:                141.00    
Total token throughput (tok/s):          17070.14  
---------------Time to First Token----------------
Mean TTFT (ms):                          3805.74   
Median TTFT (ms):                        2806.38   
P99 TTFT (ms):                           15286.70  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.20     
Median TPOT (ms):                        26.45     
P99 TPOT (ms):                           36.02     
---------------Inter-token Latency----------------
Mean ITL (ms):                           83.95     
Median ITL (ms):                         107.89    
P99 ITL (ms):                            171.96    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          6400.00   
Median E2EL (ms):                        5407.63   
P99 E2EL (ms):                           17945.71  
---------------Speculative Decoding---------------
Acceptance rate (%):                     59.00     
Acceptance length:                       3.36      
Drafts:                                  29875     
Draft tokens:                            119500    
Accepted tokens:                         70504     
Per-position acceptance (%):
  Position 0:                            82.11     
  Position 1:                            64.77     
  Position 2:                            49.54     
  Position 3:                            39.57     
==================================================
```
[after]

```python
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             100       
Benchmark duration (s):                  61.45     
Total input tokens:                      1013750   
Total generated tokens:                  100000    
Request throughput (req/s):              16.27     
Output token throughput (tok/s):         1627.25   
Peak output token throughput (tok/s):    1213.00   
Peak concurrent requests:                141.00    
Total token throughput (tok/s):          18123.49  
---------------Time to First Token----------------
Mean TTFT (ms):                          3577.34   
Median TTFT (ms):                        2734.66   
P99 TTFT (ms):                           13903.08  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          24.75     
Median TPOT (ms):                        25.12     
P99 TPOT (ms):                           32.56     
---------------Inter-token Latency----------------
Mean ITL (ms):                           78.90     
Median ITL (ms):                         106.61    
P99 ITL (ms):                            147.59    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          6027.40   
Median E2EL (ms):                        5228.75   
P99 E2EL (ms):                           16151.06  
---------------Speculative Decoding---------------
Acceptance rate (%):                     58.61     
Acceptance length:                       3.34      
Drafts:                                  30026     
Draft tokens:                            120104    
Accepted tokens:                         70395     
Per-position acceptance (%):
  Position 0:                            81.71     
  Position 1:                            64.13     
  Position 2:                            49.03     
  Position 3:                            39.58     
==================================================
```
####Accuracy/Precision Test Results
| DataSet                 | \[Feature] Score | Baseline Score | 
| -------------------- | ---------------- | -------------- | 
|C-Eval      |       0.9064     |  0.9049          | 
| mm_bench  |     0.9194    |  0.9168          | 




## Comments

(no discussion comments)