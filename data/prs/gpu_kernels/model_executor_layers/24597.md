# PR #24597: Kimi K2 Fused MoE kernels Optimization configs

- field: quantization  | area: model_executor/layers
- author: samanamp
- state: MERGED  created: 2025-09-10T16:19:25Z  merged: 2025-09-11T06:06:16Z
- url: https://github.com/vllm-project/vllm/pull/24597
- labels: performance, ready

## Body

## Purpose
Optimize Kimi K2 Fused MoE kernels Optimization configs

Execution improvement: We see 17.8% in `Output token throughput (tok/s)`. The bulk of improvement is in `Inter-token Latency`.

## Test Plan
1. Single benchmark
```
python benchmarks/kernels/benchmark_moe_old.py --model $MODEL --dtype "fp8_w8a8" --tp 4
```

2. Model E2E
Server side:
```
MODEL=moonshotai/Kimi-K2-Instruct
python -m vllm.entrypoints.openai.api_server --model $MODEL --enable-expert-parallel --max-model-len=65000 -tp 4 --port 8101 --gpu_memory_utilization=0.95
```

Bench:
```
python benchmarks/benchmark_serving.py --backend vllm --model $MODEL --dataset-name random --random-input-len 7500 --random-output-len 7500 --max-concurrency 128 --num-prompts 512 --port 8101`
```
## Test Result
1. Single benchmark
<img width="643" height="402" alt="Screenshot 2025-09-10 at 9 18 12 AM" src="https://github.com/user-attachments/assets/44320921-dbf2-434c-b7fc-1738d4d0851f" />
<img width="645" height="393" alt="Screenshot 2025-09-10 at 9 18 31 AM" src="https://github.com/user-attachments/assets/6ef2a004-2217-4186-b20e-5f387509385c" />
<img width="641" height="389" alt="Screenshot 2025-09-10 at 9 20 20 AM" src="https://github.com/user-attachments/assets/b3cb9f32-2840-45c0-98e8-69420c48d6e6" />




2. Model E2E

BEFORE:
```
============ Serving Benchmark Result ============
Successful requests:                     200       
Maximum request concurrency:             64        
Benchmark duration (s):                  499.10    
Total input tokens:                      1499747   
Total generated tokens:                  276363    
Request throughput (req/s):              0.40      
Output token throughput (tok/s):         553.73    
Total Token throughput (tok/s):          3558.65   
---------------Time to First Token----------------
Mean TTFT (ms):                          12740.87  
Median TTFT (ms):                        13355.67  
P99 TTFT (ms):                           21087.93  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          272.12    
Median TPOT (ms):                        360.46    
P99 TPOT (ms):                           387.18    
---------------Inter-token Latency----------------
Mean ITL (ms):                           62.89     
Median ITL (ms):                         57.45     
P99 ITL (ms):                            363.68    
==================================================
```

AFTER:
```
============ Serving Benchmark Result ============
Successful requests:                     200       
Maximum request concurrency:             64        
Benchmark duration (s):                  424.97    
Total input tokens:                      1499747   
Total generated tokens:                  277351    
Request throughput (req/s):              0.47      
Output token throughput (tok/s):         652.64    
Total Token throughput (tok/s):          4181.70   
---------------Time to First Token----------------
Mean TTFT (ms):                          13136.72  
Median TTFT (ms):                        13603.26  
P99 TTFT (ms):                           22379.70  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          269.90    
Median TPOT (ms):                        359.65    
P99 TPOT (ms):                           386.72    
---------------Inter-token Latency----------------
Mean ITL (ms):                           52.96     
Median ITL (ms):                         47.41     
P99 ITL (ms):                            363.21    
==================================================

```
We see 17.8% in `Output token throughput (tok/s)`. The bulk of improvement is in `Inter-token Latency`.

---
- [x ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x ] The test plan, such as providing test command.
- [x ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

(no discussion comments)