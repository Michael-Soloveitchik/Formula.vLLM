# PR #24330: QWEN3 Thinking Fused MoE kernels Optimization configs

- field: quantization  | area: model_executor/layers
- author: samanamp
- state: MERGED  created: 2025-09-05T16:13:22Z  merged: 2025-09-07T03:18:54Z
- url: https://github.com/vllm-project/vllm/pull/24330
- labels: performance, ready, qwen

## Body

<!-- markdownlint-disable -->

## Purpose
Optimize QWEN3 Thinking Fused MoE kernels Optimization configs

We see 13.7% in `Output token throughput (tok/s)`.  12% in `Median TPOT (ms)` and 17% improvement in P99 TPOT (ms).

## Test Plan
1. Single benchmark
```
python benchmarks/kernels/benchmark_moe.py --model $MODEL --dtype "fp8_w8a8" --tp 4
```

2. Model E2E
Server side:
```
MODEL=Qwen/Qwen3-235B-A22B-Thinking-2507-FP8
python -m vllm.entrypoints.openai.api_server --model $MODEL --enable-expert-parallel --max-model-len=65000 -tp 4 --port 8101 --gpu_memory_utilization=0.95
```

Bench:
```
python benchmarks/benchmark_serving.py --backend vllm --model $MODEL --dataset-name random --random-input-len 7500 --random-output-len 7500 --max-concurrency 64 --num-prompts 128 --port 8101`
```
## Test Result
1. Single benchmark
<img width="802" height="496" alt="Screenshot 2025-09-05 at 9 16 13 AM" src="https://github.com/user-attachments/assets/b3f97dda-08ae-4cd6-8a53-c838b93478f2" />
<img width="806" height="494" alt="Screenshot 2025-09-05 at 9 16 25 AM" src="https://github.com/user-attachments/assets/74607b30-cb31-4499-b0ba-80661a5b5770" />


2. Model E2E

BEFORE:
```
============ Serving Benchmark Result ============                                                                                                                                           
Successful requests:                     128                                                                                                                                                 
Maximum request concurrency:             64                                                                                                                                                  
Benchmark duration (s):                  576.22                                                                                                                                              
Total input tokens:                      959630                                                                                                                                              
Total generated tokens:                  960000                                                                                                                                              
Request throughput (req/s):              0.22                                                                                                                                                
Output token throughput (tok/s):         1666.03                                                                                                                                             
Total Token throughput (tok/s):          3331.42                                                                                                                                             
---------------Time to First Token----------------                                                                                                                                           
Mean TTFT (ms):                          6815.24                                                                                                                                             
Median TTFT (ms):                        1664.75                                                                                                                                             
P99 TTFT (ms):                           23540.15                                                                                                                                            
-----Time per Output Token (excl. 1st token)------                                                                                                                                           
Mean TPOT (ms):                          37.46                                                                                                                                               
Median TPOT (ms):                        38.16                                                                                                                                               
P99 TPOT (ms):                           40.42                                                                                                                                               
---------------Inter-token Latency----------------                                                                                                                                           
Mean ITL (ms):                           37.46                                                                                                                                               
Median ITL (ms):                         33.29                                                                                                                                               
P99 ITL (ms):                            136.71                                                                                                                                              
==================================================
```

AFTER:
```
============ Serving Benchmark Result ============
Successful requests:                     128        
Maximum request concurrency:             64         
Benchmark duration (s):                  506.50     
Total input tokens:                      959630     
Total generated tokens:                  960000     
Request throughput (req/s):              0.25       
Output token throughput (tok/s):         1895.36   
Total Token throughput (tok/s):          3789.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          6768.80   
Median TTFT (ms):                        1666.67   
P99 TTFT (ms):                           23394.16  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.82      
Median TPOT (ms):                        33.53      
P99 TPOT (ms):                           33.65      
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.82      
Median ITL (ms):                         30.64      
P99 ITL (ms):                            33.69      
==================================================
```
We see 13.7% in `Output token throughput (tok/s)`.  12% in `Median TPOT (ms)` and 17% improvement in P99 TPOT (ms).

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ x] The test plan, such as providing test command.
- [ x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

(no discussion comments)