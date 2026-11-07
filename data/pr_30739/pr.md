# PR #30739: [BugFix] Support online dense model DP without overhead

- field: scheduling  | area: v1/engine
- author: njhill
- state: MERGED  created: 2025-12-16T04:08:06Z  merged: 2026-01-02T15:36:38Z
- url: https://github.com/vllm-project/vllm/pull/30739
- labels: ready, v1, kv-connector

## Body

Currently, there's unnecessary overhead when running non-MoE models in a data parallel configuration because the steps across the ranks are synchronized with redundant all-reduce ops and coordination is done to ensure "idle" ranks perform dummy forward passes.

This PR changes the parallel config at the worker level to be equivalent to DP=1 for non-MoE models, so each rank operates independently. When internal load-balancing is used, the DP coordinator still runs to propagate stats back from the engines for load balancing purposes, but the step/wave synchronization logic is disabled.

Fixes https://github.com/vllm-project/vllm/issues/24461.
Fixes https://github.com/vllm-project/vllm/issues/30655.

This is supported in the online / AsyncLLM case only.

The [offline DP](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/data_parallel.py) will now fail during startup for non-MoE models (it really makes no sense to use it in that configuration).

Benchmark on 4xH100:
```
vllm serve Qwen/Qwen3-8B --data-parallel-size 4 --uvicorn-log-level=error
```
```
vllm bench serve \
    --backend vllm \
    --model Qwen/Qwen3-8B \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 512 \
    --ignore-eos \
    --port 8033 \
    --num-prompts 4000 \
    --max-concurrency 200 \
    --seed 42
```

#### Before
```
============ Serving Benchmark Result ============
Successful requests:                     4000      
Failed requests:                         0         
Maximum request concurrency:             200       
Benchmark duration (s):                  104.41    
Total input tokens:                      512000    
Total generated tokens:                  2048000   
Request throughput (req/s):              38.31     
Output token throughput (tok/s):         19615.26  
Peak output token throughput (tok/s):    21597.00  
Peak concurrent requests:                400.00    
Total token throughput (tok/s):          24519.08  
---------------Time to First Token----------------
Mean TTFT (ms):                          131.78    
Median TTFT (ms):                        124.31    
P99 TTFT (ms):                           404.36    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.93      
Median TPOT (ms):                        9.95      
P99 TPOT (ms):                           10.08     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.93      
Median ITL (ms):                         9.79      
P99 ITL (ms):                            15.93     
==================================================
```

#### After
```
============ Serving Benchmark Result ============
Successful requests:                     4000      
Failed requests:                         0         
Maximum request concurrency:             200       
Benchmark duration (s):                  99.24     
Total input tokens:                      512000    
Total generated tokens:                  2048000   
Request throughput (req/s):              40.31     
Output token throughput (tok/s):         20636.52  
Peak output token throughput (tok/s):    22454.00  
Peak concurrent requests:                400.00    
Total token throughput (tok/s):          25795.66  
---------------Time to First Token----------------
Mean TTFT (ms):                          88.94     
Median TTFT (ms):                        74.67     
P99 TTFT (ms):                           379.48    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.50      
Median TPOT (ms):                        9.50      
P99 TPOT (ms):                           9.66      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.50      
Median ITL (ms):                         9.38      
P99 ITL (ms):                            12.86     
==================================================
```

## Comments

### chatgpt-codex-connector @ 2025-12-16T04:08:12Z

Codex usage limits have been reached for code reviews. Please check with the admins of this repo to increase the limits by adding credits.
---
### njhill @ 2025-12-19T04:31:49Z

The one CI failure [`blackwell-fusion-and-compile-tests`](https://buildkite.com/vllm/ci/builds/44282#019b344d-3336-4432-bcab-3925b30e8fe2) is also failing on main.
---
### njhill @ 2025-12-29T20:34:52Z

Some additional benchmarks to see how close this is to linear scaling now (as suggested by @benchislett).

Shows about a 5% throughput overhead from the load-balancing which seems fairly high but:
- There would also be _some_ overhead from external LB (especially without queue awareness)
- Without this PR the overhead is ~ 15% 

Note these were run on a different H100 system to those above:

baseline
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             50        
Benchmark duration (s):                  94.80     
Total input tokens:                      128000    
Total generated tokens:                  512000    
Request throughput (req/s):              10.55     
Output token throughput (tok/s):         5400.99   
Peak output token throughput (tok/s):    5900.00   
Peak concurrent requests:                100.00    
Total token throughput (tok/s):          6751.24   
---------------Time to First Token----------------
Mean TTFT (ms):                          99.95     
Median TTFT (ms):                        98.57     
P99 TTFT (ms):                           196.58    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.07      
Median TPOT (ms):                        9.06      
P99 TPOT (ms):                           9.21      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.09      
Median ITL (ms):                         8.99      
P99 ITL (ms):                            12.62     
==================================================
```

`--data-parallel-size=4 --api-server-count=2`
```
============ Serving Benchmark Result ============
Successful requests:                     4000      
Failed requests:                         0         
Maximum request concurrency:             200       
Benchmark duration (s):                  95.64     
Total input tokens:                      512000    
Total generated tokens:                  2048000   
Request throughput (req/s):              41.82     
Output token throughput (tok/s):         21412.58  
Peak output token throughput (tok/s):    23104.00  
Peak concurrent requests:                400.00    
Total token throughput (tok/s):          26765.73  
---------------Time to First Token----------------
Mean TTFT (ms):                          44.06     
Median TTFT (ms):                        29.27     
P99 TTFT (ms):                           276.32    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.20      
Median TPOT (ms):                        9.18      
P99 TPOT (ms):                           9.68      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.20      
Median ITL (ms):                         9.08      
P99 ITL (ms):                            12.54     
==================================================
```

`--data-parallel-size=8 --api-server-count=4`
```
============ Serving Benchmark Result ============
Successful requests:                     8000      
Failed requests:                         0         
Maximum request concurrency:             400       
Benchmark duration (s):                  99.78     
Total input tokens:                      1024000   
Total generated tokens:                  4096000   
Request throughput (req/s):              80.18     
Output token throughput (tok/s):         41050.01  
Peak output token throughput (tok/s):    44736.00  
Peak concurrent requests:                800.00    
Total token throughput (tok/s):          51312.51  
---------------Time to First Token----------------
Mean TTFT (ms):                          104.06    
Median TTFT (ms):                        82.76     
P99 TTFT (ms):                           382.65    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.35      
Median TPOT (ms):                        9.24      
P99 TPOT (ms):                           10.12     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.40      
Median ITL (ms):                         9.38      
P99 ITL (ms):                            24.30     
==================================================
```

`--data-parallel-size=8 --api-server-count=4` (current main)
```
============ Serving Benchmark Result ============
Successful requests:                     8000      
Failed requests:                         0         
Maximum request concurrency:             400       
Benchmark duration (s):                  110.59    
Total input tokens:                      1024000   
Total generated tokens:                  4096000   
Request throughput (req/s):              72.34     
Output token throughput (tok/s):         37038.73  
Peak output token throughput (tok/s):    41507.00  
Peak concurrent requests:                800.00    
Total token throughput (tok/s):          46298.42  
---------------Time to First Token----------------
Mean TTFT (ms):                          154.06    
Median TTFT (ms):                        142.43    
P99 TTFT (ms):                           409.12    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.43     
Median TPOT (ms):                        10.43     
P99 TPOT (ms):                           10.82     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.51     
Median ITL (ms):                         10.17     
P99 ITL (ms):                            25.18     
==================================================
```

---
### mgoin @ 2026-01-02T23:19:17Z

@njhill do you think we should consider automatically setting api_server_count as we scale dp? For instance in your benchmarks you seemed to use 1/2 dp size
---