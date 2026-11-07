# PR #37695: [Perf] Use torch compile to fuse pack topk in trtllm moe

- field: quantization  | area: model_executor/layers
- author: wzhao18
- state: MERGED  created: 2026-03-20T15:21:34Z  merged: 2026-03-27T23:30:46Z
- url: https://github.com/vllm-project/vllm/pull/37695
- labels: performance, ready, nvidia

## Body

## Purpose
Fuse the packing topk ids and weights op in trtllm MoE using torch.compile. From benchmarking, this gives ~2% speedup in Minimax M2.5 TP=2 Concurrency 64 1K/1K across FP8 and NVFP4.

## Test Plan
Use minimax m2.5 fp8 and nvfp4 (private checkpoint) for accuracy and performance testing.

## Test Result
FP8:
```
vllm serve MiniMaxAI/MiniMax-M2.5 \
    --trust-remote-code \
    --stream-interval 20 \
    --tensor-parallel-size 2

vllm bench serve \
    --endpoint /v1/completions \
    --port 8000 \
    --model MiniMaxAI/MiniMax-M2.5 \
    --dataset-name random \
    --random-input 1024 \
    --random-output 1024 \
    --num-prompt 1280 \
    --max-concurrency 64 \
    --num-warmups 100  \
    --ignore-eos    \
    --trust-remote-code

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9303|±  |0.0070|
|     |       |strict-match    |     5|exact_match|↑  |0.9287|±  |0.0071|

Main:
============ Serving Benchmark Result ============
Successful requests:                     1280      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  482.50    
Total input tokens:                      1310720   
Total generated tokens:                  1310720   
Request throughput (req/s):              2.65      
Output token throughput (tok/s):         2716.55   
Peak output token throughput (tok/s):    173.00    
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          5433.09   
---------------Time to First Token----------------
Mean TTFT (ms):                          390.04    
Median TTFT (ms):                        350.18    
P99 TTFT (ms):                           1283.75   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          23.19     
Median TPOT (ms):                        23.23     
P99 TPOT (ms):                           23.58     
---------------Inter-token Latency----------------
Mean ITL (ms):                           456.29    
Median ITL (ms):                         446.20    
P99 ITL (ms):                            1108.22   
==================================================

Branch:
============ Serving Benchmark Result ============
Successful requests:                     1280      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  474.09    
Total input tokens:                      1310720   
Total generated tokens:                  1310720   
Request throughput (req/s):              2.70      
Output token throughput (tok/s):         2764.70   
Peak output token throughput (tok/s):    199.00    
Peak concurrent requests:                127.00    
Total token throughput (tok/s):          5529.39   
---------------Time to First Token----------------
Mean TTFT (ms):                          468.97    
Median TTFT (ms):                        373.37    
P99 TTFT (ms):                           1358.45   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.71     
Median TPOT (ms):                        22.73     
P99 TPOT (ms):                           23.16     
---------------Inter-token Latency----------------
Mean ITL (ms):                           446.70    
Median ITL (ms):                         437.64    
P99 ITL (ms):                            1055.59   
==================================================
```

NVFP4:
```
vllm serve MiniMax-M2.5-NVFP4  \
    --trust-remote-code \
    --stream-interval 20 \
    --tensor-parallel-size 2

vllm bench serve \
    --endpoint /v1/completions \
    --port 8000 \
    --model MiniMax-M2.5-NVFP4 \
    --dataset-name random \
    --random-input 1024 \
    --random-output 1024 \
    --num-prompt 1280 \
    --max-concurrency 64 \
    --num-warmups 100  \
    --ignore-eos    \
    --trust-remote-code

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9257|±  |0.0072|
|     |       |strict-match    |     5|exact_match|↑  |0.9212|±  |0.0074|

Main:
============ Serving Benchmark Result ============
Successful requests:                     1280      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  318.45    
Total input tokens:                      1310720   
Total generated tokens:                  1310720   
Request throughput (req/s):              4.02      
Output token throughput (tok/s):         4115.97   
Peak output token throughput (tok/s):    254.00    
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          8231.94   
---------------Time to First Token----------------
Mean TTFT (ms):                          284.88    
Median TTFT (ms):                        231.55    
P99 TTFT (ms):                           940.15    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.28     
Median TPOT (ms):                        15.32     
P99 TPOT (ms):                           15.56     
---------------Inter-token Latency----------------
Mean ITL (ms):                           300.62    
Median ITL (ms):                         293.73    
P99 ITL (ms):                            737.18    
==================================================

Branch:
============ Serving Benchmark Result ============
Successful requests:                     1280      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  312.61    
Total input tokens:                      1310720   
Total generated tokens:                  1310720   
Request throughput (req/s):              4.09      
Output token throughput (tok/s):         4192.77   
Peak output token throughput (tok/s):    261.00    
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          8385.54   
---------------Time to First Token----------------
Mean TTFT (ms):                          386.67    
Median TTFT (ms):                        413.34    
P99 TTFT (ms):                           927.91    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.90     
Median TPOT (ms):                        14.88     
P99 TPOT (ms):                           15.23     
---------------Inter-token Latency----------------
Mean ITL (ms):                           293.09    
Median ITL (ms):                         288.91    
P99 ITL (ms):                            582.89    
==================================================
```


---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### vadiklyutiy @ 2026-03-26T18:29:53Z

I was thinking why thes ops do not capture by `@support_torch_compile`
Found #31985 where it is planned to remove high level custom op and after that this code should be `toch.compile`-ed automatically
---