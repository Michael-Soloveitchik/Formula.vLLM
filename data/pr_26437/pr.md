# PR #26437: [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h)

- field: attention  | area: model_executor/layers
- author: vadiklyutiy
- state: MERGED  created: 2025-10-08T18:08:37Z  merged: 2025-10-16T04:18:32Z
- url: https://github.com/vllm-project/vllm/pull/26437
- labels: ready, v1, qwen

## Body

## Purpose
Qwen3-next MTP suffered from d<->h memory transfers on prefill phase. That makes MTP slower than STM (at least for some inputs). 

This PR fixes the following
1. Remove boolean indexing from GDN. Change it on direct indexing. (`tensor[bool_mask]` cause d2h transfer of number of `True` element needed for creation resulting tensor).
2. Increase cache size in `@tensor_cache`. There are 2 calls for every GDN attn, they are grouped by 3. We need at least 2*3 cache entry. 
3. Make early `.contiguous()` for q,k,v in GDN. This is not help right now but helps with `torch.compile` to make better fusion. 


## Test Result
B200. Test MTP with 512 concurrency. 
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --no-enable-prefix-caching --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 2}' --cuda-graph-sizes=2048
```
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 1024 --random-output 1024 --max-concurrency 512 --num-prompt 512 --ignore-eos
```

Before
```
============ Serving Benchmark Result ============
Successful requests:                     512
Maximum request concurrency:             512
Benchmark duration (s):                  32.88
Total input tokens:                      524288
Total generated tokens:                  524288
Request throughput (req/s):              15.57
Output token throughput (tok/s):         15947.50
Peak output token throughput (tok/s):    9216.00
Peak concurrent requests:                512.00
Total Token throughput (tok/s):          31894.99
---------------Time to First Token----------------
Mean TTFT (ms):                          5594.84
Median TTFT (ms):                        5429.04
P99 TTFT (ms):                           11260.93
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.89
Median TPOT (ms):                        22.99
P99 TPOT (ms):                           29.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           64.31
Median ITL (ms):                         56.32
P99 ITL (ms):                            165.80
==================================================
```

After
```
============ Serving Benchmark Result ============
Successful requests:                     512
Maximum request concurrency:             512
Benchmark duration (s):                  29.34
Total input tokens:                      524288
Total generated tokens:                  524288
Request throughput (req/s):              17.45
Output token throughput (tok/s):         17872.27
Peak output token throughput (tok/s):    9727.00
Peak concurrent requests:                512.00
Total Token throughput (tok/s):          35744.55
---------------Time to First Token----------------
Mean TTFT (ms):                          3953.48
Median TTFT (ms):                        3795.80
P99 TTFT (ms):                           7914.50
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          21.12
Median TPOT (ms):                        21.04
P99 TPOT (ms):                           26.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           59.34
Median ITL (ms):                         56.31
P99 ITL (ms):                            117.85
==================================================
```

Speedup of Output token throughput is **12%**


For comparison STP mode:
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --no-enable-prefix-caching --cuda-graph-sizes=2048
```
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 1024 --random-output 1024 --max-concurrency 512 --num-prompt 512 --ignore-eos
```

```
============ Serving Benchmark Result ============
Successful requests:                     512
Maximum request concurrency:             512
Benchmark duration (s):                  37.51
Total input tokens:                      524288
Total generated tokens:                  524288
Request throughput (req/s):              13.65
Output token throughput (tok/s):         13976.35
Peak output token throughput (tok/s):    17920.00
Peak concurrent requests:                512.00
Total Token throughput (tok/s):          27952.70
---------------Time to First Token----------------
Mean TTFT (ms):                          3698.59
Median TTFT (ms):                        3467.95
P99 TTFT (ms):                           7608.89
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.47
Median TPOT (ms):                        32.75
P99 TPOT (ms):                           34.83
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.47
Median ITL (ms):                         29.54
P99 ITL (ms):                            131.08
==================================================
```


## Comments

### xinli-sw @ 2025-10-10T01:59:08Z

Hi @benchislett @LucasWilkinson could you assign reviewers for this change? Thanks!
---
### youkaichao @ 2025-10-10T09:19:05Z

cc @sighingnow should be back
---
### mergify @ 2025-10-14T04:33:20Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @vadiklyutiy.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---