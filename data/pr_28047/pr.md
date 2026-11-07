# PR #28047: [Hybrid] Mamba2 prefix cache blocks freeing for running requests

- field: kv_cache  | area: v1/core
- author: s3woz
- state: MERGED  created: 2025-11-04T14:46:09Z  merged: 2025-12-25T20:54:06Z
- url: https://github.com/vllm-project/vllm/pull/28047
- labels: ready, v1

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model

## Purpose
This PR implements freeing of prefix cache blocks of running request for Mamba2 hybrid models.

Logic before this PR:
- Mamba2 logic keeps all prefix cache blocks for running requests, preventing their eviction.

This PR marks the prefix cache blocks that aren't strictly needed by running requests as available for eviction. Note: This doesn't mean that they are evicted immediately, but rather that they are added to the list of memory blocks potentially available for eviction if more memory is needed. Thus, unless explicitly evicted, they remain available for the prefix caching mechanism.

The benefits of this PR become especially evident for long prompts and tight memory conditions.

## Test Plan
Long prompts:
- case 1: prompt 65536 => **latency reduction through this PR: 18%**
```
vllm bench latency --model ibm-granite/granite-4.0-tiny-preview --input-len 65536 \
    --output-len 10 --batch-size 8 --enable-prefix-caching --num-iters 10 --num-iters-warmup 2
```
- case 2: prompt 100000 - memory tight situation => **latency reduction through this PR: 96%**
```
vllm bench latency --model ibm-granite/granite-4.0-tiny-preview --input-len 100000 \
    --output-len 10 --batch-size 8 --enable-prefix-caching --num-iters 10 --num-iters-warmup 2
```

Short prompts (standard vllm bench serve) - the prompts don't leverage APC => **throughput increase through this PR: 2.5%**
```
vllm serve --enable-prefix-caching ibm-granite/granite-4.0-tiny-preview
vllm bench serve \
        --model ibm-granite/granite-4.0-tiny-preview \
        --dataset-name sharegpt \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
        --ignore_eos
```
## Test Result
Long prompts - Case 1:
Main:
```
Avg latency: 0.4160831045010127 seconds
10% percentile latency: 0.3994843637163285 seconds
25% percentile latency: 0.4013041044890997 seconds
50% percentile latency: 0.4097101140068844 seconds
75% percentile latency: 0.4214230982470326 seconds
90% percentile latency: 0.4347237055102596 seconds
99% percentile latency: 0.47011098143615526 seconds
```
This PR:
```
Avg latency: 0.34099626759416424 seconds
10% percentile latency: 0.3223433266801294 seconds
25% percentile latency: 0.3421008375007659 seconds
50% percentile latency: 0.3453638869978022 seconds
75% percentile latency: 0.34755055123969214 seconds
90% percentile latency: 0.34979368318163323 seconds
99% percentile latency: 0.3580754731138586 seconds
```

Long prompts - case 2:
Main:
```
Avg latency: 11.027259302898893 seconds
10% percentile latency: 10.976501701082451 seconds
25% percentile latency: 10.997937818021455 seconds
50% percentile latency: 11.002901612999267 seconds
75% percentile latency: 11.030581445753342 seconds
90% percentile latency: 11.101565936085535 seconds
99% percentile latency: 11.161986404604978 seconds
```
This PR:
```
Avg latency: 0.4853071654040832 seconds
10% percentile latency: 0.46571668109390885 seconds
25% percentile latency: 0.47812563750630943 seconds
50% percentile latency: 0.4824099554971326 seconds
75% percentile latency: 0.4912291752625606 seconds
90% percentile latency: 0.49907766550895755 seconds
99% percentile latency: 0.521558863160899 seconds
```

Short prompts:
Main:
```
============ Serving Benchmark Result ============
Successful requests:                     983
Failed requests:                         17
Benchmark duration (s):                  48.97
Total input tokens:                      236351
Total generated tokens:                  222312
Request throughput (req/s):              20.07
Output token throughput (tok/s):         4539.73
Peak output token throughput (tok/s):    11925.00
Peak concurrent requests:                983.00
Total Token throughput (tok/s):          9366.14
---------------Time to First Token----------------
Mean TTFT (ms):                          11042.79
Median TTFT (ms):                        10219.31
P99 TTFT (ms):                           24133.59
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          267.42
Median TPOT (ms):                        124.23
P99 TPOT (ms):                           938.34
---------------Inter-token Latency----------------
Mean ITL (ms):                           91.54
Median ITL (ms):                         40.08
P99 ITL (ms):                            1035.01
==================================================
```
This PR:
```
============ Serving Benchmark Result ============
Successful requests:                     983
Failed requests:                         17
Benchmark duration (s):                  47.99
Total input tokens:                      237555
Total generated tokens:                  223058
Request throughput (req/s):              20.48
Output token throughput (tok/s):         4647.84
Peak output token throughput (tok/s):    11856.00
Peak concurrent requests:                983.00
Total Token throughput (tok/s):          9597.75
---------------Time to First Token----------------
Mean TTFT (ms):                          10313.96
Median TTFT (ms):                        8748.50
P99 TTFT (ms):                           23262.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          265.20
Median TPOT (ms):                        122.41
P99 TPOT (ms):                           949.72
---------------Inter-token Latency----------------
Mean ITL (ms):                           90.57
Median ITL (ms):                         40.05
P99 ITL (ms):                            1040.46
==================================================
```

## Comments

### mergify @ 2025-11-11T12:46:45Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @s3woz.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tdoublep @ 2025-11-19T13:01:14Z

cc @heheda12345 
---