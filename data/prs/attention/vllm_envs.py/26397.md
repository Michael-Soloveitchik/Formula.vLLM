# PR #26397: [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek

- field: attention  | area: vllm/envs.py
- author: minosfuture
- state: MERGED  created: 2025-10-08T05:34:51Z  merged: 2025-10-24T17:24:09Z
- url: https://github.com/vllm-project/vllm/pull/26397
- labels: ready, v1, deepseek

## Body

## Purpose

Add MLA prefill backend: trtllm_ragged_attention_deepseek
* controlled by `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1`

* [x] Need to fix accuracy issue
  * 0.54 -> 0.59 fixed by zeroing output tensor for context chunk prefill
  * 0.59 -> 0.61 (0.22 -> 0.64 w/ prefix caching) fixed by transposing lse
  * 0.61 -> 0.65 fixed by zeroing workspace
* [x] performance benchmark

## Test Plan

* [ ] microbenchmark
  * in follow-up work after refactoring
* [x] vllm bench serve
* [x] lm_eval gsm8k

## Test Result

### bench serve

12.7% qps improvement
-10.2% TTFT improvement

#### Baseline
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/fi.log
```
```
============ Serving Benchmark Result ============
Successful requests:                     2048
Maximum request concurrency:             2048
Benchmark duration (s):                  200.42
Total input tokens:                      4192256
Total generated tokens:                  2097152
Request throughput (req/s):              10.22
Output token throughput (tok/s):         10463.54
Peak output token throughput (tok/s):    16094.00
Peak concurrent requests:                2048.00
Total Token throughput (tok/s):          31380.41
---------------Time to First Token----------------
Mean TTFT (ms):                          64511.82
Median TTFT (ms):                        55968.36
P99 TTFT (ms):                           132592.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          86.54
Median TPOT (ms):                        89.57
P99 TPOT (ms):                           93.98
---------------Inter-token Latency----------------
Mean ITL (ms):                           86.54
Median ITL (ms):                         70.26
P99 ITL (ms):                            208.33
==================================================
```
#### This PR
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/ragged.log
```

```
============ Serving Benchmark Result ============
Successful requests:                     2048
Maximum request concurrency:             2048
Benchmark duration (s):                  177.74
Total input tokens:                      4192256
Total generated tokens:                  2097152
Request throughput (req/s):              11.52
Output token throughput (tok/s):         11799.21
Peak output token throughput (tok/s):    18098.00
Peak concurrent requests:                2048.00
Total Token throughput (tok/s):          35386.10
---------------Time to First Token----------------
Mean TTFT (ms):                          57924.67
Median TTFT (ms):                        49533.53
P99 TTFT (ms):                           118626.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          76.15
Median TPOT (ms):                        78.87
P99 TPOT (ms):                           83.38
---------------Inter-token Latency----------------
Mean ITL (ms):                           76.16
Median ITL (ms):                         60.18
P99 ITL (ms):                            181.91
==================================================
```

### gsm8k

#### Baseline
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" --enforce-eager 2>&1 | tee /tmp/fi.log

local-completions (model=deepseek-ai/DeepSeek-V2-Lite-Chat,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=256), gen_kwargs: (None), limit: None, num_fewshot: 8, batch_size: 1
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.6520|±  |0.0131|
|     |       |strict-match    |     8|exact_match|↑  |0.6482|±  |0.0132|

#### This PR
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" --enforce-eager 2>&1 | tee /tmp/ragged.log

local-completions (model=deepseek-ai/DeepSeek-V2-Lite-Chat,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=256), gen_kwargs: (None), limit: None, num_fewshot: 8, batch_size: 1
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.655|±  |0.0131|
|     |       |strict-match    |     8|exact_match|↑  |0.649|±  |0.0131|

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

### mergify @ 2025-10-14T04:33:06Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @minosfuture.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---