# PR #35777: [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next

- field: gpu_kernels  | area: model_executor/layers
- author: xyang16
- state: MERGED  created: 2026-03-02T18:00:43Z  merged: 2026-03-09T06:41:01Z
- url: https://github.com/vllm-project/vllm/pull/35777
- labels: ready, qwen

## Body


## Purpose

This PR adds `fused_sigmoid_gating_delta_rule_update` kernel, to save memory traffic and launch overhead. The idea is inspired by [vllm-ascend](https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/ops/triton/fla/sigmoid_gating.py)



Summary:
* Fuse `fused_gdn_gating` and `fused_recurrent_gated_delta_rule` into `fused_sigmoid_gating_delta_rule_update` kernel.
* Support both spec decoding and non spec decoding.
* Replaced `b_q / sqrt` with `b_q * rsqrt`
* Added unit test test_fused_sigmoid_gating_delta_rule.py

## Test Plan


```
pytest -s -v tests/kernels/test_fused_sigmoid_gating_delta_rule.py
```

## Test Result

Unit tests passed.

## Profiling

Main:

<img width="1905" height="742" alt="Screenshot 2026-03-02 at 9 57 48 AM" src="https://github.com/user-attachments/assets/e069223f-df6a-4903-b058-f39d83bd919f" />

PR:

<img width="1901" height="742" alt="Screenshot 2026-03-02 at 9 59 41 AM" src="https://github.com/user-attachments/assets/a7588337-075b-41a3-b5a5-5d6a827d60b3" />

Main: Launched `fused_gdn_gating_kernel` and `fused_recurrent_gated_delta_rule_fwd_kernel` kernels, takes about 5µs.

PR: Launched `fused_sigmoid_gating_delta_rule_update_kernel` launched, takes about 3µs.

## Benchmarking

Benchmarked on H200.

* Qwen3-Next

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
    --model Qwen/Qwen3-Next-80B-A3B-Thinking \
    --dataset-name sharegpt \
    --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 300 \
    --num-prompts 1000 \
    --max-concurrency 8 \
    --num-warmups 50 \
    --ignore-eos \
    --temperature 0 \
    --percentile-metrics "ttft,tpot,itl,e2el"
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  277.20    
Total input tokens:                      229334    
Total generated tokens:                  300000    
Request throughput (req/s):              3.61      
Output token throughput (tok/s):         1082.26   
Peak output token throughput (tok/s):    1264.00   
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1909.58   
---------------Time to First Token----------------
Mean TTFT (ms):                          171.61    
Median TTFT (ms):                        191.72    
P99 TTFT (ms):                           206.54    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.84      
Median TPOT (ms):                        6.83      
P99 TPOT (ms):                           7.47      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.84      
Median ITL (ms):                         6.80      
P99 ITL (ms):                            7.67      
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2217.20   
Median E2EL (ms):                        2219.50   
P99 E2EL (ms):                           2378.66   
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  274.13    
Total input tokens:                      229334    
Total generated tokens:                  300000    
Request throughput (req/s):              3.65      
Output token throughput (tok/s):         1094.38   
Peak output token throughput (tok/s):    1288.00   
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1930.97   
---------------Time to First Token----------------
Mean TTFT (ms):                          173.51    
Median TTFT (ms):                        192.43    
P99 TTFT (ms):                           210.37    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.75      
Median TPOT (ms):                        6.73      
P99 TPOT (ms):                           7.36      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.75      
Median ITL (ms):                         6.71      
P99 ITL (ms):                            7.63      
----------------End-to-end Latency----------------
Mean E2EL (ms):                          2192.52   
Median E2EL (ms):                        2198.85   
P99 E2EL (ms):                           2355.93   
==================================================
```

Output token throughput improves 1%.

* Qwen3-Next MTP

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking \
    --tensor-parallel-size 8 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 4}'
```
```
vllm bench serve \
    --model Qwen/Qwen3-Next-80B-A3B-Thinking \
    --dataset-name sharegpt \
    --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 300 \
    --num-prompts 1000 \
    --max-concurrency 8 \
    --num-warmups 50 \
    --ignore-eos \
    --temperature 0 \
    --percentile-metrics "ttft,tpot,itl,e2el"
```


Main:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  249.41    
Total input tokens:                      229334    
Total generated tokens:                  300000    
Request throughput (req/s):              4.01      
Output token throughput (tok/s):         1202.86   
Peak output token throughput (tok/s):    506.00    
Peak concurrent requests:                14.00     
Total token throughput (tok/s):          2122.38   
---------------Time to First Token----------------
Mean TTFT (ms):                          137.83    
Median TTFT (ms):                        150.55    
P99 TTFT (ms):                           269.18    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.20      
Median TPOT (ms):                        6.27      
P99 TPOT (ms):                           8.97      
---------------Inter-token Latency----------------
Mean ITL (ms):                           21.30     
Median ITL (ms):                         15.47     
P99 ITL (ms):                            125.43    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1992.31   
Median E2EL (ms):                        2012.57   
P99 E2EL (ms):                           2808.97   
---------------Speculative Decoding---------------
Acceptance rate (%):                     61.31     
Acceptance length:                       3.45      
Drafts:                                  87071     
Draft tokens:                            348284    
Accepted tokens:                         213544    
Per-position acceptance (%):
  Position 0:                            83.34     
  Position 1:                            66.26     
  Position 2:                            52.84     
  Position 3:                            42.82     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  243.71    
Total input tokens:                      229334    
Total generated tokens:                  300000    
Request throughput (req/s):              4.10      
Output token throughput (tok/s):         1230.99   
Peak output token throughput (tok/s):    492.00    
Peak concurrent requests:                15.00     
Total token throughput (tok/s):          2172.01   
---------------Time to First Token----------------
Mean TTFT (ms):                          132.46    
Median TTFT (ms):                        146.33    
P99 TTFT (ms):                           261.55    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.07      
Median TPOT (ms):                        6.11      
P99 TPOT (ms):                           8.92      
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.85     
Median ITL (ms):                         15.27     
P99 ITL (ms):                            121.96    
----------------End-to-end Latency----------------
Mean E2EL (ms):                          1946.20   
Median E2EL (ms):                        1957.20   
P99 E2EL (ms):                           2766.98   
---------------Speculative Decoding---------------
Acceptance rate (%):                     61.41     
Acceptance length:                       3.46      
Drafts:                                  86979     
Draft tokens:                            347916    
Accepted tokens:                         213639    
Per-position acceptance (%):
  Position 0:                            83.54     
  Position 1:                            66.40     
  Position 2:                            52.91     
  Position 3:                            42.77     
==================================================
```

Output token throughput improves 2%.

* Qwen3.5

```
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
  --model Qwen/Qwen3.5-35B-A3B \
  --dataset-name sharegpt \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 300 \
  --max-concurrency 8 \
  --num-prompts 1000 \
  --num-warmups 50 \
  --ignore-eos \
  --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  386.74    
Total input tokens:                      237165    
Total generated tokens:                  300000    
Request throughput (req/s):              2.59      
Output token throughput (tok/s):         775.71    
Peak output token throughput (tok/s):    912.00    
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1388.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          148.37    
Median TTFT (ms):                        151.86    
P99 TTFT (ms):                           192.13    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.85      
Median TPOT (ms):                        9.84      
P99 TPOT (ms):                           10.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.85      
Median ITL (ms):                         9.86      
P99 ITL (ms):                            11.44     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  384.28    
Total input tokens:                      237165    
Total generated tokens:                  300000    
Request throughput (req/s):              2.60      
Output token throughput (tok/s):         780.68    
Peak output token throughput (tok/s):    912.00    
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1397.84   
---------------Time to First Token----------------
Mean TTFT (ms):                          148.43    
Median TTFT (ms):                        152.61    
P99 TTFT (ms):                           191.52    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.78      
Median TPOT (ms):                        9.76      
P99 TPOT (ms):                           10.69     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.78      
Median ITL (ms):                         9.78      
P99 ITL (ms):                            11.34     
==================================================
```

* Qwen3.5 MTP

```
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 4}'
```

```
vllm bench serve \
  --model Qwen/Qwen3.5-35B-A3B \
  --dataset-name sharegpt \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 300 \
  --max-concurrency 8 \
  --num-prompts 1000 \
  --num-warmups 50 \
  --ignore-eos \
  --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  317.74    
Total input tokens:                      237165    
Total generated tokens:                  300000    
Request throughput (req/s):              3.15      
Output token throughput (tok/s):         944.17    
Peak output token throughput (tok/s):    376.00    
Peak concurrent requests:                15.00     
Total token throughput (tok/s):          1690.59   
---------------Time to First Token----------------
Mean TTFT (ms):                          125.82    
Median TTFT (ms):                        137.00    
P99 TTFT (ms):                           215.92    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.06      
Median TPOT (ms):                        7.95      
P99 TPOT (ms):                           11.80     
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.86     
Median ITL (ms):                         22.14     
P99 ITL (ms):                            97.29     
---------------Speculative Decoding---------------
Acceptance rate (%):                     55.67     
Acceptance length:                       3.23      
Drafts:                                  93129     
Draft tokens:                            372516    
Accepted tokens:                         207398    
Per-position acceptance (%):
  Position 0:                            80.20     
  Position 1:                            61.29     
  Position 2:                            46.09     
  Position 3:                            35.12     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  313.54    
Total input tokens:                      237165    
Total generated tokens:                  300000    
Request throughput (req/s):              3.19      
Output token throughput (tok/s):         956.81    
Peak output token throughput (tok/s):    368.00    
Peak concurrent requests:                14.00     
Total token throughput (tok/s):          1713.21   
---------------Time to First Token----------------
Mean TTFT (ms):                          124.21    
Median TTFT (ms):                        135.12    
P99 TTFT (ms):                           213.40    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.95      
Median TPOT (ms):                        7.91      
P99 TPOT (ms):                           11.31     
---------------Inter-token Latency----------------
Mean ITL (ms):                           25.71     
Median ITL (ms):                         22.11     
P99 ITL (ms):                            95.44     
---------------Speculative Decoding---------------
Acceptance rate (%):                     56.25     
Acceptance length:                       3.25      
Drafts:                                  92493     
Draft tokens:                            369972    
Accepted tokens:                         208109    
Per-position acceptance (%):
  Position 0:                            80.85     
  Position 1:                            61.81     
  Position 2:                            46.70     
  Position 3:                            35.63     
==================================================
```

## Accuracy Testing

* Qwen3-Next

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Thinking,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.721|±  |0.0124|
|     |       |strict-match    |     5|exact_match|↑  |0.718|±  |0.0124|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7233|±  |0.0123|
|     |       |strict-match    |     5|exact_match|↑  |0.7248|±  |0.0123|
```

* Qwen3-Next MTP

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Thinking,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7195|±  |0.0124|
|     |       |strict-match    |     5|exact_match|↑  |0.7187|±  |0.0124|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7293|±  |0.0122|
|     |       |strict-match    |     5|exact_match|↑  |0.7286|±  |0.0122|
```

* Qwen3.5

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-35B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8575|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8469|±  |0.0099|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8628|±  |0.0095|
|     |       |strict-match    |     5|exact_match|↑  |0.8544|±  |0.0097|
```

* Qwen3.5 MTP

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-35B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8537|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8423|±  |0.0100|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8537|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8423|±  |0.0100|
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

### ZJY0516 @ 2026-03-03T13:27:21Z

@xyang16 Could you please also test qwen 3.5?
---
### xyang16 @ 2026-03-04T01:17:14Z

> @xyang16 Could you please also test qwen 3.5?

@ZJY0516 Thanks for review! I updated added qwen 3.5 test result in the description.
---
### vadiklyutiy @ 2026-03-04T13:23:57Z

In #31722 I made some tuning of GDN decode. 
Could you pls double check that new approach doesn't break that tuning. For example, by running  E2E test from #31722. There was tested higher concurrency case.
---
### ZJY0516 @ 2026-03-05T12:28:43Z

Do we have any update?
---
### xyang16 @ 2026-03-06T02:31:51Z

@vadiklyutiy @ZJY0516 Right now I only have H200. I will request a B200, and run the test, I will update the result by eod. Thanks!
---
### ywang96 @ 2026-03-09T06:39:06Z

I've tested this PR on GB200 with a community NVFP4 checkpoint `Kbenkhaled/Qwen3.5-27B-NVFP4`. The setup is 2048 input and 1024 output tokens. All results done as average of 5 rounds.

## Performance Results

### High Throughput (concurrency=64)

| Branch | Run 1 | Run 2 | Run 3 | Average |
|--------|-------|-------|-------|---------|
| Main   | 16.65 | 16.23 | 16.11 | **16.33 ms** |
| PR     | 15.68 | 15.73 | 15.56 | **15.66 ms** |

**TPOT improvement: 4.1% (16.33 -> 15.66 ms)**

### Low Latency (concurrency=1)

| Branch | Run 1 | Run 2 | Run 3 | Average |
|--------|-------|-------|-------|---------|
| Main   | 8.58  | 8.58  | 8.58  | **8.58 ms** |
| PR     | 7.45  | 7.45  | 7.45  | **7.45 ms** |

**TPOT improvement: 13.2% (8.58 -> 7.45 ms)**


## Accuracy Results (GSM8K)

| Branch | Run | flexible-extract | strict-match |
|--------|-----|-----------------|--------------|
| Main   | 1   | 0.8506          | 0.8431       |
| Main   | 2   | 0.8317          | 0.8226       |
| PR     | 1   | 0.8870          | 0.8848       |
| PR     | 2   | 0.8499          | 0.8370       |

| Branch | Avg flexible-extract | Avg strict-match |
|--------|---------------------|-----------------|
| Main   | **0.8412**          | **0.8329**      |
| PR     | **0.8685**          | **0.8609**      |

Given these results I'm stamping this PR and merging it. Thanks for this fantastic contribution!




---