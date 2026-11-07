# PR #38372: [Hybrid] Simplify accepted token counting in spec decode for hybrid models

- field: decoding  | area: v1/worker
- author: fuscof-ibm
- state: MERGED  created: 2026-03-27T15:46:28Z  merged: 2026-04-14T22:19:10Z
- url: https://github.com/vllm-project/vllm/pull/38372
- labels: ready, v1

## Body

## Purpose

Replace the concatenate-sentinel-then-argmax approach with a simple sum of valid tokens. Since rejected tokens are marked with -1 and valid tokens are always contiguous from position 0, counting non-(-1) tokens directly gives the number of accepted tokens.

This eliminates unnecessary memory allocation (torch.cat, torch.full) and is more readable.


## Test Plan

Correctness test:

```
python -m pytest tests/v1/e2e/general/test_mamba_prefix_cache.py -vv -x -s
```

Latency test:

```
vllm bench latency    \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8    \
    --tensor-parallel-size 4        \
    --max-model-len 262144          \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' 
```

Serving benchmark with server:

```
vllm serve   \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8    \
    --tensor-parallel-size 4        \
    --max-model-len 262144          \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' 
 ```

and client:

```
vllm bench serve \
    --dataset-name sharegpt \
    --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json  \
    --request-rate 1 \
   --num-prompts 200 \
   --ignore_eos
 ```
 
 
Eval:

Running the gsm8k config from this  [pr](https://github.com/vllm-project/vllm/pull/35406)

```
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py --config-list-file=tests/evals/gsm8k/configs/hybrid/models-h100.txt -k 'Qwen3-Next-FP8-TP4-MTP-Align'
```

## Test Result


###

`main`:




```
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  202.24    
Total input tokens:                      43560     
Total generated tokens:                  44697     
Request throughput (req/s):              0.99      
Output token throughput (tok/s):         221.01    
Peak output token throughput (tok/s):    414.00    
Peak concurrent requests:                8.00      
Total token throughput (tok/s):          436.39    
---------------Time to First Token----------------
Mean TTFT (ms):                          63.82     
Median TTFT (ms):                        54.05     
P99 TTFT (ms):                           151.92    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.55      
Median TPOT (ms):                        4.34      
P99 TPOT (ms):                           8.16      
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.09     
Median ITL (ms):                         9.56      
P99 ITL (ms):                            29.78     
---------------Speculative Decoding---------------
Acceptance rate (%):                     66.51     
Acceptance length:                       2.33      
Drafts:                                  19163     
Draft tokens:                            38326     
Accepted tokens:                         25490     
Per-position acceptance (%):
  Position 0:                            77.91     
  Position 1:                            55.11     
==================================================
```

`this branch`:

```
============ Serving Benchmark Result ============
Successful requests:                     200       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  202.69    
Total input tokens:                      43560     
Total generated tokens:                  44697     
Request throughput (req/s):              0.99      
Output token throughput (tok/s):         220.52    
Peak output token throughput (tok/s):    403.00    
Peak concurrent requests:                8.00      
Total token throughput (tok/s):          435.44    
---------------Time to First Token----------------
Mean TTFT (ms):                          61.25     
Median TTFT (ms):                        52.50     
P99 TTFT (ms):                           139.46    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.68      
Median TPOT (ms):                        4.54      
P99 TPOT (ms):                           7.43      
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.29     
Median ITL (ms):                         9.79      
P99 ITL (ms):                            30.97     
---------------Speculative Decoding---------------
Acceptance rate (%):                     66.41     
Acceptance length:                       2.33      
Drafts:                                  19180     
Draft tokens:                            38360     
Accepted tokens:                         25475     
Per-position acceptance (%):
  Position 0:                            78.12     
  Position 1:                            54.70     
==================================================
```

### Latency

`main`:

```
Avg latency: 1.2613529006640116 seconds
10% percentile latency: 1.2087979823350907 seconds
25% percentile latency: 1.2460906174965203 seconds
50% percentile latency: 1.2647552555426955 seconds
75% percentile latency: 1.2864032299257815 seconds
90% percentile latency: 1.2968297036364675 seconds
99% percentile latency: 1.4830271347053352 seconds
```

`this branch`:

```
Avg latency: 1.2510471599176527 seconds
10% percentile latency: 1.2067790552973747 seconds
25% percentile latency: 1.2366792568936944 seconds
50% percentile latency: 1.2483715219423175 seconds
75% percentile latency: 1.2593151540495455 seconds
90% percentile latency: 1.2803281325846911 seconds
99% percentile latency: 1.3696827828139067 seconds
```

Main results:
- _Tightened Distribution_: The gap between the 10th and 99th percentiles is narrower on this branch (~0.16s) compared to main (~0.27s), indicating more predictable performance.
- _Significant Tail Improvement_: The P99 latency dropped by 7.64% (over 110ms), which is the most impactful change. This suggests that this change has successfully reduced "jitter" or worst-case outliers.

### Accuracy on GSM8k

`main`:

```
GSM8K Results for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8:
  Measured metric: 0.8749
  Expected metric: 0.8500
  Tolerance: 0.0800
  Questions: 1319
  Invalid rate: 0.000
  Latency: 83.6s
  QPS: 15.8
✅ GSM8K test passed for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```

this `branch`:

```
GSM8K Results for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8:
  Measured metric: 0.8643
  Expected metric: 0.8500
  Tolerance: 0.0800
  Questions: 1319
  Invalid rate: 0.000
  Latency: 88.1s
  QPS: 15.0
✅ GSM8K test passed for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
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

(no discussion comments)