# PR #32425: [LoRA] Update LoRA expand kernel heuristic

- field: gpu_kernels  | area: lora/ops
- author: xyang16
- state: MERGED  created: 2026-01-15T18:53:38Z  merged: 2026-01-16T18:38:07Z
- url: https://github.com/vllm-project/vllm/pull/32425
- labels: ready

## Body

<!-- markdownlint-disable -->

## Purpose

This PR is to update the LoRA expand kernel heuristic.

* During profiling we found qkv_proj expand kernel takes more time than shrink kernel. We found it's because layers like `QKVParallelLinearWithLoRA` has num_slices > 1, N (output_size) is smaller in each slice. Therefore, we can set smaller block_n to launch more blocks and increase SM concurrency.
* Profiling shows `_lora_expand_kernel` improves about 30%. (With tuned config, it can achieve the 2x more performance gain.)
* e2e output token throughput improves 1%.

## Test Plan

```
pytest -s -v tests/lora/test_punica_ops.py
```

## Test Result
Unit tests passed.

## Profiling

Main:

<img width="1271" height="374" alt="Screenshot 2026-01-15 at 12 38 52 PM" src="https://github.com/user-attachments/assets/150a12df-d89c-48fc-87a3-47d64bc77d55" />

PR:

<img width="1263" height="369" alt="Screenshot 2026-01-15 at 1 13 01 PM" src="https://github.com/user-attachments/assets/89cbce2b-d80f-4546-b4b0-e1c852a25fb4" />

Main: `_lora_expand_kernel` takes about 14µs.

PR: `_lora_expand_kernel` takes about 10µs.

## Benchmarking

```
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-num-seqs 16 \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8, 16]}' \
  --enable-lora \
  --max-loras 1 \
  --lora-modules lora1=VaarunC/gpt-oss-20b-lora \
  --max-lora-rank 16 \
  --no-enable-prefix-caching
```

```
vllm bench serve \
  --model openai/gpt-oss-20b \
  --lora-modules lora1 \
  --dataset-name sharegpt \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 800 \
  --max-concurrency 16 \
  --num-prompts 1000 \
  --num-warmups 20 \
  --ignore-eos
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  438.70    
Total input tokens:                      226792    
Total generated tokens:                  800000    
Request throughput (req/s):              2.28      
Output token throughput (tok/s):         1823.57   
Peak output token throughput (tok/s):    2016.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2340.53   
---------------Time to First Token----------------
Mean TTFT (ms):                          91.53     
Median TTFT (ms):                        87.47     
P99 TTFT (ms):                           232.26    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.61      
Median TPOT (ms):                        8.63      
P99 TPOT (ms):                           8.91      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.61      
Median ITL (ms):                         8.44      
P99 ITL (ms):                            12.24     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  434.72    
Total input tokens:                      226792    
Total generated tokens:                  800000    
Request throughput (req/s):              2.30      
Output token throughput (tok/s):         1840.27   
Peak output token throughput (tok/s):    2032.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2361.97   
---------------Time to First Token----------------
Mean TTFT (ms):                          108.03    
Median TTFT (ms):                        105.87    
P99 TTFT (ms):                           221.48    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.51      
Median TPOT (ms):                        8.53      
P99 TPOT (ms):                           8.80      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.51      
Median ITL (ms):                         8.40      
P99 ITL (ms):                            9.63      
==================================================
```
e2e output token throughput improves 1%.

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