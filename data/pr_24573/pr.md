# PR #24573: [EPLB] Reduce EPLB Inference Overhead

- field: parallelism  | area: model_executor/layers
- author: abmfy
- state: MERGED  created: 2025-09-10T10:28:37Z  merged: 2025-09-22T16:31:05Z
- url: https://github.com/vllm-project/vllm/pull/24573
- labels: ready

## Body


## Purpose

PR #18343 introduced the **Expert Parallelism Load Balancer (EPLB)**. By replicating a single logical expert into multiple physical experts, we can achieve better load balancing across experts.

However, this replication introduces some inference-time overhead: after the MoE routing module, we must select among multiple replicas of the same logical expert and also record expert load metrics for the rearrangement algorithm.

Previously, `torch.rand` was used to select expert replicas. Unfortunately, this method is slow and not `torch.compile`-friendly.

In this PR, we aim to **reduce EPLB overhead** by:

1. Switching from `torch.rand` to a **modulo-based pseudo-random selection**.  
   - The pseudo-random method is intentionally simple, based only on the token index and routing rank `k`.
2. Removing unnecessary **masking of local experts**, since we now collect **global physical expert load metrics**.
3. Extracting all EPLB-related logic from `select_experts` into a `torch.compile`-friendly function.

## Test Plan

To isolate EPLB inference overhead, we test with EPLB enabled but with `num_redundant_experts=0`, and **without rearranging experts**. This ensures that any observed differences are solely due to replica selection and load recording overhead.

## Test Result

We benchmarked **1000 random prompts** with **1000 input tokens** and **100 output tokens** on DeepSeek-V3-0324, on a DP16 setting. Prefix caching was disabled to measure the raw computational cost.

```bash
vllm bench serve \
    --model $MODEL \
    --dataset-name random \
    --ignore-eos \
    --port ${PORT:-8080} \
    --random-input-len 1000 \
    --random-output-len 100 \
```

**w/o EPLB:**

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  34.61     
Total input tokens:                      996701    
Total generated tokens:                  100000    
Request throughput (req/s):              28.89     
Output token throughput (tok/s):         2889.16   
Total Token throughput (tok/s):          31685.45  
---------------Time to First Token----------------
Mean TTFT (ms):                          16177.52  
Median TTFT (ms):                        17730.53  
P99 TTFT (ms):                           28279.18  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          184.22    
Median TPOT (ms):                        168.75    
P99 TPOT (ms):                           310.01    
---------------Inter-token Latency----------------
Mean ITL (ms):                           184.22    
Median ITL (ms):                         64.62     
P99 ITL (ms):                            3563.61   
==================================================
```

**w/ EPLB, main:**

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  36.04     
Total input tokens:                      996701    
Total generated tokens:                  100000    
Request throughput (req/s):              27.74     
Output token throughput (tok/s):         2774.48   
Total Token throughput (tok/s):          30427.73  
---------------Time to First Token----------------
Mean TTFT (ms):                          16787.26  
Median TTFT (ms):                        15501.58  
P99 TTFT (ms):                           29449.69  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          192.38    
Median TPOT (ms):                        205.27    
P99 TPOT (ms):                           316.54    
---------------Inter-token Latency----------------
Mean ITL (ms):                           192.38    
Median ITL (ms):                         67.10     
P99 ITL (ms):                            3727.04   
==================================================
```

**w/ EPLB, this PR:**

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  35.19     
Total input tokens:                      996701    
Total generated tokens:                  100000    
Request throughput (req/s):              28.41     
Output token throughput (tok/s):         2841.33   
Total Token throughput (tok/s):          31160.91  
---------------Time to First Token----------------
Mean TTFT (ms):                          16370.60  
Median TTFT (ms):                        15091.95  
P99 TTFT (ms):                           28831.30  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          188.17    
Median TPOT (ms):                        201.00    
P99 TPOT (ms):                           308.62    
---------------Inter-token Latency----------------
Mean ITL (ms):                           188.17    
Median ITL (ms):                         65.13     
P99 ITL (ms):                            3621.91   
==================================================
```

**Summary:**  
Not accounting for the benefits of improved expert load balancing, on the main branch, EPLB introduces ~3.97% throughput drop. With this PR, we recover ~2.41%, narrowing the gap to **~1.66%** compared to running without EPLB.  

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mgoin @ 2025-09-10T21:50:15Z

LGTM - please fix the pre-commit
---