# PR #24266: QWEN3 Coder Fused MoE kernels Optimization configs

- field: quantization  | area: model_executor/layers
- author: samanamp
- state: MERGED  created: 2025-09-04T17:10:29Z  merged: 2025-09-04T20:33:44Z
- url: https://github.com/vllm-project/vllm/pull/24266
- labels: performance, ready, qwen

## Body

## Purpose
Optimize QWEN3 Coder Fused MoE kernels Optimization configs

Execution improvement: 6.5% in `Output token throughput (tok/s)`.  17% in `Median TPOT (ms)` and 5.9x improvement in P99 TPOT (ms).
## Test Plan
1. Single benchmark
```
python benchmarks/kernels/benchmark_moe_old.py --model $MODEL --dtype "fp8_w8a8" --tp 4
```

2. Model E2E
Server side:
```
MODEL=Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
python -m vllm.entrypoints.openai.api_server --model $MODEL --enable-expert-parallel --max-model-len=65000 -tp 4 --port 8101 --gpu_memory_utilization=0.95
```

Bench:
```
python benchmarks/benchmark_serving.py --backend vllm --model $MODEL --dataset-name random --random-input-len 7500 --random-output-len 7500 --max-concurrency 128 --num-prompts 512 --port 8101`
```
## Test Result
1. Single benchmark
<img width="648" height="401" alt="Screenshot 2025-09-04 at 10 01 37 AM" src="https://github.com/user-attachments/assets/da8b7641-9d4f-445f-85ad-a2dd6f3d7a52" />
<img width="647" height="398" alt="Screenshot 2025-09-04 at 10 02 01 AM" src="https://github.com/user-attachments/assets/2e078708-5cb5-4766-a9e7-3b39318e3457" />
<img width="646" height="399" alt="Screenshot 2025-09-04 at 10 02 13 AM" src="https://github.com/user-attachments/assets/740bfab2-a94f-4342-896e-5ae0287adaa4" />

2. Model E2E

BEFORE:
```
============ Serving Benchmark Result ============
Successful requests:                     256       
Maximum request concurrency:             128       
Benchmark duration (s):                  961.48    
Total input tokens:                      1918985   
Total generated tokens:                  1818711   
Request throughput (req/s):              0.27      
Output token throughput (tok/s):         1891.57   
Total Token throughput (tok/s):          3887.42   
---------------Time to First Token----------------
Mean TTFT (ms):                          18056.20  
Median TTFT (ms):                        8936.43   
P99 TTFT (ms):                           56461.27  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          73.51     
Median TPOT (ms):                        63.22     
P99 TPOT (ms):                           490.19    
---------------Inter-token Latency----------------
Mean ITL (ms):                           61.99     
Median ITL (ms):                         55.98     
P99 ITL (ms):                            490.46    
==================================================
```
AFTER:
```
============ Serving Benchmark Result ============
Successful requests:                     256       
Maximum request concurrency:             128       
Benchmark duration (s):                  934.94    
Total input tokens:                      1918985   
Total generated tokens:                  1883193   
Request throughput (req/s):              0.27      
Output token throughput (tok/s):         2014.25   
Total Token throughput (tok/s):          4066.78   
---------------Time to First Token----------------
Mean TTFT (ms):                          18337.78  
Median TTFT (ms):                        8664.39   
P99 TTFT (ms):                           56497.93  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          60.35     
Median TPOT (ms):                        59.73     
P99 TPOT (ms):                           83.55     
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.93     
Median ITL (ms):                         52.33     
P99 ITL (ms):                            489.33    
==================================================
```
We see 6.5% in `Output token throughput (tok/s)`.  17% in `Median TPOT (ms)` and 5.9x improvement in P99 TPOT (ms).

---

- [ x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ x] The test plan, such as providing test command.
- [ x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### github-actions @ 2025-09-04T17:10:40Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### jeejeelee @ 2025-09-05T11:05:56Z

Thank you for your contribution. May I ask if you used triton3.4.0 for the tuning?
---