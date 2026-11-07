# PR #16436: [Core] Speed up decode by remove synchronizing operation in sampler

- field: scheduling  | area: model_executor/layers
- author: chanh
- state: MERGED  created: 2025-04-10T22:31:17Z  merged: 2025-04-21T18:18:23Z
- url: https://github.com/vllm-project/vllm/pull/16436
- labels: ready

## Body

This code inside `apply_penalties` does advanced indexing on a tensor which triggers `nonzero` which requires a CPU sync currently with PyTorch. 

With `torch.cuda.set_sync_debug_mode("warn")` PyTorch framework confirms this:
```
/home/coder/vllm/venv/lib/python3.10/site-packages/torch/cuda/__init__.py:1067: UserWarning: Synchronization debug mode is a prototype feature and does not yet detect all synchronizing operations (Triggered internally at /pytorch/torch/csrc/cuda/Module.cpp:915.)
  torch._C._cuda_set_sync_debug_mode(debug_mode)
/home/coder/vllm/vllm/model_executor/layers/utils.py:52: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:152.)
  logits[logits > 0] /= torch.where(prompt_mask | output_mask,
/home/coder/vllm/vllm/model_executor/layers/utils.py:54: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:152.)
  logits[logits <= 0] *= torch.where(prompt_mask | output_mask,
/home/coder/vllm/vllm/v1/worker/gpu_model_runner.py:1153: UserWarning: called a synchronizing CUDA operation (Triggered internally at /pytorch/c10/cuda/CUDAFunctions.cpp:152.)
  valid_sampled_token_ids = sampled_token_ids.tolist()
```

This seems to be a known issue and was encountered here: https://github.com/pytorch/pytorch/issues/12461

> nonzero that is called in this conversion has a legitimate synchronization - it is necessary to pass the information from the device about how many non-zero elements were found in the boolean index tensor, as this information would be later required on the cpu, to resize the index tensor, and to configure launch parameters/kernel arguments for subsequent kernels. I'm not sure this sync can be avoided, because if mask comes as a result of an operation on the GPU, CPU has no way of getting the number of nonzeros in the mask, which is objectively needed.

By refactoring the code to avoid the indexing, we can remove the sync and allow much more of the sampling phase CPU work to overlap with the forward pass on the GPU, providing an 8% speedup to decoding for smaller models.

Before:
<img width="1584" alt="Screenshot 2025-04-10 at 11 50 24 AM" src="https://github.com/user-attachments/assets/92a2c90f-243f-40a0-8730-eac39e0da632" />

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  103.22    
Total input tokens:                      100000    
Total generated tokens:                  10000     
Request throughput (req/s):              0.97      
Output token throughput (tok/s):         96.88     
Total Token throughput (tok/s):          1065.73   
---------------Time to First Token----------------
Mean TTFT (ms):                          37.21     
Median TTFT (ms):                        32.09     
P99 TTFT (ms):                           71.54     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.74      
Median TPOT (ms):                        6.67      
P99 TPOT (ms):                           7.20      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.74      
Median ITL (ms):                         6.69      
P99 ITL (ms):                            7.93      
==================================================
```

After:
<img width="1438" alt="Screenshot 2025-04-10 at 11 50 38 AM" src="https://github.com/user-attachments/assets/91601c18-a379-469d-9f24-d70124d9d355" />

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  103.17    
Total input tokens:                      100000    
Total generated tokens:                  10000     
Request throughput (req/s):              0.97      
Output token throughput (tok/s):         96.93     
Total Token throughput (tok/s):          1066.19   
---------------Time to First Token----------------
Mean TTFT (ms):                          35.62     
Median TTFT (ms):                        30.71     
P99 TTFT (ms):                           60.89     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.18      
Median TPOT (ms):                        6.11      
P99 TPOT (ms):                           6.50      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.18      
Median ITL (ms):                         6.12      
P99 ITL (ms):                            7.43      
==================================================
```

Benchmark:
```
VLLM_FLASH_ATTN_VERSION=3 VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching --dtype float16 --disable-log-requests -O3

vllm bench serve \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --request-rate 1 \
        --num-prompts 100 \
        --random-input-len 1000 \
        --random-output-len 100 \
        --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
        --ignore-eos
```

## Comments

### github-actions @ 2025-04-10T22:31:26Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---