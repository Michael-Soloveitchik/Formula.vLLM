# PR #14540: [Perf] Improve MLA on V1

- field: attention  | area: v1/attention
- author: simon-mo
- state: MERGED  created: 2025-03-10T05:50:14Z  merged: 2025-03-10T19:06:58Z
- url: https://github.com/vllm-project/vllm/pull/14540
- labels: ready, v1

## Body

This PR helps V1 to _mostly match_ and exceed (in most cases) V0's performance for MLA. Mostly by two things
1. Fix @LucasWilkinson's `rotary_emb` specialization (#14384, #14471, #14476) to reduce CPU overhead.
  * Identified that the cause of 0 GSM8K score comes from the cuda kernel needs the input to be continuous.
  * Fixed it by make the input contiguous if possible. A better fix will be to change the kernel (help wanted). 
2. Reordered some operation in the `build` function, which ended up costing quite a bit overhead in my timing (p99 tail latency up to 1ms) 
  * This is by ensuring there is not GPU -> CPU communication. CPU -> GPU is fine. 


All the following ran in 8xH200. 

## Performance Test (R1)
We are still a bit worse on the short range but we became significantly better on longer range. 64% boost for 6k input. 

VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8
Throughput: 1.09 requests/s, 4342.27 total tokens/s, **1085.57 output tokens/s**

VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8
Throughput: 1.13 requests/s, 4536.67 total tokens/s, **1134.17 output tokens/s**

VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 6000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8
Throughput: 0.87 requests/s, 6060.61 total tokens/s, **865.80 output tokens/s**

VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model /home/vllm-dev/DeepSeek-R1 --load-format dummy --trust-remote-code --input-len 6000 --output-len 1000 --num-prompts 50 --tensor-parallel-size 8
Throughput: 0.53 requests/s, 3692.82 total tokens/s, **527.55 output tokens/s**


## Performance Test (Small)
We are 15% better for small model for 3k input. 

VLLM_USE_V1=1 python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V2-Lite --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 
Throughput: 3.84 requests/s, 15364.27 total tokens/s, **3841.07 output tokens/s**

VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V2-Lite --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 
Throughput: 3.32 requests/s, 13275.67 total tokens/s, **3318.92 output tokens/s**

VLLM_USE_V1=0 python benchmarks/benchmark_throughput.py --model deepseek-ai/DeepSeek-V2-Lite --load-format dummy --trust-remote-code --input-len 3000 --output-len 1000 --num-prompts 50 --enable-chunked-prefill false
Throughput: 3.32 requests/s, 13264.68 total tokens/s, **3316.17 output tokens/s**


## Accuracy Test
No regression.

```
VLLM_USE_V1="1" lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot=5 --limit 100 --log_samples --output_path lmeval-results

vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.65|±  |0.0479|
|     |       |strict-match    |     5|exact_match|↑  | 0.64|±  |0.0482|


VLLM_USE_V1="0" lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384 --task gsm8k --num_fewshot=5 --limit 100 --log_samples --output_path lmeval-results

vllm (pretrained=deepseek-ai/DeepSeek-V2-Lite-Chat,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,trust_remote_code=True,max_model_len=16384), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.66|±  |0.0476|
|     |       |strict-match    |     5|exact_match|↑  | 0.66|±  |0.0476|
```

## Comments

### github-actions @ 2025-03-10T05:50:28Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### ZhongYingMatrix @ 2025-03-13T03:08:44Z

hi @simon-mo Thx for ur great work! Speaking of D2H operation, I notice that `has_context` on [here](https://github.com/vllm-project/vllm/blob/d9f83d62068baa8ac4923cdb45f7945c7d6651a3/vllm/attention/backends/mla/common.py#L1391) would be a single element bool tensor, which incur H2D in following condition operation. Would it has an impact on performance?
cc @LucasWilkinson 
---
### simon-mo @ 2025-03-13T03:15:02Z

good find. Fix welcomed! 
---