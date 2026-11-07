# PR #20788: [Meta] Official Eagle mm support, first enablement on llama4

- field: decoding  | area: model_executor/models
- author: morgendave
- state: MERGED  created: 2025-07-10T21:52:21Z  merged: 2025-07-31T17:35:07Z
- url: https://github.com/vllm-project/vllm/pull/20788
- labels: documentation, new-model, speculative-decoding, ready, v1, llama

## Body

## Purpose
Enable MM inference for EAGLE, targeting mllama4 in this PR but generally easy to extend to other models
Issue with this PR:
MM chunked prefill needs to be disabled, or set mbnt to a large number. There will be a follow up PR using unshifting eagle prefill fix this

## Test Plan
```
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py  --num_spec_tokens 7 --num_prompts 1 --method eagle  --model_dir /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --eagle_dir /home/zhiweiz/local/models/scout_draft_HF_20250605_202942 --tp 4 --custom-mm-prompts
```

vllm serve with benchmark testing
cmd
```
#!/bin/bash
# Configuration of environment variables
export VLLM_USE_MODELSCOPE=False
export VLLM_TORCH_PROFILER_DIR=~/vllm_profile
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
# Command to run the vllm server
spec_dec_config='{"method": "eagle", "model": "/home/zhiweiz/local/models/scout_draft_HF_20250605_202942", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "max_model_len": 32768}'
vllm serve /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --disable-log-requests \
    -tp 4 \
    --max-num-seqs 128 \
    --max_num_batched_tokens=80000 \
    --max-model-len=32768 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --speculative-config="$spec_dec_config" \
    --num-lookahead-slots=3 \
    2>&1 | tee /data/users/$USER/logs/server/vllm_17b16e_vllm_serving$(date +%Y%m%d_%H%M%S).log
```
baseline without `--speculative-config` flag

benchmark cmd
```
python benchmarks/benchmark_serving.py --backend openai-chat --model /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --seed 0 --max-concurrency 16 --endpoint /v1/chat/completions 2>&1 | tee /data/users/$USER/tmp/vllm_17b16e_vllm_loadgen$(date +%Y%m%d_%H%M%S).log\n
```

## Test Result
```
--------------------------------------------------
total_num_output_tokens: 155
num_drafts: 45
num_draft_tokens: 315
num_accepted_tokens: 110
mean acceptance length: 3.44
--------------------------------------------------
acceptance at token 0: 0.87
acceptance at token 1: 0.69
acceptance at token 2: 0.47
acceptance at token 3: 0.20
acceptance at token 4: 0.16
acceptance at token 5: 0.04
acceptance at token 6: 0.02
```

Eagle MM benchmark
```
Maximum request concurrency: 16
100%|██████████| 1000/1000 [03:16<00:00,  5.10it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  196.05
Total input tokens:                      87321
Total generated tokens:                  117053
Request throughput (req/s):              5.10
Output token throughput (tok/s):         597.04
Total Token throughput (tok/s):          1042.43
---------------Time to First Token----------------
Mean TTFT (ms):                          201.32
Median TTFT (ms):                        164.53
P99 TTFT (ms):                           797.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.02
Median TPOT (ms):                        24.69
P99 TPOT (ms):                           48.73
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.54
Median ITL (ms):                         40.44
P99 ITL (ms):                            160.04
==================================================
```

```
Maximum request concurrency: 16
100%|██████████| 1000/1000 [04:02<00:00,  4.12it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  242.87
Total input tokens:                      87321
Total generated tokens:                  116716
Request throughput (req/s):              4.12
Output token throughput (tok/s):         480.58
Total Token throughput (tok/s):          840.12
---------------Time to First Token----------------
Mean TTFT (ms):                          187.59
Median TTFT (ms):                        157.39
P99 TTFT (ms):                           826.26
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.00
Median TPOT (ms):                        31.85
P99 TPOT (ms):                           37.72
---------------Inter-token Latency----------------
Mean ITL (ms):                           31.55
Median ITL (ms):                         22.52
P99 ITL (ms):                            131.54
==================================================
```


 ## Follow ups
Need to make offline inference work with vision datasets


## Comments

### github-actions @ 2025-07-10T21:52:29Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### DarkLight1337 @ 2025-07-15T03:19:52Z

Is #20591 supposed to be merged first?
---
### morgendave @ 2025-07-15T17:29:35Z

> Is #20591 supposed to be merged first?

Yes, this would be rebased after that
---
### mergify @ 2025-07-16T04:18:45Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @morgendave.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-07-17T06:37:56Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @morgendave.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### ekagra-ranjan @ 2025-07-21T20:17:54Z

@morgendave - I have a couple of ques based on the benchmark results share in the PR description:
1. The ratio of median TPOT of EAGLE MM vs baseline is 28.9% whereas the AL is high as 3.44. I was wondering if this low speedup is due to the high batch size 16 set here. Do you have any numbers for BS=1 and 4?
2. ITL and TPOT are known to be [different for SD methods](https://github.com/vllm-project/vllm/issues/7242#issuecomment-2373568143). However, I see them different for the baseline here. Do you know the reason?
---
### morgendave @ 2025-07-22T17:53:57Z

> > @morgendave - I have a couple of ques based on the benchmark results share in the PR description:
> > 
> > 1. The ratio of median TPOT of EAGLE MM vs baseline is 28.9% whereas the AL is high as 3.44. I was wondering if this low speedup is due to the high batch size 16 set here. Do you have any numbers for BS=1 and 4?
> > 2. ITL and TPOT are known to be [different for SD methods](https://github.com/vllm-project/vllm/issues/7242#issuecomment-2373568143). However, I see them different for the baseline here. Do you know the reason?

 For 1, I can rerun it but currently this PR is also rebased to the trunk while the new attention metadata builder broke llama4 eagle completely, I'm working with @LucasWilkinson to debug. Also the AL is for dummy data, if we test in benchmark mode (forgot to post), the AL is around 2.4
For 2, I'm actually not entirely sure, how is the ITL being calculated and recorded?


---
### Vasanth51430 @ 2025-07-23T03:20:47Z

How much is it adaptable to change for Qwen2VL like you did for llama eagle @morgendave 
---
### Vasanth51430 @ 2025-07-23T06:00:35Z

also how to test this in my local what llama model should i have to test this can you please confirm that too if you dont mind please @morgendave 
---
### morgendave @ 2025-07-23T20:13:00Z

> How much is it adaptable to change for Qwen2VL like you did for llama eagle

I haven't looked Qwen2VL but as long as it is just embedding in without additional draft based encoder it should work similarly. eagle mm in general needs to overcome the next token embedding issue if it is partial mm prefill but would be addressed in my next PR
> also how to test this in my local what llama model should i have to test this can you please confirm that too if you dont mind please @morgendave

You can use the scout model I put in the e2e test. Specifically:
"meta-llama/Llama-4-Scout-17B-16E-Instruct",
"morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct"


---
### Vasanth51430 @ 2025-08-05T08:09:55Z

Hi congratulations. Is there plan to add support for eagle 2 which incorporates tree based decoding @morgendave ?

I am planning to implement for Qwen2VL will keep you posted
---