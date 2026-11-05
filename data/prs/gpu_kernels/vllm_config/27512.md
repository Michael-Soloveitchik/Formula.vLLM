# PR #27512: [PERF] Decouple projections from GDN custom op

- field: gpu_kernels  | area: vllm/config
- author: vadiklyutiy
- state: MERGED  created: 2025-10-25T20:46:47Z  merged: 2025-11-04T16:11:41Z
- url: https://github.com/vllm-project/vllm/pull/27512
- labels: ready, qwen

## Body

## Purpose
This PR is refactoring of GDN. 

The main goal is to allow wider using of `torch.compile`.

1. Separated forward pass of GDN attention into three distinct pieces: Input Projection, Core Attention, Output Projection. Before projections was in the GDN custom op and were not covered by `torch.compile`.
2. Added `RMSNormGated` class that implements torch native gated rmsnorm and use it for GDN. `torch.compile` creates a good code for `RMSNormGated` even better than custom triton kernel used before.


## Functional Test Result
`lm_eval`
Before
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8059|±  |0.0109|
```
After
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8544|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

## Perf Test Result
Server
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_MOE_FP16=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --enable-expert-parallel --no-enable-prefix-caching --async-scheduling --max_cudagraph_capture_size=2048
```

**Prefill** 
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 8192 --random-output 1 --max-concurrency 512 --num-prompt 512 --ignore-eos
```
Before: Total Token throughput (tok/s):          104098.78
After:    Total Token throughput (tok/s):          105270.70
Speedup: **1.1%**

**Decode1**
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 512 --num-prompt 512 --ignore-eos
```
Before: Output token throughput (tok/s):         19212.17
After: Output token throughput (tok/s):         22384.37
Speedup: **16.5%**


**Decode2** 
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 1024 --num-prompt 1024 --ignore-eos
```
Before: Output token throughput (tok/s):         28821.37
After: Output token throughput (tok/s):         30298.90
Speed up: **5.1%**


**Decode3**
Server
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_MOE_FP16=1 vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct -tp 4 --enable-expert-parallel --no-enable-prefix-caching --async-scheduling
```
(without increasing `--max_cudagraph_capture_size`)
```
vllm bench serve --backend vllm --model Qwen/Qwen3-Next-80B-A3B-Instruct --endpoint /v1/completions --dataset-name random --random-input 32 --random-output 1024 --max-concurrency 1024 --num-prompt 1024 --ignore-eos
```

Before: Output token throughput (tok/s):         16586.93
After: Output token throughput (tok/s):         18953.92
Speed up: **14.3%**






## Comments

### ZJY0516 @ 2025-10-27T09:07:50Z

CC @heheda12345 
---
### vadiklyutiy @ 2025-10-28T23:09:58Z

@all
Could you please take a look at this PR

---
### mergify @ 2025-10-30T19:11:58Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @vadiklyutiy.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### vadiklyutiy @ 2025-10-31T00:46:07Z

> LGTM but I'll let someone more familiar with Qwen3 approve

is there someone familiar with Qwen3-next except @sighingnow ?
---
### mgoin @ 2025-10-31T01:58:43Z

cc @tlrmchlsmth 
---
### heheda12345 @ 2025-10-31T06:01:08Z

@codex review
---
### chatgpt-codex-connector @ 2025-10-31T06:06:53Z

Codex Review: Didn't find any major issues. Delightful!

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](http://chatgpt.com/codex/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>
---
### ZJY0516 @ 2025-10-31T10:50:51Z

> looks good! cc @zhiyuan1i we should be able to do similar optimization for kimi linear.

I have made a pr https://github.com/vllm-project/vllm/pull/27871 for kimi. 

---
### vadiklyutiy @ 2025-11-01T16:38:37Z

Regarding CI fails. It looks like they are not related to this PR and a lot of latest commits to fail also has similar fails. 
Is it ok to merge this PR? 
---
### heheda12345 @ 2025-11-01T22:46:25Z

@codex review
---
### chatgpt-codex-connector @ 2025-11-01T22:51:10Z

Codex Review: Didn't find any major issues. Hooray!

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](http://chatgpt.com/codex/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>
---
### vadiklyutiy @ 2025-11-03T20:26:53Z

CI results are weird. I can't repeat them locally. 
Try to disable some parts to localize 
---
### heheda12345 @ 2025-11-03T22:51:52Z

Can you fix the pre-commit and rebase your PR from main? There are many CI fixes on main branch these days.
---
### vadiklyutiy @ 2025-11-04T10:37:13Z

> Can you fix the pre-commit and rebase your PR from main? There are many CI fixes on main branch these days.

Rebased. Still the same and still can't reproduce locally :(
---