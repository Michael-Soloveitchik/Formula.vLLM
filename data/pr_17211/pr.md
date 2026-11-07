# PR #17211: [V1][Spec Decode] Apply torch.compile & cudagraph to EAGLE

- field: decoding  | area: model_executor/models
- author: luyuzhe111
- state: MERGED  created: 2025-04-26T00:02:14Z  merged: 2025-04-29T21:10:00Z
- url: https://github.com/vllm-project/vllm/pull/17211
- labels: documentation, speculative-decoding, ready, v1

## Body

Task 8 of https://github.com/vllm-project/vllm/issues/15901

A few notes regarding the implementation.

### Torch.compile 
1. @support_torch_compile is convenient but requires a specific signature for the model. Changes in `vllm/model_executor/models/llama_eagle.py` address this requirement. 
2. Further, to make torch.compile work, we need a separate cache dir for EAGLE model. Without edits in `vllm/compilation/backends.py`, we wouldn't be able to cache EAGLE's compilation properly since the EAGLE module was registered under vllm config from the target model (see [here](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/eagle.py#L226))
3. One notable bug related to torch.compile is this [line](https://github.com/vllm-project/vllm/pull/17211/files#diff-a4809a837fbf535a8f0999b11087a53ec1c53948b50c0a1fe64396bc86de9461R153). Essentially, the default data type for input ids is int32 and the EAGLE model was also compiled with this data type. However, `tensor.argmax()` returns int64 by default. Feeding int64 input ids to the compiled model will completely mess things up and lead to gibberish draft tokens. Currently the compiled model does not even give warnings when the input data type mismatches. Wonder if we can prevent similar bugs in the future by some more checks. 

### CudaGraph
Changes in `vllm/v1/spec_decode/eagle.py` and `vllm/v1/worker/gpu_model_runner.py` are mostly for CudaGraph. Nothing fancy other than registering additional persistent buffers and making sure to use them for EAGLE's forward pass. I do want to mention that with torch.compile & CudaGraph, the EAGLE model's forward pass has been drastically improved (2.5x faster), which makes the small but abundant torch operations look inefficient. Any advice to further optimize these overheads is greatly appreciated.

Finally, it would be great to have https://github.com/vllm-project/vllm/pull/17010 reviewed and merged so that we don't have to pull in other PRs to test the acceptance length. 

Note that the current PR does not directly make torch.compile & cuda graph available for EAGLE3. I think it's worth a separate PR since the work is non-trivial due to the fact that EAGLE-3's input hidden states have dynamic shapes. Maybe @benchislett could help. 

@WoosukKwon @LiuXiaoxuanPKU 

## Comments

### github-actions @ 2025-04-26T00:02:24Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### mergify @ 2025-04-26T00:02:53Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @luyuzhe111.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### WoosukKwon @ 2025-04-26T17:39:53Z

Thanks for the PR. This is so cool!
I’ll take a look, but it would be great if we could also get @youkaichao’s review.
---
### luyuzhe111 @ 2025-04-29T05:05:40Z

Hi @ekagra-ranjan, thanks for this [PR](https://github.com/vllm-project/vllm/pull/17323)! but wondering if you could create a new PR later that makes sure the draft model takes in `vllm_config` instead of `model_config`, if you are interested? This is a requirement for compatibility with `torch.compile` decorator. for this PR I will just create a condition to handle eagle-1 and eagle-3 models separately. thanks! cc @WoosukKwon 
---
### luyuzhe111 @ 2025-04-29T08:09:10Z

> One last thing: Can you please add tests for this?

@WoosukKwon I feel like acceptance length tests are probably the most meaningful tests for this PR. I tested the acceptance length by cherry picking commits from this [PR](https://github.com/vllm-project/vllm/pull/17010). You can see that the difference is less than 0.01. 

With `meta-llama/Llama-3.1-8B-Instruct`, `yuhuili/EAGLE-LLaMA3.1-Instruct-8B`

On MT Bench

When max number generated tokens = 256

| Number of Speculated Tokens | 1 | 2 | 3 | 4 | 5 |
|----------------------------|------|------|------|------|------|
| Eager | 1.71 | 2.09 | 2.30 | 2.38 | 2.43 |
| Compilation & CudaGraph | 1.70 | 2.10 | 2.29 | 2.38 | 2.43 |

I can do a follow-up PR adding acceptance length tests after https://github.com/vllm-project/vllm/pull/17010 is merged.
---
### ekagra-ranjan @ 2025-04-29T16:03:40Z

Thank you @luyuzhe111 for the PR!  
Wondering if you have any benchmark for speedup in output token/s for with and w/o cuda graph and torch compile for EAGLE llama 3.1 on MTBench TP1 BS1?

> I do want to mention that with torch.compile & CudaGraph, the EAGLE model's forward pass has been drastically improved (2.5x faster), which makes the small but abundant torch operations look inefficient. 

This is fantastic! 
Could you share the script you used to measure just the EAGLE's fwd pass? 
By "abundant torch operation", do you mean the torch ops outside the forward pass but within the `propose()` ?
---
### luyuzhe111 @ 2025-04-29T18:25:39Z

Hey @ekagra-ranjan Re: 

1. I remember you had some numbers for benchmarking on mt bench. can you share the setup so that I can run the benchmarking again for torch.compile + cuda graph? thanks!
2. for the forward pass I had to look at the profiler, so no script here. 
3. right I meant those operations that prepare inputs for the EAGLE model. 


---
### ekagra-ranjan @ 2025-04-29T19:46:28Z

@luyuzhe111 - [here](https://github.com/vllm-project/vllm/pull/16937#issuecomment-2828889840) is an example setup for benchmark. Looking fwd to the results!
---
### luyuzhe111 @ 2025-04-29T22:25:54Z

@ekagra-ranjan I got the following results:

Target model: `meta-llama/Llama-3.1-8B-Instruct`
EAGLE model: `yuhuili/EAGLE-LLaMA3.1-Instruct-8B`
Hardware: A100 (40GB)
Script: `VLLM_USE_V1=1 python examples/offline_inference/eagle.py —dataset="./data/mt_bench/question.jsonl" —num_spec_tokens x —max_num_seqs 1 —num_prompts 80`

Regular Decoding OTPS: 72
<img width="478" alt="Screenshot 2025-04-29 at 3 21 29 PM" src="https://github.com/user-attachments/assets/24917f8a-b67a-4e9f-854a-1e59762f4851" />

it looks like a further 10% speedup.
---
### zou3519 @ 2025-05-01T19:38:20Z

> One notable bug related to torch.compile is this [line](https://github.com/vllm-project/vllm/pull/17211/files#diff-a4809a837fbf535a8f0999b11087a53ec1c53948b50c0a1fe64396bc86de9461R153). Essentially, the default data type for input ids is int32 and the EAGLE model was also compiled with this data type. However, tensor.argmax() returns int64 by default. Feeding int64 input ids to the compiled model will completely mess things up and lead to gibberish draft tokens. Currently the compiled model does not even give warnings when the input data type mismatches. Wonder if we can prevent similar bugs in the future by some more checks.

These checks lead to performance degradation so vLLM decided to drop all of them. Some of the lower compilation_levels (e.g. 1) should do the checking but those also have slightly different behavior for other things
---