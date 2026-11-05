# PR #16937: [V1][Spec Decode] EAGLE-3 Support

- field: decoding  | area: model_executor/models
- author: benchislett
- state: MERGED  created: 2025-04-21T17:54:15Z  merged: 2025-04-25T22:43:07Z
- url: https://github.com/vllm-project/vllm/pull/16937
- labels: documentation, ready, v1

## Body

## Overview

This PR adds EAGLE3 support for llama3 models in the vLLM V1 engine. The main change to existing code is that _auxiliary hidden states must be returned from the target model_. EAGLE3 consumes a concatenation of hidden states extracted from intermediate layers in the target model. This will require modification of any model which wants to support EAGLE3.

## EAGLE3 Model Implementation

The EAGLE reference implementation of EAGLE3 can be found [here](https://github.com/SafeAILab/EAGLE/blob/main/eagle/model/cnets.py). There are a number of changes from previous EAGLE implementations:
- The projection of concatenated [input_embeds, hidden_states] is moved into the attention matrices. Now, the `self.fc` matrix projects the 3x hidden state inputs into a single embedding which is concatenated into [input_embeds, projected_hidden_states]. The final projection to `hidden_dim` happens in the attention as a part of the `QKVParallelLinear` matrices.
- A pair of hidden_states are returned from the EAGLE layer. This is so that the normed value can be used for sampling, and the pre-normalization value can be passed into the next iteration of speculation. An alternative way to handle this is to apply the norm in the compute_logits function. I am indifferent to how we choose to accomplish this.
- Output token mapping must be implemented, as the EAGLE3 lm_head is not the same as the target model and has output dimension 32k which is mapped to the target model vocab size using a simple index mapping (called "d2t" in the EAGLE codebase).

## Acceptance Rate

To generate the acceptance rates, you will need #16367. I merged this at one point for testing, then reverted the diff so it doesn't appear on the PR. For now, you can just check out the previous commit  `git checkout aa11bef1632c345224dda0aa2a56248c5de8ea55`. Alternatively, for latest changes, you can try `git revert 0721cfa` which undoes the revert and returns the experimental diff to the branch.

To run, download the mt_bench data and invoke:
```
VLLM_USE_V1=1 python examples/offline_inference/eagle.py --dataset="./data/mt_bench/question.jsonl" --num_spec_tokens 7 --max_num_seqs 1 --num_prompts 80
```

Here are the results for meta-llama/**Llama-3.1-8B-Instruct** with **yuhuili/EAGLE**-LLaMA3.1-Instruct-8B on my **RTX 4090** (**baseline**):
```
Processed prompts: 100%|...| 80/80 [03:23<00:00,  2.55s/it, est. speed input: 39.49 toks/s, output: 82.47 toks/s]
--------------------------------------------------
mean acceptance length:         2.48
--------------------------------------------------
acceptance at token 0:1.00
acceptance at token 1:0.68
acceptance at token 2:0.39
acceptance at token 3:0.21
acceptance at token 4:0.11
acceptance at token 5:0.06
acceptance at token 6:0.03
acceptance at token 7:0.02
```

Here are the EAGLE3 results with **yuhuili/EAGLE3**-LLaMA3.1-Instruct-8B (**this PR**):
```
Processed prompts: 100%|...| 80/80 [01:56<00:00,  1.46s/it, est. speed input: 68.86 toks/s, output: 143.81 toks/s]
--------------------------------------------------
mean acceptance length:         3.57
--------------------------------------------------
acceptance at token 0:1.00
acceptance at token 1:0.75
acceptance at token 2:0.55
acceptance at token 3:0.42
acceptance at token 4:0.31
acceptance at token 5:0.24
acceptance at token 6:0.18
acceptance at token 7:0.13
```

Notably, the conditional acceptance rate per-token does not seem to decrease. Each token is ~75% likely to be sampled assuming the previous token is accepted. 

Profiling indicates that CUDA graphs and avoiding synchronization in EAGLE code would significantly improve overall performance as drafting time approaches the decode time for increased amounts of speculated tokens. 

## Testing 

I am in the process of updating acceptance tests to evaluate EAGLE3. I hope that code review can be done while test integration is ongoing.

## Comments

### github-actions @ 2025-04-21T17:54:24Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### mergify @ 2025-04-21T17:54:55Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-04-21T19:39:52Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### markmc @ 2025-04-22T21:48:22Z

 
> ## Acceptance Rate
> To generate the acceptance rates, you will need #16367.

Thanks for your patience on this! See #17010 - with this you can get `num_draft_tokens` and `num_accepted_tokens` to calculate the draft acceptance rate. And #16665 provides `num_drafts` for mean acceptance length and `num_accepted_tokens_per_position` for per-position acceptance probabilities.

---
### mergify @ 2025-04-23T14:14:10Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### WoosukKwon @ 2025-04-23T20:32:51Z

@benchislett Also, can you please add correctness tests for EAGLE3?
---
### ekagra-ranjan @ 2025-04-24T16:23:53Z

Thanks for the PR @benchislett . The results you shared in the PR description is using K=7 which would beyond the optimal K. At K=7, the e2e gain in output token/s as 143/82 = 1.74x. I did a [bench](https://github.com/vllm-project/vllm/pull/16035#issuecomment-2790985075) with K=2 and K=4 and EAGLE 1 ~had output token/s as 170 and is faster than EAGLE 3 with K=7~ (I was using H100 and your results are on RTX 4090 so raw numbers are not comparable) . 

Could you pls bench on K=2 and K=4 and share how much faster is EAGLE 3 over EAGLE 1?
---
### benchislett @ 2025-04-24T16:46:49Z

@ekagra-ranjan the branch is functional in its current state. you are welcome to benchmark it yourself, but I think the cost should be the same. The acceptance rate should be the only relevant factor, which you can compute from the K=7 token-wise numbers I listed. There are still a number of optimizations missing from EAGLE in V1 that will lead to best performance, so it is probably too early to measure final numbers.
---
### ekagra-ranjan @ 2025-04-24T17:04:25Z

> but I think the cost should be the same. The acceptance rate should be the only relevant factor

@benchislett - I don't think the cost is the same. The cost to verify tokens would be similar for K=2 or K=7 for lower bs but the the cost to draft tokens will be higher for K=7 compared to K=2 since its an autoregressive process. The earlier bench done on V1 vLLM for Eagle 1 show that K=2 is faster than K=4 end to end even though K=4 has higher AL than K=2. 
---
### benchislett @ 2025-04-24T17:08:26Z

@ekagra-ranjan I mean to say that the cost of evaluating an EAGLE layer should be the same for EAGLE1 vs EAGLE3. So the performance difference of EAGLE1 for some K and EAGLE3 for the same K will be the relative difference in acceptance lengths for that K. Granted, EAGLE3 might have an optimal perf for a K which is higher than that of EAGLE1.
---
### ekagra-ranjan @ 2025-04-24T21:22:26Z

Output tokens/s on H100 BS1

cmd: 
`VLLM_USE_V1=1 python examples/offline_inference/eagle.py --dataset="./data/mt_bench/question.jsonl" --num_spec_tokens 7 --max_num_seqs 1 --num_prompts 80`

vanilla (`meta-llama/Llama-3.1-8B-Instruct`): 133

EAGLE 1 (`yuhuili/EAGLE-LLaMA3.1-Instruct-8B`):
* K=2: 201.69, gain:  1.51x, AL=2.1
* K=4: 198, gain: 1.48x, AL=2.39
* K=7: 169.92, gain: 1.27x, AL=2.47

EAGLE 3 (`yuhuili/EAGLE3-LLaMA3.1-Instruct-8B`):
* K=2: 220.21, 
  * gain: 1.65x, gain wrt EAGLE 1: 1.09x
  * AL=2.36, gain wrt EAGLE 1: 1.123x
* K=4: 228.71
  * gain: 1.72x, gain wrt EAGLE 1: 1.15x
  * AL=3.08, gain wrt EAGLE 1: 1.288x
* K=7: 219.38
  * gain: 1.65x, gain wrt EAGLE 1: 1.29x 
  * AL=3.54, gain wrt EAGLE 1: 1.43x
---
### ekagra-ranjan @ 2025-04-24T22:24:13Z

I just did the [bench](https://github.com/vllm-project/vllm/pull/16937#issuecomment-2828889840) for K=2,4,7 and compared EAGLE1 and EAGLE3. The AL matches with your run.
* In your RTX 4090 run, EAGLE-3 K=7 gives 1.74x more speedup than EAGLE-1. However, it is 1.29x on H100.
* K=4 was better for E3 and K=2 for E1. The ratio of E3/E1 at their optimal K is 1.14x

@benchislett - what are your thoughts on the RTX 4090 vs H100 results and this benchmark?
---
### DarkLight1337 @ 2025-04-25T03:11:51Z

The failing test looks related to this PR, PTAL
---
### benchislett @ 2025-04-25T14:24:27Z

@ekagra-ranjan this seems reasonable. Are you measuring output token throughput, or TPOT? These are not quite the same, I believe output token throughput is averaged over the duration of the whole run so prefill time will be included. There will be a slight difference if we measure one way or the other. I re-iterate that the goal of this PR is to establish initial support, and not to optimize the implementation. There is lots of perf to be gained by refining the EAGLE code, but that is outside the scope of this PR.
---
### benchislett @ 2025-04-25T14:27:04Z

@DarkLight1337 (cc @WoosukKwon) I'm not sure how to address these failing tests. There are two failures:
- The base EAGLE test is failing with 68% matches compared to the baseline of 70%. I think this might just be variance, as I didn't change much about the EAGLE code and did not significantly change the launch configuration in the test.
- The EAGLE3 launch is hitting OOM. The tests passed on my 4090, so I wonder if this is a matter of the pytorch cleanup not freeing memory between tests or some similar artifact.

Besides re-running or tweaking the acceptance conditions for the tests, I'm not sure how to move forward here
---
### ekagra-ranjan @ 2025-04-25T18:50:40Z

>  Are you measuring output token throughput, or TPOT?

@benchislett Thanks for pointing it out. The output tokens/s by running `/offline_inference/eagle.py` is indeed not removing the TTFT. I ran `benchmark_serving.py` to measure the TPOT at BS1 for Llama 3.1 and its 7.5ms which corresponds to 133 tokens/s which is very close to the output tokens/s reported by `/offline_inference/eagle.py`. This is because the input len is small so the numbers I have shared can be treated as TPOT. The TTFT for MTBench is ~12ms so it contributes to <2% even if output len is 100, i.e., [12/(7.5*100)] *100%. 

What are your thoughts on RTX 4090 getting 1.74x gain for E3 over E1 whereas H100 getting only 1.29x? RTX 4090 is getting 1.34x more gains than H100. I tried comparing for the fp16 tensorcore FLOPS/mem BW of RTX 4090 vs H100 hoping that RTX 4090 is relatively more memory bandwidth than H100 but it does seem so. As per [this](https://vast.ai/pricing/gpu/RTX-4090), fp16 tensorcore flops = 330 TFOPS and mem bw is 1 TBPS so the memory bw is just 300x slower than compute flops whereas H100 is 600x slower (1979/3.35). 
---
### WoosukKwon @ 2025-04-25T22:46:14Z

@benchislett @DarkLight1337 I've merged the PR and submitted a new PR #17209 to fix the test (by just lowering the bar).
I think the failure is because we changed the target model from llama 3 to llama 3.1.
---
### lfopensource @ 2025-05-14T17:44:59Z

exciting feature
---
### fan-niu @ 2025-05-22T02:23:57Z

@benchislett Hi this task so cool, i run v0.8.5.post1 Eagle3 broken with llama3-70b, could you help to check this issue? thanks a lot. 

https://github.com/vllm-project/vllm/issues/18452
---