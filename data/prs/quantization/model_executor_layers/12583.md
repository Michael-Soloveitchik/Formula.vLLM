# PR #12583: Expert Parallelism (EP) Support for DeepSeek Models

- field: quantization  | area: model_executor/layers
- author: cakeng
- state: MERGED  created: 2025-01-30T18:24:13Z  merged: 2025-02-24T15:33:22Z
- url: https://github.com/vllm-project/vllm/pull/12583
- labels: ready, v1

## Body

The current vLLM execution only supports TP when running MoE models.

This PR adds support for Expert Parallelism (EP) for the FusedMoE Kernel and DeepSeek V2 model, which should be extendable to V3 and other MoE models as well.

When VLLM_TEST_ENABLE_EP=1, EP is automatically applied on the FusedMoE layers using the user specified tensor_parallel_size as the expert parallel size.

CUDAgraph works with EP.

[Doc](https://docs.google.com/document/d/1XfJJU-3nO8LZwYhBnKMjy5jLShrjZ0NjbrPdR3GasRs/edit?usp=sharing)


## Comments

### github-actions @ 2025-01-30T18:24:25Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### youkaichao @ 2025-02-04T09:23:54Z

need to check the user-interface, but I feel right now we can just reuse the tp size for ep?

in the future, when we have DP, EP size will automatically be DP x TP.
---
### LucasWilkinson @ 2025-02-04T18:42:27Z

> but the problem with CUDA graph seems to be on the current attention layer (MLA?) implementation.

can you please elaborate on this, a bit? MLA + CUDA graphs + TP is working fine on main as far as I am aware 
---
### cakeng @ 2025-02-04T23:25:05Z

@youkaichao The current design support TP within an EP, but we can easily change that to have EP only on MoE layers. I think we will need more discussion with others on that design decision, the current implementation of EP+TP is based on a discussion with @WoosukKwon  and @simon-mo.

@LucasWilkinson I just merged the main branch and CUDA graph is now working with EP+TP.
---
### cakeng @ 2025-02-11T00:36:24Z

Added VLLM_TEST_ENABLE_EP env variable to turn on/off EP, default is turned off (Full TP on FusedMoE layers). When VLLM_TEST_ENABLE_EP=1, EP of TP size will be applied on the FusedMoE layers. 
---
### zarzen @ 2025-02-18T19:29:24Z

Will this PR allow to use EP with TP disabled? 
---
### cakeng @ 2025-02-19T19:10:31Z

@zarzen  By TP disabled you mean without using TP in the MoE layers, right? Yes, the current implementation only uses EP in the MoE layers, when VLLM_TEST_ENABLE_TP is set.
---
### cakeng @ 2025-02-20T21:39:05Z

I think there is a bug in the current commit regarding topk_weights, it should not produce correct results but it is producing correct results so I will check if it is actually a bug and fix it if it is until tonight. 

EDIT: There was a mistake in the test_moe.py script that was making the tests pass when it should fail. It was not a topk_weights bug, but a gating_output.view problem as @yiz-liu mentioned below. It is fixed now.
---
### simon-mo @ 2025-02-21T20:50:29Z

@cakeng can you merge the latest main branch to pick up some fixes for test failures? Thank you! 
---
### simon-mo @ 2025-02-24T07:49:26Z

@cakeng a test failure in CI

https://buildkite.com/vllm/ci/builds/14038#019534e8-1eef-4be1-8e89-3f718502ddb4/2582-4097

```
[2025-02-23T23:32:34Z] FAILED models/test_initialization.py::test_can_initialize[QuantMixtralForCausalLM] - AttributeError: 'FusedMoE' object has no attribute 'num_experts'
```
---
### liweiqing1997 @ 2025-03-07T08:05:32Z

Hello, I would like to inquire about the communication overhead of the current implementation.

 It seems that regardless of whether the MOE's EP (Expert Parallelism) is enabled, each rank will still perform an all-reduce operation on the hidden states, so the amount of data transmitted during communication is the same whether EP is enabled or not. 
 The only difference is the communication latency, which is due to the need for all ranks to synchronize and wait for the MOE computations to finish when EP is enabled. I'm not sure if my analysis is correct.

I noticed that after enabling EP on a 16-card setup, the time taken for the all-reduce kernel has doubled, and the throughput has dropped by 30%.
---
### lewisword @ 2025-03-14T10:16:44Z

May I ask if this feature can be used in a service-oriented way? I see from the example in `examples/offline_inference/data_parallel.py` that it uses an offline multi-process invocation approach. @cakeng 
---
### cakeng @ 2025-03-18T23:43:05Z

@lewisword Yes it should work, but the API to enable EP has been changed to the `--enable-expert-parallel` engine argument (#14305). Also, the EP performance is not necessarily better than TP in the current implementations.
---
### xiuxin121 @ 2025-03-26T06:30:56Z

vllm ep看上去还没有实现通信（dispatch+ combine）和计算的overlap?这在deepseek这样的moe模型中显得很重要。
---
### Neo9061 @ 2025-04-20T03:16:21Z

Can I double check if this feature only works with --enforce-eager?  when I turn this flag off, there is Dynamo related error.

Asking as the last statement of this PR description saying it works with CUDAGraph
---