# PR #23569: [Perf][V1] Fully overlap model execution

- field: scheduling  | area: v1/worker
- author: benchislett
- state: MERGED  created: 2025-08-25T16:00:03Z  merged: 2025-09-06T01:20:17Z
- url: https://github.com/vllm-project/vllm/pull/23569
- labels: performance, ready, v1

## Body

## Purpose

This PR allows the model runner to function asynchronously when using async scheduling. This allows full overlap of the cpu operations (including prepare_inputs) and the model forward pass. This diff is functional and does not support speculative decoding, PP, or guided decoding. 

Expected speedup is 5-10% over the current async scheduling.

This PR is pending some light refactoring (see inline comments) and testing but is otherwise ready for review.

## Design Analysis

### Overview 

This PR implements overlapped model execution by allowing the model runner to return a CUDA tensor reference to the sampled token ids, instead of the pythonized token ids. The result is passed via `queue` to an output worker thread which blocks until the value is ready and then places the pythonized result on the main output queue. This way, the model runner can run ahead to handle new inputs before the GPU has finished processing the previous iteration. This results in an elimination of the CPU overhead of input preparation and sampling.

In order to implement this, the `output_token_ids` and `token_ids_cpu` are no longer updated after the sampling step. Instead, a reference to the previous `sampled_token_ids` is kept and the ids are copied into `self.input_ids` during the next step's `prepare_inputs` phase. This means that this approach will not be compatible with ngram speculative decoding (since that requires the output token to be known on the cpu), and will need to be adapted for other speculative decoding (which can [with some modification] receive their inputs from the gpu tensor directly).

### Compatibility with Key Features

Currently, this PR is **not** compatible with:
- Pipeline Parallelism
- Speculative Decoding
- Guided Decoding

I expect that for Speculative Decoding and Guided Decoding, the integration will be straightforward.

For Guided Decoding, an open PR #23224 introduces a refactoring of the structured outputs manager into its own process, allowing the filled bitmask to be received directly by the model runner(s) just before it needs to be applied, enabling overlapped computation of the bitmask.

For Speculative Decoding, work on enabling MLA+MTP #22684 implements a refactor to the speculative decoding runtime to eliminate all gpu->cpu synchronizations. This means that it should nicely integrate into this async execution framework by simply caching the rectangular tensor of sampled token ids which includes the speculated tokens, and copying into `input_ids` in the same manner as this PR already does.

### Drawbacks

This PR's correctness can be enforced by straightforward end-to-end testing with async scheduling enabled. However, it is not so easy to maintain the absence of synchronizations in the execute_model code. Currently, the FlashInfer implementation has two such synchronizations, so Flash Attention is used for benchmarking instead. This is a notable flaw in this design and, if accepted, will require vigilant regression testing for performance degradation due to accidentally introduced synchronization points.

Further, enforcing a fully sync-free execution limits compatibility with features such as n-gram speculative decoding, which inherently require the sampled token id to be serialized to the host. There may be future techniques that limit our ability to effectively maintain zero-synchronizations, and therefore limit the compatibility with this style of async scheduling.

## Profile Results

The following is a snapshot of nsys profile of a decode iteration (BS=1) before and after this change. The setup is Llama 3.2 1B-Instruct on 1xB200 with async scheduling and full cudagraph enabled in both cases.

### Before:
<img width="731" height="390" alt="before-image" src="https://github.com/user-attachments/assets/3c68ca4e-9b51-45b9-bb4b-cbffa1d28552" />

### After:
<img width="731" height="390" alt="after-image" src="https://github.com/user-attachments/assets/364e20b0-2ba4-4c48-9c2f-7fdebe8d8e70" />

To reproduce, run:

```
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve meta-llama/Llama-3.2-1B-Instruct --compilation-config '{"full_cuda_graph": true}' --no-enable-prefix-caching --async-scheduling
```

## Test Plan

Tests are coming soon. See discussion in drawbacks above for test considerations.

## Test Result


## Comments

### mergify @ 2025-08-25T16:01:18Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### lhtin @ 2025-08-28T08:55:21Z

@benchislett Thank you for your excellent patches. We have implemented a similar logic internally based on an earlier vLLM version. May I ask, is it currently compatible if using Ray for startup(i.e. using RayDistributedExecutor)?
---
### benchislett @ 2025-08-28T14:44:18Z

@lhtin I believe vllm's `--async-scheduling` is only compatible with the MP executor. If that is inaccurate, please let me know and I will include this for the ray executor as well. It should be a fairly small patch on top of any executor that already supports async scheduling. 
---
### njhill @ 2025-08-28T22:19:48Z

Something I just thought of ... the executors are intended to be platform agnostic so we might want to move/encapsulate the cuda stream stuff.
---
### JaheimLee @ 2025-08-29T03:17:28Z

@benchislett Hi, can #22262 be compatible with your change? 
---
### lhtin @ 2025-08-29T07:56:07Z

> @lhtin I believe vllm's `--async-scheduling` is only compatible with the MP executor. If that is inaccurate, please let me know and I will include this for the ray executor as well. It should be a fairly small patch on top of any executor that already supports async scheduling.

@benchislett According to my observation, the `--async-scheduling` is compatible with the Ray executor. I just tested it, and it is indeed compatible. This can also be seen from the bellow code:

https://github.com/vllm-project/vllm/blob/2554b27baa58b15843367f92d7f73d71bb89033d/vllm/v1/executor/ray_distributed_executor.py#L57-L64
---
### benchislett @ 2025-08-29T14:46:01Z

@njhill Do you have thoughts on how to handle this? I don't really have the means to develop this feature for other hardwares. I can move the code into a utility file, but that somewhat just moves the goalposts. If async scheduling is to be supported, I think this async copying logic will need to be implemented per-backend.

Could you suggest somewhere that makes sense for this code to live?
---
### benchislett @ 2025-08-29T14:47:27Z

@JaheimLee there is work in the pipeline to fully incorporate speculative decoding with overlapped async execution. #22262 is a first step, but there will need to be special care taken once these PRs are merged. I think it should be scoped for a follow-up PR.
---
### njhill @ 2025-08-29T15:28:21Z

> @njhill Do you have thoughts on how to handle this? I don't really have the means to develop this feature for other hardwares. I can move the code into a utility file, but that somewhat just moves the goalposts. If async scheduling is to be supported, I think this async copying logic will need to be implemented per-backend.
> 
> Could you suggest somewhere that makes sense for this code to live?

@benchislett I will look at it today. It's not that we need to support it rn for other hardware, just that other hw obviously still needs to work and we should abstract the hw specific parts from the executor's pov.

@lhtin in theory this should be compatible with ray executor, I just haven't tested myself or looked to see whether adjustments are needed. This core batch overlap logic is used by both PP and async scheduling, and it might be currently that ray assumes PP in this case. But that's great if it already works.




---
### mergify @ 2025-08-29T16:38:32Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### robertgshaw2-redhat @ 2025-09-03T16:54:00Z

just one QQ about this:
- A) if a user submits a request with guided decoding ... what happens?
- B) if a user starts the server with spec decoding ... what happens?
---
### robertgshaw2-redhat @ 2025-09-03T16:54:48Z

cc @LucasWilkinson re: GPU syncs FYI
---
### benchislett @ 2025-09-03T18:11:42Z

@robertgshaw2-redhat I just checked this:

- A) the request is returned with incorrect outputs. The FSM gets messed up, and there's currently no logic to turn away requests when using guided decoding with async scheduling. It doesn't seem to crash, but I might have just gotten lucky.
- B) `ValueError: Currently, speculative decoding is not supported with async scheduling.`
---
### njhill @ 2025-09-03T18:36:50Z

> A) the request is returned with incorrect outputs. The FSM gets messed up, and there's currently no logic to turn away requests when using guided decoding with async scheduling. It doesn't seem to crash, but I might have just gotten lucky.

Yes similar to spec decoding we should add request validation for this as part of this PR (until we add support).

@benchislett I also realized we should maybe pool/reused the cuda events, I guess that's best practice for performance.

---
### njhill @ 2025-09-04T06:41:23Z

I ran some single-request concurrency benchmarks on H100 `meta-llama/Llama-3.2-1B-Instruct` TP=1.

```
python benchmarks/benchmark_serving.py \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 2000 \
    --ignore-eos \
    --num-prompts 8 \
    --max-concurrency 1 \
    --seed 42
```

TPS is total tokens / second.

| Executor | AsyncSched | Code | TPS | Diff |
| ---------| ----- | ------- |----- | ---- |
| Uni      | No    | main     | 549.85 | +0% |
| Multi      | No    | main   | 549.00 | -0.2% |
| Multi      | Yes    | main   | 568.68 | +3.4% |
| Multi      | Yes    | This PR  | 685.11 | +24.6% |


Interestingly in non-async case there appears to be basically no difference between multi-proc and uni-proc. We could consider making multi-proc the default for all cases.
---
### jiangpeng36 @ 2025-09-04T13:18:53Z

I ran benchmarks on Qwen/Qwen/Qwen3-235B-A22B

```
python -m vllm.entrypoints.openai.api_server --model=$MODEL \
	--trust-remote-code --enforce-eager \
	--distributed-executor-backend=mp \
	-tp=16 \
	--port 8006 \
	--max-model-len 32000 \
	--block-size 128 \
	--gpu-memory-utilization 0.99
```
|AsyncSched|TTFT|TPOT|
|--|--|--|
|No|116|99|
|Yes|200|92|


The TPOT speedup 7.4%, but TTFT is slowdown 72%, why TTFT slowdown so much?
---
### benchislett @ 2025-09-04T13:44:00Z

@jiangpeng36 could you measure on `main` as well? I think that some TTFT hit might be a consequence of async scheduling and not necessarily this PR. If there is a regression introduced by this PR I can investigate. Also, enforce-eager is not recommended for best performance; if using async scheduling then cuda graphs should also be used whenever possible.
---
### benchislett @ 2025-09-04T13:48:03Z

@njhill there are certain use-cases where uniproc has advantages. When I benchmarked VLMs a few months ago, uniproc was a must-have because of the overhead of serializing and sending the input images over IPC.
---
### njhill @ 2025-09-05T18:28:05Z

Running the CI:
- Pipeline parallel test failure looks probably related: https://buildkite.com/vllm/ci/builds/29560#01991a70-e7ff-4b7c-99ee-756d910fad3c
- Other distributed test failures aren't related but we want to fix those so that we can run them on this before merging I think
- Structured output test failure _may_ be related? https://buildkite.com/vllm/ci/builds/29560#01991ad5-316b-409a-8682-b0e2d11d4e9e
---
### benchislett @ 2025-09-05T19:24:19Z

@njhill The PP failure should now be resolved: I was able to reproduce locally and identified that in some cases, using the async output thread when async scheduling is still disabled can cause issues _for PP_, since it will put multiple entries in the work queue causing a desync.

I slightly restructured the logic in `multiproc_executor.py` so that the new output processing is only used when async scheduling is enabled. That fixed the problem for me locally.

I was not able to reproduce the structured output test, it passes for me locally. I can't think of a reason why it would be failing in CI. Let's see if it still fails with this latest commit.
---
### benchislett @ 2025-09-05T20:11:00Z

Guided decoding test also seems to be failing on other PRs, for example:
https://buildkite.com/vllm/ci/builds/29580/steps/canvas?sid=01991b49-879b-401a-a87b-7a71eb6415f8
---
### njhill @ 2025-09-05T21:42:03Z

Are there any other things that aren't (yet) compatible with async scheduling which we should add some startup or per-request validation for? I think we want to add that for structured output right? Would be good to include in this PR.
---
### benchislett @ 2025-09-05T23:14:25Z

@njhill I think we should try to limit the scope of this PR and break up additional work in follow-up PRs. From my POV, this PR is a performance optimization on top of the existing async scheduling feature, so missing UX is important but I don't view it as in-scope for this PR.
---
### njhill @ 2025-09-05T23:28:20Z

> @njhill I think we should try to limit the scope of this PR and break up additional work in follow-up PRs. From my POV, this PR is a performance optimization on top of the existing async scheduling feature, so missing UX is important but I don't view it as in-scope for this PR.

Yep agree, wasn't a big thing and doesn't need to block the PR. The optimization comment was for follow-on anyhow.
---
### woodlgz @ 2025-09-08T08:53:52Z

@benchislett hi, I was reading this code. I wondered if there is any chance current step 's prepare_inputs which conduct inplace update may pollute cpu-gpu copy buffer in those copy_ operations from previous step's prepare_inputs.

Maybe event recorded on previous step's _prepare_inputs operations should also be synchronized in AsyncGPUModelRunnerOutput.get_output method if there's any such event.
---
### njhill @ 2025-09-08T20:35:14Z

@woodlgz these operations are all happening in one cuda stream though so should be serialized.

AsyncGPUModelRunnerOutput and the gpu->cpu copy cuda stream deal only with the `sampled_token_ids` tensor and a new one of these is allocated by the sampler every step.
---
### woodlgz @ 2025-09-09T03:43:43Z

> @woodlgz these operations are all happening in one cuda stream though so should be serialized.
> 
> AsyncGPUModelRunnerOutput and the gpu->cpu copy cuda stream deal only with the `sampled_token_ids` tensor and a new one of these is allocated by the sampler every step.

hi, @njhill , thanks for reply. I got your idea, but my confusion is mainly about prepare_inputs which involves cpu -> gpu copy and reuses cpu buffers (input_ids for example) every time. GPU operations of course will be serialized but what if cpu buffer is rewritten before its related cudaMemcpyAsync (H2D) finished?  this can happen when current step's prepare_inputs host work (involves populating needed cpu buffer) is overlapping with last step's device work (involves cpu->gpu copy and model gpu execution). usually cudaMemcpyAsync (H2D) takes very short time, so it may not be a problem, but there's still risks here, right?
---
### wangqia0309 @ 2025-09-09T05:23:45Z

I found a bug: when asynchronous scheduling is enabled, vLLM crashes if the request includes a frequency_penalty (greater than 0).
```python
/pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:163: operator(): block: [0,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "scatter gather kernel index out of bounds"` failed.
torch.AcceleratorError: CUDA error: device-side assert triggered

Stack trace:
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/penalties.py", line 24, in apply_all_penalties
    return apply_penalties(logits, prompt_token_ids, output_tokens_t,
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 73, in apply_penalties
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 45, in get_token_bin_counts_and_mask
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
```

---
### njhill @ 2025-09-09T17:55:06Z

@woodlgz ah I see what you mean. I think you're right, I've opened a PR to address it, please take a look: https://github.com/vllm-project/vllm/pull/24527

@wangqia0309 async scheduling support is still in progress - it's not yet compatible with various other features but we'll be addressing this soon.
---
### JaheimLee @ 2025-09-11T01:52:05Z

> I found a bug: when asynchronous scheduling is enabled, vLLM crashes if the request includes a frequency_penalty (greater than 0).
> 
> ```python
> /pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:163: operator(): block: [0,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "scatter gather kernel index out of bounds"` failed.
> torch.AcceleratorError: CUDA error: device-side assert triggered
> 
> Stack trace:
>   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/penalties.py", line 24, in apply_all_penalties
>     return apply_penalties(logits, prompt_token_ids, output_tokens_t,
>   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 73, in apply_penalties
>     output_bin_counts, output_mask = get_token_bin_counts_and_mask(
>   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 45, in get_token_bin_counts_and_mask
>     bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
> ```

Same problem. I think it's pretty important because many quantitative models cannot be used without it. @njhill 
---
### njhill @ 2025-09-19T18:55:39Z

@wangqia0309 @JaheimLee could you try again with https://github.com/vllm-project/vllm/pull/25279? It's possible that was the reason for the penalty crash.
---
### JaheimLee @ 2025-09-21T13:58:46Z

> @wangqia0309 @JaheimLee could you try again with #25279? It's possible that was the reason for the penalty crash.

Still has the problem.
---
### wangqia0309 @ 2025-09-22T03:37:29Z

> @wangqia0309 @JaheimLee could you try again with #25279? It's possible that was the reason for the penalty crash.

thanks, I have fixed this issue in house, It resulted from asynchronous scheduling that no longer fills the sample output token and uses -1 instead, so it needs to be asynchronously filled with the correct value in another thread
---
### lhtin @ 2025-10-16T06:49:30Z

@benchislett Currently, [async scheduling] + [ray backend] will crash, can you take a look at the first PR(https://github.com/vllm-project/vllm/pull/25887) to throw a error, and I will submit the second PR to support this feature.
---
### benchislett @ 2025-10-16T14:22:28Z

Thanks @wangqia0309 for identifying the source of the issue.

If there is no PR with a fix by next week I will create one myself. This will be fixed soon. If anyone has a local fix they would like to contribute, that would be welcome.
---
### njhill @ 2025-10-22T02:25:41Z

> > I found a bug: when asynchronous scheduling is enabled, vLLM crashes if the request includes a frequency_penalty (greater than 0).
> > ```python
> > /pytorch/aten/src/ATen/native/cuda/ScatterGatherKernel.cu:163: operator(): block: [0,0,0], thread: [0,0,0] Assertion `idx_dim >= 0 && idx_dim < index_size && "scatter gather kernel index out of bounds"` failed.
> > torch.AcceleratorError: CUDA error: device-side assert triggered
> > 
> > Stack trace:
> >   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/penalties.py", line 24, in apply_all_penalties
> >     return apply_penalties(logits, prompt_token_ids, output_tokens_t,
> >   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 73, in apply_penalties
> >     output_bin_counts, output_mask = get_token_bin_counts_and_mask(
> >   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py", line 45, in get_token_bin_counts_and_mask
> >     bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
> > ```
> 
> Same problem. I think it's pretty important because many quantitative models cannot be used without it. @njhill

@wangqia0309 @JaheimLee you may have already seen but the async scheduling + penalties incompatibility has now been addressed: https://github.com/vllm-project/vllm/pull/26467.
---