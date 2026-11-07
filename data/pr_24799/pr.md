# PR #24799: [Core] Async Scheduling X Spec Decoding Compatibility

- field: scheduling  | area: vllm/config
- author: Ronald1995
- state: MERGED  created: 2025-09-13T07:38:11Z  merged: 2025-11-17T20:16:20Z
- url: https://github.com/vllm-project/vllm/pull/24799
- labels: speculative-decoding, ready, v1

## Body

## Purpose
PR #19970  implements async_scheduling, PR #23569 implement `prepare_input` overlap base on PR #19970.  RP #24539 refactor the logic of eagle spec_code， make it don't rely on cpu's sample_token_ids.

this PR is based on  #24539 , and aims to support spec decode with async_scheduling. when enable both async_scheduling and spec decode, we won't copy draft token ids to scheduler any more, but cache it in gpu_model_runner, and update the input_ids with the `_draft_token_ids`  directly for next step `execute_model`.

because ngram and medusa rely on cpu's sample_token_ids now, maybe we could refactor it in the future, but now this PR
only support eagle spec_decode with async_scheduling.
## Test Plan
we will make the e2e test.
- async_scheduling + EAGLE-LLaMA3-Instruct-8B draft model, make sure it works well.

Test config:
```python
# dataset is prm800k, read the jsonl and make prompts.
sampling_params = SamplingParams(temperature=0, max_tokens=1024)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=2048,
    max_num_seqs=128,
    max_num_batched_tokens=4096,
    async_scheduling=True, 
    speculative_config={
            "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 2,
            "method": "eagle",
        },
    seed=1234
)
```

test device: Nvidia A100

## Test Result
### performance
|num_prompts|async_scheduling(tps)|sync_scheduling(tps)|speedup|
|--|--|--|--|
|24|2356|2314|1.8%|
|48|3759|3539|6.2%|
|96|5110|4770|7.1%|

### precision
I compare the outputs of async_scheduling and sync_scheduling with speculative decoding, 
the outputs are actually the same. so the async_scheduling doesn't make precision problem.

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

### mergify @ 2025-09-13T07:38:53Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-18T05:41:00Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Ronald1995 @ 2025-09-21T13:35:58Z

@njhill @WoosukKwon   I have made a comprehensive test and test result is reported, would you please review this PR, thanks!
---
### Ronald1995 @ 2025-09-23T11:00:54Z

@robertgshaw2-redhat @njhill @WoosukKwon @benchislett  I'm looking forward to your review, thanks!
---
### mergify @ 2025-10-03T14:04:03Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Ronald1995 @ 2025-10-03T17:09:44Z

@benchislett thanks for your reviews, i have replied to all your reviews and make code changes according to your suggestions,.
if you have any other suggestions, please let me know, thanks!
---
### benchislett @ 2025-10-03T19:17:39Z

@Ronald1995 FYI, I think there are still two unresolved comments. Maybe you didn't push all the changes?
- https://github.com/vllm-project/vllm/pull/24799#discussion_r2399365483
- https://github.com/vllm-project/vllm/pull/24799#discussion_r2399347925
---
### Ronald1995 @ 2025-10-04T02:18:54Z

> @Ronald1995 FYI, I think there are still two unresolved comments. Maybe you didn't push all the changes?
> 
> * [[Core] Async Scheduling X Spec Decoding Compatibility #24799 (comment)](https://github.com/vllm-project/vllm/pull/24799#discussion_r2399365483)
> * [[Core] Async Scheduling X Spec Decoding Compatibility #24799 (comment)](https://github.com/vllm-project/vllm/pull/24799#discussion_r2399347925)

@benchislett the changes about these two comments isn't pushed by mistake, i re-push it, please review it again, thanks.
---
### mergify @ 2025-10-07T02:25:29Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### elvischenv @ 2025-10-07T08:14:03Z

Seems this PR broke the original `--async-scheduling` on B200:
```
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 vllm serve openai/gpt-oss-120b --async-scheduling
```
```
vllm bench serve --model openai/gpt-oss-120b --dataset-name random --ignore-eos --max-concurrency 1 --num-prompts 10 --random-input-len 1024 --random-output-len 1024 
```
Error:
```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] WorkerProc hit an exception.
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Traceback (most recent call last):
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2457, in synchronize_input_prep
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     yield
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2520, in execute_model
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     use_cascade_attn) = self._prepare_inputs(scheduler_output)
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1277, in _prepare_inputs
(Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self._prepare_input_ids(
(Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1066, in _prepare_input_ids
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self.input_ids.gpu.scatter_(
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] torch.AcceleratorError: CUDA error: device-side assert triggered
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
(Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```
cc @nvpohanh 
---
### Ronald1995 @ 2025-10-08T08:03:59Z

> Seems this PR broke the original `--async-scheduling` on B200:
> 
> ```
> VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 vllm serve openai/gpt-oss-120b --async-scheduling
> ```
> 
> ```
> vllm bench serve --model openai/gpt-oss-120b --dataset-name random --ignore-eos --max-concurrency 1 --num-prompts 10 --random-input-len 1024 --random-output-len 1024 
> ```
> 
> Error:
> 
> ```
> /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] WorkerProc hit an exception.
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Traceback (most recent call last):
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2457, in synchronize_input_prep
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     yield
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2520, in execute_model
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     use_cascade_attn) = self._prepare_inputs(scheduler_output)
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1277, in _prepare_inputs
> (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self._prepare_input_ids(
> (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1066, in _prepare_input_ids
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self.input_ids.gpu.scatter_(
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] torch.AcceleratorError: CUDA error: device-side assert triggered
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> ```
> 
> cc @nvpohanh

maybe there are some bugs after i make modifications according to PR reviews, because i'm on National Day holiday in china, i don't have any device to validate the modified code, i will debug this issue tomorrow and report the result, thanks for you issue @elvischenv 
---
### hmellor @ 2025-10-08T09:18:27Z

These conflicts are caused by our migration to `ruff`. Please see https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1759663228844749 which contains detailed instructions to make updating your branch as painless as possible.
---
### Ronald1995 @ 2025-10-09T12:44:50Z

> These conflicts are caused by our migration to `ruff`. Please see https://vllm-dev.slack.com/archives/C07R5Q1Q2BB/p1759663228844749 which contains detailed instructions to make updating your branch as painless as possible.

ok, thanks for your information
---
### Ronald1995 @ 2025-10-10T14:21:01Z

> > Seems this PR broke the original `--async-scheduling` on B200:
> > ```
> > VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 vllm serve openai/gpt-oss-120b --async-scheduling
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > ```
> > vllm bench serve --model openai/gpt-oss-120b --dataset-name random --ignore-eos --max-concurrency 1 --num-prompts 10 --random-input-len 1024 --random-output-len 1024 
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > Error:
> > ```
> > /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] WorkerProc hit an exception.
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Traceback (most recent call last):
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2457, in synchronize_input_prep
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     yield
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 2520, in execute_model
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     use_cascade_attn) = self._prepare_inputs(scheduler_output)
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1277, in _prepare_inputs
> > (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self._prepare_input_ids(
> > (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671]   File "/workspace/vllm/vllm/v1/worker/gpu_model_runner.py", line 1066, in _prepare_input_ids
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671]     self.input_ids.gpu.scatter_(
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] torch.AcceleratorError: CUDA error: device-side assert triggered
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> > (Worker_TP3 pid=1178) ERROR 10-07 06:43:36 [multiproc_executor.py:671] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> > (Worker_TP2 pid=1177) ERROR 10-07 06:43:36 [multiproc_executor.py:671] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > cc @nvpohanh
> 
> maybe there are some bugs after i make modifications according to PR reviews, because i'm on National Day holiday in china, i don't have any device to validate the modified code, i will debug this issue tomorrow and report the result, thanks for you issue @elvischenv

@elvischenv  i have fixed this issue, would you please try it again.
---
### hmellor @ 2025-10-10T14:45:49Z

I'm not the right person to review the speculative decoding part of the PR, I'll leave that to the codeowners (@benchislett @luccafong)
---
### Ronald1995 @ 2025-10-11T01:33:07Z

@benchislett do you have any other suggestions? or would you please approve this PR, thanks!
---
### vadiklyutiy @ 2025-10-13T13:37:21Z

@Ronald1995 
just curious what tested with this patch and how stable is it? Have one example to try.
---
### Ronald1995 @ 2025-10-14T01:08:25Z

> @Ronald1995 just curious what tested with this patch and how stable is it? Have one example to try.

i test this patch with meta-llama/Meta-Llama-3-8B-Instruct model and  yuhuili/EAGLE-LLaMA3-Instruct-8B, the test config is:
```python
sampling_params = SamplingParams(temperature=0, max_tokens=1024)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
    max_model_len=2048,
    max_num_seqs=128,
    max_num_batched_tokens=4096,
    async_scheduling=True, 
    speculative_config={
            "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
            "draft_tensor_parallel_size": 1,
            "num_speculative_tokens": 2,
            "method": "eagle",
        },
    seed=1234
)
```
---
### benchislett @ 2025-10-14T19:12:33Z

I just noticed an issue with degraded acceptance length when running DSR1+MTP for testing. I will update when I have more information
---
### benchislett @ 2025-10-14T20:15:14Z

To reproduce accuracy issues for DSR1:

I ran the following command on this branch on 8xB200:
```
VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve deepseek-ai/DeepSeek-R1-0528 -tp 8 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```

and benchmarked with MTBench:
```
vllm bench serve --model deepseek-ai/DeepSeek-R1-0528 --dataset-name hf --dataset-path philschmid/mt-bench --num-prompts 80 --max-concurrency 1 --port 8049
```

I got these results:
```
============ Serving Benchmark Result ============
Successful requests:                     80        
Maximum request concurrency:             1         
Benchmark duration (s):                  121.74    
Total input tokens:                      5535      
Total generated tokens:                  20463     
Request throughput (req/s):              0.66      
Output token throughput (tok/s):         168.09    
Peak output token throughput (tok/s):    70.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          213.55    
---------------Time to First Token----------------
Mean TTFT (ms):                          33.21     
Median TTFT (ms):                        31.58     
P99 TTFT (ms):                           49.46     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.84      
Median TPOT (ms):                        5.61      
P99 TPOT (ms):                           7.65      
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.45     
Median ITL (ms):                         14.45     
P99 ITL (ms):                            14.78     
==================================================
```
here's a sample of the logged acceptance metrics:
```
(APIServer pid=3303321) INFO 10-14 20:04:06 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 2.51, Accepted throughput: 102.39 tokens/s, Drafted throughput: 203.38 tokens/s, Accepted: 1024 tokens, Drafted: 2034 tokens, Per-position acceptance rate: 0.791, 0.473, 0.246, Avg Draft acceptance rate: 50.3%
```

Then I reran with `--async-scheduling` and got this:
```
============ Serving Benchmark Result ============
Successful requests:                     80        
Maximum request concurrency:             1         
Benchmark duration (s):                  144.31    
Total input tokens:                      5535      
Total generated tokens:                  20415     
Request throughput (req/s):              0.55      
Output token throughput (tok/s):         141.47    
Peak output token throughput (tok/s):    72.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          179.82    
---------------Time to First Token----------------
Mean TTFT (ms):                          40.36     
Median TTFT (ms):                        39.25     
P99 TTFT (ms):                           55.08     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.93      
Median TPOT (ms):                        6.81      
P99 TPOT (ms):                           9.04      
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.02     
Median ITL (ms):                         13.98     
P99 ITL (ms):                            19.92     
==================================================
```

with fewer accepted tokens:
```
(APIServer pid=3342866) INFO 10-14 20:12:31 [metrics.py:100] SpecDecoding metrics: Mean acceptance length: 1.94, Accepted throughput: 65.60 tokens/s, Drafted throughput: 209.39 tokens/s, Accepted: 656 tokens, Drafted: 2094 tokens, Per-position acceptance rate: 0.560, 0.262, 0.117, Avg Draft acceptance rate: 31.3%
```

---
### vadiklyutiy @ 2025-10-15T23:50:38Z

I tested performance of this patch on Qwen3-Next model, tp=1, B200, batch=1, 3 toks prediction and see improvement from 217.7 toks/s to 231.8 toks/s (+6.5%)
---
### Ronald1995 @ 2025-10-16T00:31:34Z

> I tested performance of this patch on Qwen3-Next model, tp=1, B200, batch=1, 3 toks prediction and see improvement from 217.7 toks/s to 231.8 toks/s (+6.5%)

@vadiklyutiy thanks for your result. 
@benchislett qwen3-next and deepseek-r1 are both using mtp model, i think all eagle types of speculative model use the same `propose_token_ids` function in `gpu_model_runner`, so they should have the same behaviour, but the result you shows mean there is issue of deepseek-r1 mtp model, it's odd to me and i will debug for this.
---
### benchislett @ 2025-10-16T14:24:54Z

@Ronald1995 I think it might be related to the larger model causing a rare race condition more than it would be due to an MTP-specific difference, for the reasons you identified. But I have no concrete information on the cause of this regression besides the AR discrepancy issue I measured.
---
### Ronald1995 @ 2025-10-16T15:27:11Z

> @Ronald1995 I think it might be related to the larger model causing a rare race condition more than it would be due to an MTP-specific difference, for the reasons you identified. But I have no concrete information on the cause of this regression besides the AR discrepancy issue I measured.

@benchislett ok, i have fixed issues you reviewed recently and made explanations to the questions.

as for this issue, you reminds me that you set `--max-concurrency 1` for bench client, PR #19970 shows the performance speedup is positive correlation to scheduled requests, because async_scheduler have two more threads and extra `prepare_input_ids` operations, it will make a performance loss, if the performance speedup is smaller than performance loss, it's possible that final performance is regressed,  especially for larger model, because the model forward time is longer, the performance speedup of async_scheduling is relatively smaller.

the `Total Token throughput` metric is regressed of deepseek-r1 when `--max-concurrency 1` in async_scheduling  could be explained, if we promote max-concurrency value, the metric is expected to be promoted, but `Avg Draft acceptance rate` is also regressed, it make me confused now, i will debug it and report the result later.
---
### Ronald1995 @ 2025-10-16T16:53:14Z

> > @Ronald1995 I think it might be related to the larger model causing a rare race condition more than it would be due to an MTP-specific difference, for the reasons you identified. But I have no concrete information on the cause of this regression besides the AR discrepancy issue I measured.
> 
> @benchislett ok, i have fixed issues you reviewed recently and made explanations to the questions.
> 
> as for this issue, you reminds me that you set `--max-concurrency 1` for bench client, PR #19970 shows the performance speedup is positive correlation to scheduled requests, because async_scheduler have two more threads and extra `prepare_input_ids` operations, it will make a performance loss, if the performance speedup is smaller than performance loss, it's possible that final performance is regressed, especially for larger model, because the model forward time is longer, the performance speedup of async_scheduling is relatively smaller.
> 
> the `Total Token throughput` metric is regressed of deepseek-r1 when `--max-concurrency 1` in async_scheduling could be explained, if we promote max-concurrency value, the metric is expected to be promoted, but `Avg Draft acceptance rate` is also regressed, it make me confused now, i will debug it and report the result later.

@benchislett i find bench server print many lines of logged acceptance metrics test, and they have irregular changes, i think the log you show may not prove there are accuracy issues. i compare the output content for sync scheduling and async scheduling with prm800k_500 dataset.
- `Meta-Llama-3-8B-Instruct` : eagle method, the output content are actually the same.
- `DeepSeek-V3-4layers-MTP-FP8`: mtp method, the output content are actually the same.
so i believe this pr don't make accuracy issues, as for performance loss, as i said, it's possible when `--max-concurrency 1` for larger model, if we promote `max-concurrency`, it will gain performance speedup.
---
### benchislett @ 2025-10-16T20:03:16Z

@Ronald1995 I think you are misunderstanding the issue. The problem appears to be that draft tokens are not being generated (or received) properly. The verification code is fine, but fewer tokens are accepted when using this feature (async sched + spec) than without (only spec). Running the same experiment with the flag on/off, I should see (almost) exactly the same number of drafted and accepted tokens. Instead, I get the following data (from my prev post):
```
Accepted: 1024 tokens, Drafted: 2034 tokens # Without async sched
Accepted: 656 tokens, Drafted: 2094 tokens # With async sched
```
This is not just a performance issue. It means that the draft tokens are getting rejected too often. For example, if there is a race condition and the verification buffer is not filled in time, some tokens in the input might not be updated in time and the verification could reject more readily. I think I have shown sufficient evidence to believe there is an issue here.
---
### benchislett @ 2025-10-16T20:24:03Z

As you can see from the benchmark logs I posted, the engine iteration is actually observably faster when running with async scheduling:
```
Mean ITL (ms):                           14.45     
Median ITL (ms):                         14.45     
P99 ITL (ms):                            14.78   
...
Mean ITL (ms):                           14.02     
Median ITL (ms):                         13.98     
P99 ITL (ms):                            19.92     
```
but the TPOT is slower, due to fewer tokens being accepted:
```
Mean TPOT (ms):                          5.84      
Median TPOT (ms):                        5.61      
P99 TPOT (ms):                           7.65    

Mean TPOT (ms):                          6.93      
Median TPOT (ms):                        6.81      
P99 TPOT (ms):                           9.04
```
---
### Ronald1995 @ 2025-10-17T08:42:16Z

> @Ronald1995 I think you are misunderstanding the issue. The problem appears to be that draft tokens are not being generated (or received) properly. The verification code is fine, but fewer tokens are accepted when using this feature (async sched + spec) than without (only spec). Running the same experiment with the flag on/off, I should see (almost) exactly the same number of drafted and accepted tokens. Instead, I get the following data (from my prev post):
> 
> ```
> Accepted: 1024 tokens, Drafted: 2034 tokens # Without async sched
> Accepted: 656 tokens, Drafted: 2094 tokens # With async sched
> ```
> 
> This is not just a performance issue. It means that the draft tokens are getting rejected too often. For example, if there is a race condition and the verification buffer is not filled in time, some tokens in the input might not be updated in time and the verification could reject more readily. I think I have shown sufficient evidence to believe there is an issue here.

ok, i got your point, i will reproduce your test to debug it.
---
### Ronald1995 @ 2025-10-20T15:06:41Z

@benchislett  i have made some tests to reproduce your test result, here are my test results.
- use your original config
  result is the same as yours. async_scheduling has lower ITL but higher TPOT.
- set num_speculative_tokens = 1
server:
```shell 
VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve deepseek-ai/DeepSeek-R1-0528 -tp 8 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```
client script is the same as yours.
async_scheduling result:
```text
============ Serving Benchmark Result ============
Successful requests:                     80        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  307.59    
Total input tokens:                      5535      
Total generated tokens:                  20375     
Request throughput (req/s):              0.26      
Output token throughput (tok/s):         66.24     
Peak output token throughput (tok/s):    39.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          84.24     
---------------Time to First Token----------------
Mean TTFT (ms):                          89.46     
Median TTFT (ms):                        77.82     
P99 TTFT (ms):                           197.69    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.80     
Median TPOT (ms):                        14.51     
P99 TPOT (ms):                           16.77     
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.45     
Median ITL (ms):                         26.42     
P99 ITL (ms):                            27.84     
==================================================
```
sync_scheduling:
```text
============ Serving Benchmark Result ============
Successful requests:                     80        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  316.42    
Total input tokens:                      5535      
Total generated tokens:                  20375     
Request throughput (req/s):              0.25      
Output token throughput (tok/s):         64.39     
Peak output token throughput (tok/s):    37.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          81.88     
---------------Time to First Token----------------
Mean TTFT (ms):                          74.65     
Median TTFT (ms):                        62.14     
P99 TTFT (ms):                           220.20    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          15.29     
Median TPOT (ms):                        15.00     
P99 TPOT (ms):                           17.32     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.33     
Median ITL (ms):                         27.31     
P99 ITL (ms):                            28.01     
==================================================
```
in this config, both ITL and TPOT are speedup by use async_scheduling, ITL speedup 3.3%, TPOT speedup 3.3%
- set num_speculative_tokens = 3 but disable cudagraph of DeepSeekMtp implementd by #25109 
```python
#@support_torch_compile
class DeepSeekMTP(nn.Module, SupportsPP):
```
server:
```shell 
VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve deepseek-ai/DeepSeek-R1-0528 -tp 8 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```
client script is the same as yours.
async_scheduling result:
```text
============ Serving Benchmark Result ============
Successful requests:                     80        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  259.19    
Total input tokens:                      5535      
Total generated tokens:                  20375     
Request throughput (req/s):              0.31      
Output token throughput (tok/s):         78.61     
Peak output token throughput (tok/s):    34.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          99.97     
---------------Time to First Token----------------
Mean TTFT (ms):                          105.27    
Median TTFT (ms):                        85.59     
P99 TTFT (ms):                           475.04    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.35     
Median TPOT (ms):                        11.91     
P99 TPOT (ms):                           16.17     
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.40     
Median ITL (ms):                         30.20     
P99 ITL (ms):                            48.17     
==================================================
```
sync_scheduling result:
```text
============ Serving Benchmark Result ============
Successful requests:                     80        
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  270.94    
Total input tokens:                      5535      
Total generated tokens:                  20375     
Request throughput (req/s):              0.30      
Output token throughput (tok/s):         75.20     
Peak output token throughput (tok/s):    32.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          95.63     
---------------Time to First Token----------------
Mean TTFT (ms):                          81.04     
Median TTFT (ms):                        63.58     
P99 TTFT (ms):                           404.71    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.02     
Median TPOT (ms):                        12.57     
P99 TPOT (ms):                           17.34     
---------------Inter-token Latency----------------
Mean ITL (ms):                           32.00     
Median ITL (ms):                         32.01     
P99 ITL (ms):                            32.81     
==================================================
```
in this config, both ITL and TPOT are speedup by use async_scheduling, ITL speedup 5.3%, TPOT speedup 5.4%
- set num_speculative_tokens = 3 and set enforce_eager=True
server:
```shell 
VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP8=1 vllm serve deepseek-ai/DeepSeek-R1-0528 -tp 8 --max-model-len 8192 --enforce-eager --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
```
client script is the same as yours.
async_scheduling result:
```text
============ Serving Benchmark Result ============
Successful requests:                     10        
Maximum request concurrency:             1         
Benchmark duration (s):                  174.77    
Total input tokens:                      789       
Total generated tokens:                  2560      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         14.65     
Peak output token throughput (tok/s):    7.00      
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          19.16     
---------------Time to First Token----------------
Mean TTFT (ms):                          929.44    
Median TTFT (ms):                        316.35    
P99 TTFT (ms):                           3583.42   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          64.89     
Median TPOT (ms):                        60.77     
P99 TPOT (ms):                           82.47     
---------------Inter-token Latency----------------
Mean ITL (ms):                           161.43    
Median ITL (ms):                         158.79    
P99 ITL (ms):                            225.90    
==================================================
````
sync_scheduling result:
```text
============ Serving Benchmark Result ============
Successful requests:                     10        
Maximum request concurrency:             1         
Benchmark duration (s):                  168.98    
Total input tokens:                      789       
Total generated tokens:                  2560      
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         15.15     
Peak output token throughput (tok/s):    7.00      
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          19.82     
---------------Time to First Token----------------
Mean TTFT (ms):                          622.80    
Median TTFT (ms):                        161.48    
P99 TTFT (ms):                           1925.25   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          63.82     
Median TPOT (ms):                        59.06     
P99 TPOT (ms):                           85.40     
---------------Inter-token Latency----------------
Mean ITL (ms):                           158.78    
Median ITL (ms):                         158.38    
P99 ITL (ms):                            219.83    
==================================================
```
There is performance loss when enable async_scheduling with max_concurrency=1, i have tested if increase max_concurrency, async_scheduling  will speedup. the point here is that if disable cudagraph, it won't occur ITL is lower but TOPT is higher.

I guess there are hidden bugs for cudagraph with DeepseekMTP, i need to speed more time to figure it out. But as for this PR, i have made a lot of tests, i think the implementation itself of async_scheduling with spec decoding is fine. 

I will add assertion in code to make sure when use async_scheduling and deepseek_mtp, num_speculative_tokens should less equal than 1 and add `TODO` to fix this issue in another PR. By doing this, i hope you could merge this PR first, please let me know what you think, thanks!
---
### benchislett @ 2025-10-20T15:22:00Z

@Ronald1995 I am not fully convinced that this issue is resolved. I investigated further last week and I am still able to consistently reproduce the issue on blackwell. Adding a `torch.cuda.synchronize()` into the `gpu_model_runner.execute_model` code almost anywhere will alleviate the issue. As such I suspect there might be some problems overlapping the draft model prepare_inputs and the next iteration's prepare_inputs. I will take a closer look today and inspect the individual data structures to see if there is any problem. 

If the EAGLE prepare_inputs and main model's prepare_inputs share any cpu-side data, I believe it might be possible that one of them could overwrite this data while the other has an async HtoD memcpy in-flight, leading to a race condition. We have an event in the main model's prepare_inputs to ensure that this does not happen between iterations of the main model, but there is intentionally no safeguard for this in the spec decoding PR. I will validate if this is the cause of the issue I am seeing, and investigate if so.

Otherwise, I am happy with the state of the PR and am hoping it can be merged this week. Thank you for your continued effort!
---
### benchislett @ 2025-10-20T22:23:20Z

@Ronald1995 I have confirmed the issue and propose the following patch:

In `gpu_model_runner.py:2908` (propose_draft_token_ids):

```
        elif self.speculative_config.use_eagle():
            if self.prepare_inputs_event is not None: # new
                self.prepare_inputs_event.synchronize() # new
            assert isinstance(self.drafter, EagleProposer)
```

This enforces a synchronization between prepare_inputs of the base model and the EAGLE drafter. With this patch, there can be no overlap between the draft model's prepare_inputs cpu execution and the gpu execution of the target model's prepare_inputs _for the same step_. It is still able to overlap between iterations, and profiling on nsight systems for Llama 3.1 8B with EAGLE3 indicates that this is not a significant block. CPU-GPU overlapping still occurs during the target model's forward pass. You are welcome to benchmark and profile with patch if you are suspicious of the performance implications.

While it would be nice to have identified a specific problematic shared buffer that can simply be replicated, this patch will serve in the meantime to alleviate the issue until the root cause can be identified. Even then, it might be beneficial to have this as a sanity check.
---
### Ronald1995 @ 2025-10-21T01:35:59Z

> @Ronald1995 I have confirmed the issue and propose the following patch:
> 
> In `gpu_model_runner.py:2908` (propose_draft_token_ids):
> 
> ```
>         elif self.speculative_config.use_eagle():
>             if self.prepare_inputs_event is not None: # new
>                 self.prepare_inputs_event.synchronize() # new
>             assert isinstance(self.drafter, EagleProposer)
> ```
> 
> This enforces a synchronization between prepare_inputs of the base model and the EAGLE drafter. With this patch, there can be no overlap between the draft model's prepare_inputs cpu execution and the gpu execution of the target model's prepare_inputs _for the same step_. It is still able to overlap between iterations, and profiling on nsight systems for Llama 3.1 8B with EAGLE3 indicates that this is not a significant block. CPU-GPU overlapping still occurs during the target model's forward pass. You are welcome to benchmark and profile with patch if you are suspicious of the performance implications.
> 
> While it would be nice to have identified a specific problematic shared buffer that can simply be replicated, this patch will serve in the meantime to alleviate the issue until the root cause can be identified. Even then, it might be beneficial to have this as a sanity check.

@benchislett Thanks for your information,  i will make more performance about your advised patch and try to identify the a specific problematic shared buffer.
---
### Ronald1995 @ 2025-10-21T01:37:10Z

> eantime, please add some validation that will raise a warning/error if "disable_padded_drafter_batch" is enabled, since this currently seems to


> An additional concern I have identified is that this PR does not support "disable_padded_drafter_batch". I believe two changes would be necessary to enable this:
> 
> * Update `self.valid_sampled_token_count_cpu` in the `disable_padded_drafter_batch` pathway of `propose_draft_token_ids` following `prepare_next_token_ids_cpu`.
> * add an exception to `if not self.use_async_scheduling` in `_bookkeeping_sync`, since this is getting thrown off and leaving `valid_sampled_token_ids` as empty.
> 
> This is very similar work that would be needed to enable other speculative decoding methods from having overlapping support, and _does not need to be included in this PR_. However, in the meantime, please add some validation that will raise a warning/error if "disable_padded_drafter_batch" is enabled, since this currently seems to lead to an ugly crash.

ok, i will fix this.
---
### Ronald1995 @ 2025-10-21T15:25:44Z

> > @Ronald1995 I have confirmed the issue and propose the following patch:
> > In `gpu_model_runner.py:2908` (propose_draft_token_ids):
> > ```
> >         elif self.speculative_config.use_eagle():
> >             if self.prepare_inputs_event is not None: # new
> >                 self.prepare_inputs_event.synchronize() # new
> >             assert isinstance(self.drafter, EagleProposer)
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > This enforces a synchronization between prepare_inputs of the base model and the EAGLE drafter. With this patch, there can be no overlap between the draft model's prepare_inputs cpu execution and the gpu execution of the target model's prepare_inputs _for the same step_. It is still able to overlap between iterations, and profiling on nsight systems for Llama 3.1 8B with EAGLE3 indicates that this is not a significant block. CPU-GPU overlapping still occurs during the target model's forward pass. You are welcome to benchmark and profile with patch if you are suspicious of the performance implications.
> > While it would be nice to have identified a specific problematic shared buffer that can simply be replicated, this patch will serve in the meantime to alleviate the issue until the root cause can be identified. Even then, it might be beneficial to have this as a sanity check.
> 
> @benchislett Thanks for your information, i will make more performance about your advised patch and try to identify the a specific problematic shared buffer.

@benchislett i have  identified the specific problematic shared buffer. please seed the [code](https://github.com/vllm-project/vllm/pull/24799/files#diff-a4809a837fbf535a8f0999b11087a53ec1c53948b50c0a1fe64396bc86de9461R394) in `eagle.py`, the root cause is in-place operations in `propose` function.
---
### Ronald1995 @ 2025-10-21T15:33:27Z

> > eantime, please add some validation that will raise a warning/error if "disable_padded_drafter_batch" is enabled, since this currently seems to
> 
> > An additional concern I have identified is that this PR does not support "disable_padded_drafter_batch". I believe two changes would be necessary to enable this:
> > 
> > * Update `self.valid_sampled_token_count_cpu` in the `disable_padded_drafter_batch` pathway of `propose_draft_token_ids` following `prepare_next_token_ids_cpu`.
> > * add an exception to `if not self.use_async_scheduling` in `_bookkeeping_sync`, since this is getting thrown off and leaving `valid_sampled_token_ids` as empty.
> > 
> > This is very similar work that would be needed to enable other speculative decoding methods from having overlapping support, and _does not need to be included in this PR_. However, in the meantime, please add some validation that will raise a warning/error if "disable_padded_drafter_batch" is enabled, since this currently seems to lead to an ugly crash.
> 
> ok, i will fix this.

@benchislett i have add validation in arg_utils.py to make sure `disable_padded_drafter_batch=False` for async_scheduling. 
as for another two changes you suggested, you think when `disable_padded_drafter_batch=True`, sync_scheduling will crash, right? if so, i think it's unnecessary to add the changes, because `disable_padded_drafter_batch=True` won't affect sync_scheduling in this PR, and i test it in my local env.
---
### benchislett @ 2025-10-21T15:57:37Z

@Ronald1995 sync scheduling should be functional and unchanged. I can confirm this but I'm pretty sure the only issue is `spec + disable-padded-drafter-batch + async sched`. Thanks for adding the validation
---
### benchislett @ 2025-10-21T16:14:37Z

@Ronald1995 I think it makes more sense to solve this problem by calling `.clone()` on the relevant tensors in `prepare_inputs` and `prepare_inputs_padded` in `eagle.py`. I plan to rewrite that logic into a custom kernel anyways, so it is preferable if those metadata are mutable in the first place. I confirmed myself that this implementation resolves the accuracy discrepancy also. 
---
### benchislett @ 2025-10-21T16:15:23Z

Good work finding the root cause!
---
### njhill @ 2025-10-21T16:42:01Z

Thanks @Ronald1995 @benchislett for all of the work on this! I am taking a look now too, and I think it's important for @WoosukKwon to review.

One thing missing is an e2e CI test covering this. It should be added to https://github.com/vllm-project/vllm/blob/main/tests/v1/e2e/test_async_sched_and_preempt.py so that we also test e2e permutations of this in conjunction with request preemption, penalty sampling parameters, and (soon to be [merged](https://github.com/vllm-project/vllm/pull/26866)) structured outputs.

We should also have an e2e test that verifies the acceptance rate matches when running with/without.
---
### Ronald1995 @ 2025-10-23T01:31:13Z

> Thanks @Ronald1995 @benchislett for all of the work on this! I am taking a look now too, and I think it's important for @WoosukKwon to review.
> 
> One thing missing is an e2e CI test covering this. It should be added to https://github.com/vllm-project/vllm/blob/main/tests/v1/e2e/test_async_sched_and_preempt.py so that we also test e2e permutations of this in conjunction with request preemption, penalty sampling parameters, and (soon to be [merged](https://github.com/vllm-project/vllm/pull/26866)) structured outputs.
> 
> We should also have an e2e test that verifies the acceptance rate matches when running with/without.

@njhill i will add async_scheduling + spec_decoding to e2e test, which will cover preemption and acceptance rate and penalty sampling parameters, but for structured output #26866, it's not merged yet, can i add it to e2e test in next PR or #26866 could add it?


---
### Ronald1995 @ 2025-10-23T01:35:46Z

@benchislett @njhill thank for your reviews, i have modified code according your suggestions, and e2e test will be added soon, please review again, thanks!
---
### benchislett @ 2025-10-23T17:04:24Z

Please ignore structured outputs for now. I can own following-up on integrating the both of these depending on merge order.
---
### Ronald1995 @ 2025-10-24T12:41:16Z

> Please ignore structured outputs for now. I can own following-up on integrating the both of these depending on merge order.

@benchislett  ok! what's more, the e2e test is added and is validated in my local env, all other issues are also fixed. when i debug the test case, i found #26467 only make `penalties and bad_words` work for async_scheduling without spec_decoding, it's more complicated to support `penalties and bad_words`  for async_scheduling with spec_decoding. so i add assertion for this situation, and plan to implement it in another PR.

```python
 assert not (
      self.use_async_scheduling and self.input_batch.needs_output_token_ids()
  ), (
      "async scheduling with spec decoding don't support penalties "
      "and bad words in sampling parameters now."
  )
```
---
### njhill @ 2025-10-30T00:59:25Z

Thanks again for all this work @Ronald1995. I have been starting to look at the changes in more detail and have made a first pass with some simplifications: https://github.com/njhill/vllm/commit/c4e533d30e55cc827ae1968b3b8070b616c9c987.

This is targeted at your branch, assuming that https://github.com/vllm-project/vllm/pull/27756 is also merged.

That PR more generally fixes the async scheduling case where requests aren't scheduled, the previous fix I made to async sched + preemption should have included this. It means that all of the other special case handling for this isn't needed.

I am still working through some of the other changes!

---
### Ronald1995 @ 2025-10-30T01:58:59Z

> Thanks again for all this work @Ronald1995. I have been starting to look at the changes in more detail and have made a first pass with some simplifications: [njhill@c4e533d](https://github.com/njhill/vllm/commit/c4e533d30e55cc827ae1968b3b8070b616c9c987).
> 
> This is targeted at your branch, assuming that #27756 is also merged.
> 
> That PR more generally fixes the async scheduling case where requests aren't scheduled, the previous fix I made to async sched + preemption should have included this. It means that all of the other special case handling for this isn't needed.
> 
> I am still working through some of the other changes!

@njhill Thank you for simplifying the code.
---
### mergify @ 2025-10-31T01:43:24Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### vadiklyutiy @ 2025-11-01T01:17:55Z

May I double check. 
After the PR merged, asynv scheduling will work fine with spec decoding? 
---
### vadiklyutiy @ 2025-11-01T01:18:32Z

@benchislett Maybe it's worth to run CI against this PR? 
---
### Ronald1995 @ 2025-11-01T03:12:22Z

> May I double check. After the PR merged, asynv scheduling will work fine with spec decoding?

The answer is yes. you can see e2e testcase `test_async_scheduling.py`, async_scheduling with spec decode test is added to the testcase.
---
### benchislett @ 2025-11-04T05:05:44Z

Waiting for final sign-off from @njhill or @WoosukKwon before marking as "ready"
---
### njhill @ 2025-11-04T06:22:03Z

Apologies I was battling a perf regression and more CI issues today but will focus on this tomorrow.
---
### njhill @ 2025-11-06T22:52:10Z

> front-end you mentioned is coreclient? i tried to add the check in `add_request` method of coreclient, but there are kinds of of coreclient, so i add the check logic in `core.py` in case of too many modifications for coreclient.

@Ronald1995 I've pushed a commit with this change (moving the validation to processor.py), along with some other minor simplification and comment cleanup, I hope this is ok.
---
### njhill @ 2025-11-06T23:01:32Z

@Ronald1995 what will happen if `input_fits_in_drafter` is `False` here: https://github.com/vllm-project/vllm/blob/ca6f755d2416bb1bd1b924b40481a41b2b0eaf85/vllm/v1/worker/gpu_model_runner.py#L2692-L2695?

I think it won't update `self._draft_token_ids`, `input_batch.prev_sampled_token_ids`, or `self.valid_sampled_token_count_cpu`, and won't call `record()` on the event which I expect will mean that things hang?

I think we also need to address that case if so.

Edit: related PR: https://github.com/vllm-project/vllm/pull/28298
---
### mergify @ 2025-11-07T21:21:42Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Ronald1995 @ 2025-11-08T01:08:22Z

> > front-end you mentioned is coreclient? i tried to add the check in `add_request` method of coreclient, but there are kinds of of coreclient, so i add the check logic in `core.py` in case of too many modifications for coreclient.
> 
> @Ronald1995 I've pushed a commit with this change (moving the validation to processor.py), along with some other minor simplification and comment cleanup, I hope this is ok.

thanks for your commit.
---
### Ronald1995 @ 2025-11-08T01:18:52Z

> input_fits_in_drafter

`input_fits_in_drafter` condition is added after I create this PR, i didn't notice it. i will address the case, thanks for your reminder.
---
### Ronald1995 @ 2025-11-08T12:26:44Z

@njhill all new issues have been resolved, please review again, thanks!
---
### mergify @ 2025-11-10T23:35:35Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### njhill @ 2025-11-12T05:18:00Z

@Ronald1995 @benchislett I think this is ok now, apart from the test which I'm still doing more work on, hope to get that done tomorrow.

Do you know if there is a smaller MTP model we could use for the test? The startup time is relatively long with this one and we want to launch it and compare a number of different configurations.
---
### Ronald1995 @ 2025-11-12T08:15:28Z

> @Ronald1995 @benchislett I think this is ok now, apart from the test which I'm still doing more work on, hope to get that done tomorrow.
> 
> Do you know if there is a smaller MTP model we could use for the test? The startup time is relatively long with this one and we want to launch it and compare a number of different configurations.

@njhill Thanks for your work. I'm sorry i don't known a smaller MTP model than mimo-7b.
---
### lhtin @ 2025-11-12T12:36:06Z

@Ronald1995 @njhill When I'm running DeepSeek R1 with Async Scheduling + MTP, I notice there are gaps of about 1.x ms on the GPU. Does this fall within expected behavior? Have you captured and analyzed the timeline for this scenario?

<img width="754" height="377" alt="Clipboard_Screenshot_1762950760" src="https://github.com/user-attachments/assets/965dbc39-837a-42ec-a882-e48f5c2444f2" />

---
### benchislett @ 2025-11-12T23:25:59Z

@lhtin It works well for me. Here's a profile snapshot on 8xB200:
<img width="2063" height="411" alt="image" src="https://github.com/user-attachments/assets/5683a357-295a-4104-9db6-e1dcc3c9d6dd" />

Launched with:
```
VLLM_TORCH_CUDA_PROFILE=1 VLLM_FLASHINFER_MOE_BACKEND=latency VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_FLASHINFER_MOE_FP8=1 \
  nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=graph \
    --cuda-event-trace=false \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    --gpu-metrics-devices=0 \
  vllm serve deepseek-ai/DeepSeek-R1-0528 -tp 8 --max-model-len 8192 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --async-scheduling
```

Profiled with:
```
vllm bench serve --dataset-name random --ignore-eos --num-prompts 2 --max-concurrency 1 --random-input-len 2048 --random-output-len 128 --model deepseek-ai/DeepSeek-R1-0528 --base-url http://localhost:8049 --random-range-ratio 0.2 --profile
```

Could you poke around the profile and try to identify what the sync is waiting for? On the timeline of "prepare_inputs -> model fwd -> sampling -> drafting -> next iteration start" where does the sync get issued and when does it block until?
---
### lhtin @ 2025-11-13T02:58:07Z

@benchislett Thanks for your reply. If I test with MTP=2 and there are no gaps, as shown in the figure below, it indicates that when the execution time of the MTP model following the main model is relatively short—meaning the GPU finishes prev MTP step quickly—the next step may still be waiting for the CPU to launch the graph. This waiting period can result in gaps during that time.

If we can remove the sync operation in _update_states, cudaGraphLaunchcan will be executed earlier. Our internal approach involves first assuming all draft tokens have been fully accepted, then adjusting data such as `seq_lens` and `slot_mapping` directly on the GPU.

Supplement the test environment: 8xH20

MTP=2, cudaGraph overlap GPU MTP kernel:
<img width="1362" height="490" alt="Clipboard_Screenshot_1763002332" src="https://github.com/user-attachments/assets/044b09c6-7395-4a84-9c81-f9fc82643a97" />

MTP=1, cudaGraph not overlap GPU MTP kernel:
<img width="1261" height="352" alt="Clipboard_Screenshot_1763002394" src="https://github.com/user-attachments/assets/af11cb0d-c163-467e-896b-7adac1db126d" />


---
### Ronald1995 @ 2025-11-13T03:23:01Z

> @benchislett Thanks for your reply. If I test with MTP=2 and there are no gaps, as shown in the figure below, it indicates that when the execution time of the MTP model following the main model is relatively short—meaning the GPU finishes prev MTP step quickly—the next step may still be waiting for the CPU to launch the graph. This waiting period can result in gaps during that time.
> 
> If we can remove the sync operation in _update_states, cudaGraphLaunchcan will be executed earlier. Our internal approach involves first assuming all draft tokens have been fully accepted, then adjusting data such as `seq_lens` and `slot_mapping` directly on the GPU.
> 
> Supplement the test environment: 8xH20
> 
> MTP=2, cudaGraph overlap GPU MTP kernel: <img alt="Clipboard_Screenshot_1763002332" width="1362" height="490" src="https://private-user-images.githubusercontent.com/13173904/513641034-044b09c6-7395-4a84-9c81-f9fc82643a97.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjMwMDMzOTksIm5iZiI6MTc2MzAwMzA5OSwicGF0aCI6Ii8xMzE3MzkwNC81MTM2NDEwMzQtMDQ0YjA5YzYtNzM5NS00YTg0LTljODEtZjlmYzgyNjQzYTk3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEzVDAzMDQ1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUyODRkYzhjMWU0M2FiMzg3MDM4ZTQ4YzlkZmQ3YzIxZDViYTcwZGNhOTA4ZDlhNGVlMWFmZjViOGQ5NDIwZDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OvdsR2ZgJAmBkj0X4X3rrjjDZUiyik9_FDwzEYBL1Us">
> 
> MTP=1, cudaGraph not overlap GPU MTP kernel: <img alt="Clipboard_Screenshot_1763002394" width="1261" height="352" src="https://private-user-images.githubusercontent.com/13173904/513641279-af11cb0d-c163-467e-896b-7adac1db126d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjMwMDMzOTksIm5iZiI6MTc2MzAwMzA5OSwicGF0aCI6Ii8xMzE3MzkwNC81MTM2NDEyNzktYWYxMWNiMGQtYzE2My00NjdlLTg5NmItN2FkYWMxZGIxMjZkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEzVDAzMDQ1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY5MmZiNzU3OWQ4OGQxMTNmODUwZGRiZTU2MTJjMzRkZWJlNzA5YmUyZTBhOGI5MGJhMmNjZjU2MTQ0MDA2NDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OTayK1cXRelAJkMaTix5FeLhjRsj6et4Z8rHlTOXmRI">



> @benchislett Thanks for your reply. If I test with MTP=2 and there are no gaps, as shown in the figure below, it indicates that when the execution time of the MTP model following the main model is relatively short—meaning the GPU finishes prev MTP step quickly—the next step may still be waiting for the CPU to launch the graph. This waiting period can result in gaps during that time.
> 
> If we can remove the sync operation in _update_states, cudaGraphLaunchcan will be executed earlier. Our internal approach involves first assuming all draft tokens have been fully accepted, then adjusting data such as `seq_lens` and `slot_mapping` directly on the GPU.
> 
> Supplement the test environment: 8xH20
> 
> MTP=2, cudaGraph overlap GPU MTP kernel: <img alt="Clipboard_Screenshot_1763002332" width="1362" height="490" src="https://private-user-images.githubusercontent.com/13173904/513641034-044b09c6-7395-4a84-9c81-f9fc82643a97.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjMwMDMzOTksIm5iZiI6MTc2MzAwMzA5OSwicGF0aCI6Ii8xMzE3MzkwNC81MTM2NDEwMzQtMDQ0YjA5YzYtNzM5NS00YTg0LTljODEtZjlmYzgyNjQzYTk3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEzVDAzMDQ1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUyODRkYzhjMWU0M2FiMzg3MDM4ZTQ4YzlkZmQ3YzIxZDViYTcwZGNhOTA4ZDlhNGVlMWFmZjViOGQ5NDIwZDYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OvdsR2ZgJAmBkj0X4X3rrjjDZUiyik9_FDwzEYBL1Us">
> 
> MTP=1, cudaGraph not overlap GPU MTP kernel: <img alt="Clipboard_Screenshot_1763002394" width="1261" height="352" src="https://private-user-images.githubusercontent.com/13173904/513641279-af11cb0d-c163-467e-896b-7adac1db126d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjMwMDMzOTksIm5iZiI6MTc2MzAwMzA5OSwicGF0aCI6Ii8xMzE3MzkwNC81MTM2NDEyNzktYWYxMWNiMGQtYzE2My00NjdlLTg5NmItN2FkYWMxZGIxMjZkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTExMTMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMTEzVDAzMDQ1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWY5MmZiNzU3OWQ4OGQxMTNmODUwZGRiZTU2MTJjMzRkZWJlNzA5YmUyZTBhOGI5MGJhMmNjZjU2MTQ0MDA2NDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OTayK1cXRelAJkMaTix5FeLhjRsj6et4Z8rHlTOXmRI">

@lhtin does tpot get worser in your test with mtp=1?
---
### lhtin @ 2025-11-13T03:32:43Z

> @lhtin does tpot get worser in your test with mtp=1?

@Ronald1995 Compared to not using async scheduling, the TPOT has decreased, the performance is better when enable async scheduling. Just looking at the timeline, it feels like there's still room for optimization.
---
### Ronald1995 @ 2025-11-13T04:01:59Z


> @Ronald1995 Compared to not using async scheduling, the TPOT has decreased, the performance is better. Just looking at the timeline, it feels like there's still room for optimization.

@lhtin I got your point, your internal approach has big difference between this PR. my opinion is you could new another PR to implement your approach. of course, it depends on the opinions of nick and ben @benchislett @njhill 
---
### lhtin @ 2025-11-13T06:13:38Z

> @lhtin I got your point, your internal approach has big difference between this PR. my opinion is you could new another PR to implement your approach. of course, it depends on the opinions of nick and ben @benchislett @njhill

@Ronald1995 It's not related to our internal approach, nor does it mean that this implementation is bad. I really appreciate you submitting such an great PR.

What I'm saying is that with the current PR, there are gaps under certain circumstances. Does this align with the expected behavior?


---
### benchislett @ 2025-11-13T16:34:39Z

@lhtin I see what you mean. This is a consequence of having to wait for the main model forward pass to finish before we can start prepare_inputs in the next step. Since we're only overlapping with the drafting here, if the drafting is quick then we can have a blocking sync.

This is a known compromise of this PR. I think it would add a bit of complexity to address this, but it wouldn't be a big change to the diff in the PR. I think we should continue with this proposal as-is and treat your proposed change as a potential performance improvement for future work. 
---
### benchislett @ 2025-11-13T17:01:03Z

@lhtin I also wonder if that approach would work for all attention backends. I know some attention backends require the metadata on both the CPU and GPU in order to run properly, and that might be a challenge to wrestle with the current abstractions.
---
### njhill @ 2025-11-13T22:35:44Z

@Ronald1995 @benchislett ok I have pushed the test update and running full CI now
---
### lhtin @ 2025-11-14T04:17:28Z

> This is a known compromise of this PR. I think it would add a bit of complexity to address this, but it wouldn't be a big change to the diff in the PR. I think we should continue with this proposal as-is and treat your proposed change as a potential performance improvement for future work.

Understood, I completely agree to address this potential issue in a separate PR.

> @lhtin I also wonder if that approach would work for all attention backends. I know some attention backends require the metadata on both the CPU and GPU in order to run properly, and that might be a challenge to wrestle with the current abstractions.

Indeed, this would add significant complexity. This aspect requires careful analysis to strike a balance between performance and maintainability.


---
### njhill @ 2025-11-14T04:40:22Z

@Ronald1995 @benchislett unfortunately this test runs on an L4 which doesn't have enough vram to load the `XiaomiMiMo/MiMo-7B-Base` model in fp32, and is < sm_90 so the batch invariance feature doesn't work. One of these is needed to ensure stability of the outputs or else the correctness checks don't work.

I'm not sure of the best way to proceed... it would be really nice if there was a smaller test model we could use for the spec decoding cases!
---
### Ronald1995 @ 2025-11-14T07:23:16Z

> I'm not sure of the best way to proceed... it would be really nice if there was a smaller test model we could use for the spec decoding cases!

@njhill i try to find a smaller model, but it's hard to find one. maybe we can set `max_num_seqs=1` to avoid batch size's influence on output and use less prompts to accelerate the test?
---
### njhill @ 2025-11-14T19:48:50Z

I'm looking into getting a eagle model trained for llama-3.2-1B which we could use here. In the meantime I think we can merge this with the spec decode tests marked as skipped.

I have opened a separate PR with the test rework though which would be good to merge first, then can rebase this one: https://github.com/vllm-project/vllm/pull/28744
---
### mergify @ 2025-11-15T06:55:08Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Ronald1995.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---