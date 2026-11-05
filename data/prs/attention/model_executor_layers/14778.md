# PR #14778: [Model] Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copies 

- field: attention  | area: model_executor/layers
- author: cyang49
- state: MERGED  created: 2025-03-13T19:20:57Z  merged: 2025-03-14T20:36:18Z
- url: https://github.com/vllm-project/vllm/pull/14778
- labels: ready

## Body

We found an issue while profiling vLLM running Bamba-9B model inference.

Before:
![image (12)](https://github.com/user-attachments/assets/06e71567-e74d-4578-b698-30f7889f397b)

As can be seen in the Nsight Systems trace, per Mamba layer there are 2 phases where frequent memory copies happen. They are not necessary, or can be fused to reduce the number of copies. This PR fixes these issues.

After:
<img width="1393" alt="image" src="https://github.com/user-attachments/assets/dfcde710-75c8-4293-a2c0-eb2c5e7100f8" />

For the test case (offline mode, batch size=64, short prompt) I used, the fix reduces the prefill mamba layer latency from 5ms to 3ms.

The results from benchmark_serving on single H100-80GB GPU
Before:
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  283.22    
Total input tokens:                      215201    
Total generated tokens:                  198343    
Request throughput (req/s):              3.53      
Output token throughput (tok/s):         700.32    
Total Token throughput (tok/s):          1460.17   
---------------Time to First Token----------------
Mean TTFT (ms):                          105627.40 
Median TTFT (ms):                        94728.54  
P99 TTFT (ms):                           264194.77 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          393.83    
Median TPOT (ms):                        413.59    
P99 TPOT (ms):                           615.34    
---------------Inter-token Latency----------------
Mean ITL (ms):                           339.72    
Median ITL (ms):                         589.56    
P99 ITL (ms):                            751.76    
==================================================
```
After:
```
 python benchmarks/benchmark_serving.py --model $MODEL_PATH  --dataset-name sharegpt     --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  260.19    
Total input tokens:                      215201    
Total generated tokens:                  198343    
Request throughput (req/s):              3.84      
Output token throughput (tok/s):         762.29    
Total Token throughput (tok/s):          1589.37   
---------------Time to First Token----------------
Mean TTFT (ms):                          96566.51  
Median TTFT (ms):                        84883.05  
P99 TTFT (ms):                           245639.66 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          366.31    
Median TPOT (ms):                        371.88    
P99 TPOT (ms):                           680.49    
---------------Inter-token Latency----------------
Mean ITL (ms):                           311.69    
Median ITL (ms):                         507.96    
P99 ITL (ms):                            741.83    
==================================================
```

The total token throughput improved by about 8%.

Note:
There is another sequential for loop which can be fixed similarly. My test case doesn't hit this control path, though. @fabianlim could you comment?



## Comments

### github-actions @ 2025-03-13T19:21:07Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### cyang49 @ 2025-03-13T21:57:15Z

> Nice performance pickup. Is this the other sequential for loop you mentioned?
> 
> https://github.com/vllm-project/vllm/blob/02fcaa3d0a91364da4534ff2f3f6c867eb559149/vllm/model_executor/layers/mamba/mamba_mixer2.py#L470-L472
> 
> Do you want to handle it in this PR?

I need @fabianlim 's input on how to hit that case. It can be a separate PR or if I know how to test it tomorrow. Next week I'll be traveling and may not have time to do it
---
### fabianlim @ 2025-03-14T00:46:11Z

since the mamba2 unit tests are not automated maybe we should run them once ? @tlrmchlsmth 

@cyang49 this will be true if, at least one of the sequences in the current step has an initial state, which is determined by the prescence of a context. This means that either a i) chunked prefill step or ii) decode step will hit this case.

```
has_initial_states = attn_metadata.context_lens_tensor > 0
```

---
### cyang49 @ 2025-03-14T14:10:14Z

I vectorized the zero init loop and observed a slight improvement in total token throughput

```
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  247.81
Total input tokens:                      215201
Total generated tokens:                  198343
Request throughput (req/s):              4.04
Output token throughput (tok/s):         800.37
Total Token throughput (tok/s):          1668.76
---------------Time to First Token----------------
Mean TTFT (ms):                          97128.43
Median TTFT (ms):                        89290.52
P99 TTFT (ms):                           233402.22
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          360.62
Median TPOT (ms):                        355.58
P99 TPOT (ms):                           992.48
---------------Inter-token Latency----------------
Mean ITL (ms):                           294.58
Median ITL (ms):                         503.07
P99 ITL (ms):                            570.28
==================================================
```
---
### cyang49 @ 2025-03-14T15:26:32Z

@tlrmchlsmth it would be nice if we can merge this one soon, if the functionality & no negative performance impact are verified. I noticed from the trace that there are other inefficiencies in  mamba2, but I'll submit a separate PR after my trip. Let me know if there's anything else that needs changing. Thanks!
---
### DarkLight1337 @ 2025-03-14T17:13:07Z

Some CI failures have recently been fixed on main, so I suggest you to merge from main if you haven't already
---
### yury-tokpanov @ 2025-03-14T23:58:48Z

Testing this. We did notice the same in our profiles of mamba2. Overall, occupancy was pretty low in comparison to flash attention kernels.
---
### yury-tokpanov @ 2025-03-15T01:31:11Z

this breaks mamba2 based models, unfortunately:

Command I ran on H100: `lm_eval --model vllm --model_args pretrained=ibm-ai-platform/Bamba-9B,dtype=float16,gpu_memory_utilization=0.9,max_model_len=4096 --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k`

bamba-9b with this PR:

> |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|                                                                                                                                                                                                                                                                                     
> |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|                                                                                                                                                                                                                                                                                     
> |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.0781|±  |0.0074|                                                                                                      
> |     |       |strict-match    |     5|exact_match|↑  |0.0569|±  |0.0064|

bamba-9b with PR reverted:

> |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|                                                                                                      
> |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|                                                                                                      
> |gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.2449|±  |0.0118|                                                                                                                                                                                                                                                                                     
> |     |       |strict-match    |     5|exact_match|↑  |0.3692|±  |0.0133|    
---
### cyang49 @ 2025-03-15T01:56:08Z

Weird, it passed when I tested locally? Both value and stderr should be 0s?
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |    0|±  |     0|
|     |       |strict-match    |     5|exact_match|↑  |    0|±  |     0|
```
---
### yury-tokpanov @ 2025-03-15T02:02:09Z

> Weird, it passed when I tested locally? Both value and stderr should be 0s?
> 
> ```
> |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
> |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
> |gsm8k|      3|flexible-extract|     5|exact_match|↑  |    0|±  |     0|
> |     |       |strict-match    |     5|exact_match|↑  |    0|±  |     0|
> ```

No, it shouldn't be 0 accuracy. 0 means the model failed completely on a test. For the full gsm8k eval Bamba-9b should be around 37% on a strict-match accuracy (with around 1% stderr). I checked other mamba2 models (Codestral-7B, Zamba2), they are also down. 

Do you have Slack? I'd suggest you join vLLM dev Slack, we have a channel there to discuss hybrid models: https://slack.vllm.ai/ 
---
### cyang49 @ 2025-03-15T02:07:43Z

> > Weird, it passed when I tested locally? Both value and stderr should be 0s?
> > ```
> > |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
> > |-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
> > |gsm8k|      3|flexible-extract|     5|exact_match|↑  |    0|±  |     0|
> > |     |       |strict-match    |     5|exact_match|↑  |    0|±  |     0|
> > ```
> 
> No, it shouldn't be 9 accuracy. 0 means the model failed completely on a test. For the full gsm8k eval Bamba-9b should be around 37% on a strict-match accuracy (with around 1% stderr). I checked other mamba2 models (Codestral-7B, Zamba2), they are also down.
> 
> Do you have Slack? I'd suggest you join vLLM dev Slack, we have a channel there to discuss hybrid models: https://slack.vllm.ai/

Ah, thanks for explaining. I'll debug it when I get a chance. I'll also get on the vllm slack
---
### yury-tokpanov @ 2025-03-15T02:11:14Z

I think the issue is with ssm state copy, zero-initialization appears to be working fine.
---
### cyang49 @ 2025-03-15T02:15:00Z

> I think the issue is with ssm state copy, zero-initialization appears to be working fine.

It could also be that lm-eval doesn't go through the zero-init path, though
---
### cyang49 @ 2025-03-15T03:19:58Z

It appears the problem is that the semantics of the line `mamba_cache_params.ssm_state[idx].copy_(varlen_state[i])` in the for loop is different from `mamba_cache_params.ssm_state[idx].copy_(source_state)` in the lambda function :(

In the former, `idx` is a scalar integer value and the in-place copy happens, but in the latter, `idx` is an integer tensor and the indexing semantics is different. I suspect that the in-place copy doesn't happen as expected - I experimented with these two cases in the python interpreter..

It looks like the in-place `zero_()` part should  have the same issue. Not sure why it didn't cause a problem for gsm8k
---
### fabianlim @ 2025-03-15T14:43:23Z

@cyang49 when `idx` is a tensor is a copy-view, so thats why the inplace does not update the master copy. That is why i needed to loop it with a scalar in the first place.
---