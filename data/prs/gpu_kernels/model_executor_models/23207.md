# PR #23207: [Misc][qwen2_5_vl][torch.compile] Enable `supports_torch_compile` on generic nn.Module and demonstrate speedup on Qwen Vision model

- field: attention  | area: model_executor/models
- author: Lucaskabela
- state: MERGED  created: 2025-08-19T21:15:13Z  merged: 2025-10-28T22:36:43Z
- url: https://github.com/vllm-project/vllm/pull/23207
- labels: documentation, ready, llama, qwen

## Body

We enable `@supports_torch_compile` on generic nn.Modules (as opposed to only top level architecture) and demonstrate the application in qwen_2_5_vl

## Purpose
This PR is a first step towards supporting torch compile for multimodal encoders (such as vision or audio).  Since these modality specific components do not have vLLM config, we begin by taking a step to allow these modules to be compiled

## Test Plan
### Unit Test
```
with-proxy pytest tests/compile/test_multimodal_compile.py
```
```
1 passed, 9 warnings in 45.26s
```
### E2E Tests
```
VLLM_USE_V1=1 python examples/offline_inference/vision_language.py -m qwen2_5_vl
```
## Test Result
### BEFORE CHANGES:
```
^[[1;36m(EngineCore_0 pid=2638689)^[[0;0m INFO 08-19 14:14:20 [backends.py:559] Dynamo bytecode transform time: 3.26 s
^[[1;36m(EngineCore_0 pid=2638689)^[[0;0m INFO 08-19 14:14:22 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 1.965 s
^[[1;36m(EngineCore_0 pid=2638689)^[[0;0m INFO 08-19 14:14:23 [monitor.py:34] torch.compile takes 3.26 s in total
...

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:00<00:02,  1.41it/s, est. speed input: 1800.03 toks/s, output: 90.28 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:00<00:00,  1.41it/s, est. speed input: 7137.36 toks/s, output: 357.98 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:00<00:00,  5.59it/s, est. speed input: 7137.36 toks/s, output: 357.98 toks/s]
```
### AFTER CHANGES:
```
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:31 [backends.py:559] Dynamo bytecode transform time: 0.37 s
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:31 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 0.020 s
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:31 [monitor.py:34] torch.compile takes 0.37 s in total
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:34 [backends.py:548] Using cache directory: /home/lucaskabela/.cache/vllm/torch_compile_cache/1de7867ecf/rank_0_0/backbone for vLLM's torch.compile
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:34 [backends.py:559] Dynamo bytecode transform time: 3.16 s
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:37 [backends.py:161] Directly load the compiled graph(s) for dynamic shape from the cache, took 2.109 s
^[[1;36m(EngineCore_0 pid=2569806)^[[0;0m INFO 08-19 14:12:38 [monitor.py:34] torch.compile takes 3.53 s in total
...
Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts:  25%|██▌       | 1/4 [00:00<00:02,  1.43it/s, est. speed input: 1822.83 toks/s, output: 91.42 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:00<00:00,  1.43it/s, est. speed input: 7228.82 toks/s, output: 362.57 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:00<00:00,  5.66it/s, est. speed input: 7228.82 toks/s, output: 362.57 toks/s]
```

## Benchmarks
```
vllm serve Qwen/Qwen2.5-VL-3B-Instruct
```
```
vllm bench serve   --backend openai-chat   --endpoint-type openai-chat   --model Qwen/Qwen2.5-VL-3B-Instruct   --endpoint /v1/chat/completions   --dataset-name hf   --dataset-path lmarena-ai/VisionArena-Chat   --hf-split train   --num-prompts 1000
```

### Main + Torch (nightly)
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  135.59    
Total input tokens:                      94327     
Total generated tokens:                  106141    
Request throughput (req/s):              7.37      
Output token throughput (tok/s):         782.79    
Total Token throughput (tok/s):          1478.44   
---------------Time to First Token----------------
Mean TTFT (ms):                          63924.39  
Median TTFT (ms):                        63589.24  
P99 TTFT (ms):                           127523.65 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          681.74    
Median TPOT (ms):                        661.11    
P99 TPOT (ms):                           1375.61   
---------------Inter-token Latency----------------
Mean ITL (ms):                           596.16    
Median ITL (ms):                         78.84     
P99 ITL (ms):                            1564.77   
==================================================
```

### This PR + Torch (nightly)
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  129.73    
Total input tokens:                      94327     
Total generated tokens:                  106596    
Request throughput (req/s):              7.71      
Output token throughput (tok/s):         821.69    
Total Token throughput (tok/s):          1548.80   
---------------Time to First Token----------------
Mean TTFT (ms):                          61043.37  
Median TTFT (ms):                        60044.40  
P99 TTFT (ms):                           123046.15 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          648.66    
Median TPOT (ms):                        625.63    
P99 TPOT (ms):                           1308.29   
---------------Inter-token Latency----------------
Mean ITL (ms):                           569.48    
Median ITL (ms):                         76.11     
P99 ITL (ms):                            1492.54   
==================================================
```

So with this PR, we observe moderate improvements to several key metrics (~5% relative improvement to throughput,  ~2% improvement to TTFT, ~7% improvement to TPOT, and ~7% improvement to mean ITL (median ITL has only a minor improvement)

This requires only a minor rewrite (moving attention to a custom operator) in order to torch compile the main vision backbone (the VisionBlock).  

## (Optional) Documentation Update

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### github-actions @ 2025-08-19T21:15:19Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### ProExpertProg @ 2025-08-20T02:17:23Z

Sorry I'm not familiar, why do mm components not have vllm*_config set?
---
### tanruixiang @ 2025-08-20T03:13:20Z

This modification may cause recompilation during runtime. It appears that this may cause timeouts in some scenarios.🤔
---
### Lucaskabela @ 2025-08-21T15:53:26Z

> Sorry I'm not familiar, why do mm components not have vllm*_config set?

I am honestly not too sure; an alternative we could do here is forward the vllm_config to these subclasses; this would be a less flexible solution (as it requires any nn_module we want to compile to have vllm_config in the init), but it is doable
---
### Lucaskabela @ 2025-08-21T15:55:14Z

> This modification may cause recompilation during runtime. It appears that this may cause timeouts in some scenarios.🤔

This is a valid concern, I will stress test this to check - if there is some easy way to trigger recompiles, can you share examples or a script that would trigger recompiles? In the meanwhileI will continue to test and try and trigger on my own.  Thanks!
---
### Lucaskabela @ 2025-08-25T18:25:37Z

From slack discussion:

> [Try running] https://github.com/vllm-project/vllm/tree/main/benchmarks#visionarena-benchmark-for-vision-language-models; https://github.com/vllm-project/vllm/tree/main/benchmarks#%EF%B8%8F-example---multi-modal-benchmark

Will try running these soon!
---
### zou3519 @ 2025-08-25T19:48:16Z

> This modification may cause recompilation during runtime. It appears that this may cause timeouts in some scenarios.🤔

@tanruixiang what do you mean by this? The way the vLLM-compile integration works, vLLM deletes the guards that TorchDynamo produces, so there will never be recompilations during inference serving time.
---
### tanruixiang @ 2025-08-26T05:16:26Z

> @tanruixiang what do you mean by this? The way the vLLM-compile integration works, vLLM deletes the guards that TorchDynamo produces, so there will never be recompilations during inference serving time.

@zou3519  Thanks for pointing this out. I also see slack's discussion. Let's wait for the benchmark first.


---
### Lucaskabela @ 2025-08-26T21:50:47Z

@tanruixiang @zou3519  @ProExpertProg @ywang96  updated with some benchmark results for more context on our discussions
---
### miladm @ 2025-08-27T23:10:08Z

@Lucaskabela do we have a bug for future steps of this effort?
---
### Lucaskabela @ 2025-08-29T17:03:32Z

@miladm  currently just waiting on feedback here
---
### Lucaskabela @ 2025-09-09T18:41:20Z

Hi @ywang96  I think that is a very valid point - I will work on applying this to some places in order to show strong benefits
---
### mergify @ 2025-09-10T18:50:34Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Lucaskabela.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-12T13:58:17Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Lucaskabela.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Lucaskabela @ 2025-09-17T18:49:17Z

One thing I want to get ahead of here - since we have torch.compile on lower level modules, that may have multiple instances (for example, the vision block here), each block is getting compiled.  This does not lead to recompiles, as there may be different parameters, but there is more logs (i.e:

```
[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:00 [backends.py:537] vLLM's torch.compile cache is disabled.
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:00 [backends.py:550] Dynamo bytecode transform time: 0.59 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:00 [backends.py:215] Compiling a graph for dynamic shape takes 0.07 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:00 [monitor.py:34] torch.compile takes 0.67 s in total
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:537] vLLM's torch.compile cache is disabled.
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:550] Dynamo bytecode transform time: 0.34 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:215] Compiling a graph for dynamic shape takes 0.15 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [monitor.py:34] torch.compile takes 1.15 s in total
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:537] vLLM's torch.compile cache is disabled.
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:550] Dynamo bytecode transform time: 0.25 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:03 [backends.py:215] Compiling a graph for dynamic shape takes 0.08 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:04 [monitor.py:34] torch.compile takes 1.48 s in total
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:04 [backends.py:537] vLLM's torch.compile cache is disabled.
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:04 [backends.py:550] Dynamo bytecode transform time: 0.25 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:04 [backends.py:215] Compiling a graph for dynamic shape takes 0.07 s
^[...
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:18 [monitor.py:34] torch.compile takes 12.52 s in total
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:22 [backends.py:537] vLLM's torch.compile cache is disabled.
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:22 [backends.py:550] Dynamo bytecode transform time: 3.76 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:25 [backends.py:215] Compiling a graph for dynamic shape takes 2.73 s
^[[1;36m(EngineCore_DP0 pid=2099196)^[[0;0m INFO 09-16 13:09:26 [monitor.py:34] torch.compile takes 19.02 s in total
```

Want to call this out to see if there is any suggestion to fold this to lead to less log spamming, or if we can follow that up in another PR
---
### wwl2755 @ 2025-09-17T22:28:05Z

Thanks for the great work! 

I'm wondering what would be the main concern to split the large nn module into smaller ones. Would it cause any problem or is it leading better optimization? I have investigated some other models before in torch.compile with a simpler format but it seems to give less optimization than this one. (https://github.com/vllm-project/vllm/compare/main...wwl2755:vllm:mm-torch-compile with ~2% improvement in overall throughput/TTFT) 
---
### Lucaskabela @ 2025-09-17T22:50:11Z

> Thanks for the great work!
> 
> I'm wondering what would be the main concern to split the large nn module into smaller ones. Would it cause any problem or is it leading better optimization? I have investigated some other models before in torch.compile with a simpler format but it seems to give less optimization than this one. ([main...wwl2755:vllm:mm-torch-compile](https://github.com/vllm-project/vllm/compare/main...wwl2755:vllm:mm-torch-compile) with ~2% improvement in overall throughput/TTFT)

When we put compile at the top level, there may be some code that torch compile's dynamo is not able to trace.  In these instances, we have a graph break; however, graph breaks mean that we will have less opportunities to optimize the generated graph.  

So, by compiling the smaller modules as opposed to the top level one, we are able to get more improvements to the runtime 

Essentially: We want to be able to apply compile to the highest level that we can that is traceable (not containing graph breaks).  This PR gives us that flexibility, so we can get decent performance improvements without difficult rewrites or rearchitectures 
---
### wwl2755 @ 2025-09-18T00:10:54Z

> When we put compile at the top level, there may be some code that torch compile's dynamo is not able to trace. In these instances, we have a graph break; however, graph breaks mean that we will have less opportunities to optimize the generated graph.
> 
> So, by compiling the smaller modules as opposed to the top level one, we are able to get more improvements to the runtime
> 
> Essentially: We want to be able to apply compile to the highest level that we can that is traceable (not containing graph breaks). This PR gives us that flexibility, so we can get decent performance improvements without difficult rewrites or rearchitectures

Thanks for the explanation. It's a great first step on supporting torch.compile in the VIT. 
---
### DarkLight1337 @ 2025-09-26T09:19:10Z

I wonder how much improvement this makes over https://github.com/vllm-project/vllm/pull/23932?
---
### Lucaskabela @ 2025-09-29T21:40:00Z

> I wonder how much improvement this makes over #23932?

@DarkLight1337 I went ahead and changed L477 (in VisionBlock) to use iadd:
```
        residual += self.mlp(x_fused_norm)
        return residual
```
I didn't identify any other places to do this in this model; running the benchmarks this is the result:
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  137.17    
Total input tokens:                      94327     
Total generated tokens:                  106235    
Request throughput (req/s):              7.29      
Output token throughput (tok/s):         774.48    
Peak output token throughput (tok/s):    14442.00  
Peak concurrent requests:                1000.00   
Total Token throughput (tok/s):          1462.16   
---------------Time to First Token----------------
Mean TTFT (ms):                          65812.25  
Median TTFT (ms):                        65105.06  
P99 TTFT (ms):                           131395.06 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          686.31    
Median TPOT (ms):                        667.80    
P99 TPOT (ms):                           1405.43   
---------------Inter-token Latency----------------
Mean ITL (ms):                           598.41    
Median ITL (ms):                         60.96     
P99 ITL (ms):                            1604.00   
==================================================
```
Since it is only the one place I don't see any sig. change vs mainline
---
### DarkLight1337 @ 2025-09-30T02:52:01Z

> @DarkLight1337 I went ahead and changed L477 (in VisionBlock) to use iadd:

Did you use CUDAGraph for the benchmark? The original PR mentioned that add is optimized to iadd when CUDAGraph is used which might explain why. I wonder whether we can get similar improvements by just changing add to iadd without CUDAGraph.
---
### Lucaskabela @ 2025-09-30T15:55:54Z

> > @DarkLight1337 I went ahead and changed L477 (in VisionBlock) to use iadd:
> 
> Did you use CUDAGraph for the benchmark? The original PR mentioned that add is optimized to iadd when CUDAGraph is used which might explain why. I wonder whether we can get similar improvements by just changing add to iadd without CUDAGraph.

Yep, just double checked Cudagraphs are on - here is the relevant fields in config to double check:
```
cudagraph_mode":[2,1],"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], "cudagraph_copy_inputs":false,"full_cuda_graph":false,
```
---
### Lucaskabela @ 2025-10-02T15:33:59Z

Sounds good - thanks @ywang96!
---
### mergify @ 2025-10-04T00:14:07Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Lucaskabela.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Lucaskabela @ 2025-10-07T21:52:22Z

Hey @ywang96  any update on testing this change? Would love to cross validate the improvements and land this 
---
### yanyongyu @ 2025-10-10T11:47:13Z

After making a patch on our local version of vLLM, we gain about 20% vision encoder speedup (result from torch profiler) in some test cases. 👍🏻 
---
### ywang96 @ 2025-10-10T23:50:06Z

@Lucaskabela Thanks for the ping - I'm okay with us getting this in for now!
---
### Lucaskabela @ 2025-10-11T02:23:44Z

Ah, I think this PR needs to be slightly reworked to still be compatible with previous model defs (I thought it was but seems CI tells a different story) - I will burn down these failures by Monday 
---
### mergify @ 2025-10-13T13:45:07Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Lucaskabela.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### miladm @ 2025-10-14T22:30:14Z

Thanks @Lucaskabela! Can we reporting the MFU gains before and after this PR?
---
### ProExpertProg @ 2025-10-15T01:29:23Z

@Lucaskabela if we want to get this merged for this release we should make sure we test with Inductor partitioning to make sure there aren't any hidden surprises.
---
### mergify @ 2025-10-15T03:03:25Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Lucaskabela.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Lucaskabela @ 2025-10-15T16:51:12Z

> @Lucaskabela if we want to get this merged for this release we should make sure we test with Inductor partitioning to make sure there aren't any hidden surprises.

Do we have an example command to run to do this?
---
### Lucaskabela @ 2025-10-16T16:45:04Z

> > @Lucaskabela if we want to get this merged for this release we should make sure we test with Inductor partitioning to make sure there aren't any hidden surprises.
> 
> Do we have an example command to run to do this?

`-O.use_inductor_graph_partition=True` - seems this does work with the multimodal model compiled (this fails with the language_model compiled on main, both on this PR and on main so I commented that out to test)
---
### Lucaskabela @ 2025-10-16T17:42:45Z

And now for the fun part - seems after these changes there is an Eagle3 test failing.  I tried bisecting these changes to figure out what causes it to break, as it seems like something is causing the language model to compile

Turns out, this is caused by the import `set_model_tag` at the top level, and further, we need this `set_model_tag` because prefix is not respected by the backend

I have filed two issues to follow up on so that we can harden this integration - see #27044  and #27045
---
### Lucaskabela @ 2025-10-16T19:55:29Z

Seems this test is broken on main so should be good to land :) 
---
### Lucaskabela @ 2025-10-16T21:33:13Z

> Thanks @Lucaskabela! Can we reporting the MFU gains before and after this PR?

MFU Estimate (from) Throughput * Flops per token / Max theoretical

Throughput changes from (base) 1478.44 to (change) 1548.80 
We estimate Flops per token as: 2 (forward inference) * 3 * 10**9 (params) ~ 6e9 
Max theoretical for 1 x H100 in fp32 is approximately 83 TeraFlops or 8.3 * 10^13

So then, 
MFU Before: ~ 10.5%
MFU After: ~11.2%

So we have roughly a 6% relative improvement in MFU here



---
### zou3519 @ 2025-10-27T19:39:50Z

trunk is a bit broken, this might need a force merge
---
### JartX @ 2025-10-29T10:09:31Z

Hi! @Lucaskabela @ywang96  This PR completely breaks the inference when I send an image to the model, for example I send it an invoice, and it tells me that it cannot see absolutely anything.
---
### JartX @ 2025-10-29T10:11:09Z

@lgeiger can help with it? https://github.com/vllm-project/vllm/pull/27705, also have revert your PR because it fails me with images
---
### DarkLight1337 @ 2025-10-29T10:11:12Z

Does https://github.com/vllm-project/vllm/pull/27705 fix the issue?
---
### ZJY0516 @ 2025-10-29T10:11:13Z

> Hi! @Lucaskabela @ywang96 This PR completely breaks the inference when I send an image to the model, for example I send it an invoice, and it tells me that it cannot see absolutely anything.

Could you provide a repro script?
---
### JartX @ 2025-10-29T10:15:42Z

@DarkLight1337 no, only have revert the https://github.com/vllm-project/vllm/pull/27705 to avoid problems with this: https://github.com/vllm-project/vllm/pull/23207

@ZJY0516

 VLLM_ATTENTION_BACKEND=TORCH_SDPA lm_eval
--model local-chat-completions
--model_args model=Qwen/Qwen3-VL-30B-A3B-Instruct,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=10,max_retries=3,tokenized_requests=False
--tasks chartqa
--batch_size auto
--apply_chat_template
After start vllm server 


---
### JartX @ 2025-10-29T10:17:39Z

@ZJY0516 You can also try sending an image. If I send you the following: https://ibb.co/XZVsT7dB
It says it sees absolutely nothing. Just a black background with wavy lines.

It's this PR: https://github.com/vllm-project/vllm/pull/23207 where we are that has broken my inference, not @lgeiger's.
---
### lgeiger @ 2025-10-29T10:23:48Z

@JartX what Hardware are you testing this on? 
---
### JartX @ 2025-10-29T10:31:33Z

@lgeiger 
AMD ROCM RDNA3 Attn Backend ViT TORCH.SDPA

My command of docker run:

docker run -d --name vllm1 --tty --restart unless-stopped --shm-size 48gb -p 80:8000 --device /dev/kfd --device /dev/dri --group-add video --ipc host --network host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged -v ./chat_vl.jinja:/chat-template-tools.jinja -e HSA_OVERRIDE_GFX_VERSION=11.0.0 -e VLLM_USE_V1=1 -e VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 -e MIOPEN_USER_DB_PATH=/apps/miopen-cache -e MIOPEN_CUSTOM_CACHE_DIR=/apps/miopen-cache -e OMP_NUM_THREADS=1 -e VLLM_ENABLE_V1_MULTIPROCESSING=1 vllm-rocm-251029_broken_test_1 vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct --gpu-memory-utilization 0.98 --max_model_len 65536 -tp 4 --served-model-name QWEN3--port 80 --limit-mm-per-prompt '{"image":6,"video":0}' --dtype float16 --enable-log-requests --chat-template /chat-template-tools.jinja 
---
### DarkLight1337 @ 2025-10-29T10:39:50Z

cc @tjtanaa for AMD stuff
---
### lgeiger @ 2025-10-29T11:46:10Z

@JartX Maybe we need to add back the rocm-only `contiguous()` https://github.com/vllm-project/vllm/pull/23207/files#diff-19c9c9ac2ca98754bf900c292a95b9930f352f63f8619e3e0026a8e31f3e4396L432-L437?
---
### JartX @ 2025-10-29T12:21:03Z

Hi @lgeiger could you add the contiguous on torch.sdpa on your pr please?

https://github.com/vllm-project/vllm/pull/27741/files
---
### lgeiger @ 2025-10-29T12:41:40Z

>Hi @lgeiger could you add the contiguous on torch.sdpa on your pr please?

@JartX I don't have access to AMD hardware to test. So would appreciate if you or somebody else with the ability to test would make a PR with these changes.
---
### tjtanaa @ 2025-10-29T12:49:32Z

I will be debugging on Mi300X.
---
### JartX @ 2025-10-29T12:53:16Z

adding

 if current_platform.is_rocm():
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                
                works again
                
                
 @tjtanaa @DarkLight1337 @lgeiger I'm going to do a PR express thanks @lgeiger 
 
 
---
### tjtanaa @ 2025-10-29T12:55:26Z

@Jartx go ahead and open that bugfix PR. For other path on ROCm, we still need to do some fixing.
---