# PR #12755: [Model][Speculative Decoding] DeepSeek MTP spec decode

- field: decoding  | area: model_executor/models
- author: luccafong
- state: MERGED  created: 2025-02-04T23:58:11Z  merged: 2025-02-19T09:06:23Z
- url: https://github.com/vllm-project/vllm/pull/12755
- labels: speculative-decoding, ready, ci/build

## Body

Implement DeepSeek MTP: https://github.com/vllm-project/vllm/issues/12181 to support DeepSeek MTP layers for next n prediction.

**Online Serving**
Add `--num-speculative-tokens 1` for DeepSeek V3/R1:
 
```
python -m vllm.entrypoints.openai.api_server --disable-log-requests --gpu-memory-utilization 0.8  --max-model-len 65536 --max-num-seqs 128 --seed 0 --tensor-parallel-size 8 --model deepseek-ai/DeepSeek-R1 ---trust-remote-code --num-speculative-tokens 1
```
**Offline Inference**
Set  num_speculative_tokens = 1

```
llm = LLM(
    model="deepseek-ai/DeepSeek-R1",
    tensor_parallel_size=8,
    max_model_len=8192, # If you have enough memory with your hardware, you can ignore this
    num_speculative_tokens=1, # only 1 is supported for now
   draft_tensor_parallel_size=8, # optional, by default it will be the same as tensor_parallel_size.
)
```

Note: This implementation validates on MTP k=1 models only.


**Benchmark Results**

The acceptance rate is 81% ~ 82.3% on R1 k=1. 
The speedup depends on the QPS, with 1.63x speedup for QPS=1 and certain improvement with QPS<8 as shown in below table.

**Results on various QPS**

Draft TP=1

| QPS | Baseline TPOT | k=1 TPOT  | Speedup |
| --- | ------------- | --------- | ------- |
| 1   | 55.47         | 33.99     |   1.63x |
| 2   | 57.58         | 48.8      |   1.18x |
| 4   | 64.29         | 51.02     |   1.26x | 
| 6   | 122.93        | 108.15    |   1.14x |
| 8   | 120.18        | 119.14    |    1.0x | 


Draft TP=8

| QPS | Baseline TPOT | k=1,TP=8 TPOT | Speedup 
| --- | ------------- | --------- | ------- |
| 1   | 55.47         | 32.64     |   1.69x |
| 2   | 57.58         | 43.6      |   1.32x |
| 4   | 64.29         | 52.62     |   1.22x | 
| 6   | 122.93        | 129.5     |   < 1.0 |
| 8   | 120.18        | 139.49    |   < 1.0 | 



**Results on various Concurrency**
Draft TP=8

| MAX_CONCURRENCY | Baseline TPOT | k=1 TPOT | Speedup |
| --- | ------------- | --------- | ------- | 
| 1   | 23.13         | 17.24     |   1.34x |
| 2   | 28.10         | 17.07     |   1.64x |
| 4   | 27.57         | 21.48     |   1.28x | 
| 8   | 38.57         | 34.62     |   1.11x |
| 16   | 50.24        | 40.89    |    1.22x | 
| 32   | 70.88        | 56.63    |    1.25x | 



## Comments

### github-actions @ 2025-02-04T23:58:22Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### Neo9061 @ 2025-02-05T16:39:01Z

@luccafong Sorry have to ask those questions as I hope to use your implementation.

1. Have you tested it e2e with VLLM's multi-node distributed inference setting? asking as I can only deploy the model in multi-node settings.
2. If I want to re-use the MTP head to do speculation length k > 1, what is the hacking implementation you would recommend to just make it work? as k=1 is too limited in my application.
---
### benchislett @ 2025-02-05T18:30:43Z

@luccafong I have been working on a similar implementation locally, and have faced a few challenges that I'm not sure are addressed here. Have you validated the acceptance rate for k=1 for real weights?

I believe that the final RMSNorm in the DeepSeekV3 main model is not necessary for speculative decoding since the `hnorm` already normalizes the previous hidden weights received from the main model. It's unclear to me how it is classified in the DeepSeek-V3 technical report, but I think that the norm might be included in the output head and therefore not normalized as input to the MTP module. Anecdotally, I observe a small increase in acceptance rate with this change.

Also, I have noticed the acceptance rate becomes very low (<50%) when I enable the recently added MLA attention. Have you noticed this also? I am not sure what could cause this, maybe it is a bug fixed in recent commits to vLLM. I would like to know if this is an issue for your implementation.
---
### luccafong @ 2025-02-06T04:54:01Z

> @luccafong I have been working on a similar implementation locally, and have faced a few challenges that I'm not sure are addressed here. Have you validated the acceptance rate for k=1 for real weights?
> 
> I believe that the final RMSNorm in the DeepSeekV3 main model is not necessary for speculative decoding since the `hnorm` already normalizes the previous hidden weights received from the main model. It's unclear to me how it is classified in the DeepSeek-V3 technical report, but I think that the norm might be included in the output head and therefore not normalized as input to the MTP module. Anecdotally, I observe a small increase in acceptance rate with this change.
> 
> Also, I have noticed the acceptance rate becomes very low (<50%) when I enable the recently added MLA attention. Have you noticed this also? I am not sure what could cause this, maybe it is a bug fixed in recent commits to vLLM. I would like to know if this is an issue for your implementation.

The accept rate is around 56% during my testing, MLA attention could lead to different branch, 
https://github.com/luccafong/vllm/blob/ds_mtp/vllm/spec_decode/multi_step_worker.py#L98 I fixed in a later commit.

regarding the norm, thanks for pointing out, let me try adjusting to see if there is an improvement.
---
### luccafong @ 2025-02-06T04:57:29Z

> @luccafong Sorry have to ask those questions as I hope to use your implementation.
> 
> 1. Have you tested it e2e with VLLM's multi-node distributed inference setting? asking as I can only deploy the model in multi-node settings.
> 2. If I want to re-use the MTP head to do speculation length k > 1, what is the hacking implementation you would recommend to just make it work? as k=1 is too limited in my application.

1.Not tested with multi node settings; 2. We can reuse if you do some model processing, e.g. duplicate the weights to different layers, the hacky changes will not be proper since for n predict layers >1, and we do a k > n predict layers, it is difficult to decide which layer to forward multiple times. 
Note for now as commented in the other thread, some changes are needed for K>1, I am working in progress, let me update with you if it works.
---
### mergify @ 2025-02-06T07:17:03Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @luccafong.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Neo9061 @ 2025-02-06T17:55:50Z

> > @luccafong Sorry have to ask those questions as I hope to use your implementation.
> > 
> > 1. Have you tested it e2e with VLLM's multi-node distributed inference setting? asking as I can only deploy the model in multi-node settings.
> > 2. If I want to re-use the MTP head to do speculation length k > 1, what is the hacking implementation you would recommend to just make it work? as k=1 is too limited in my application.
> 
> 1.Not tested with multi node settings; 2. We can reuse if you do some model processing, e.g. duplicate the weights to different layers, the hacky changes will not be proper since for n predict layers >1, and we do a k > n predict layers, it is difficult to decide which layer to forward multiple times. Note for now as commented in the other thread, some changes are needed for K>1, I am working in progress, let me update with you if it works.

Thank you for your reply! Are you testing with single node of H200 then to load the target model?

I just found a bug to use `VLLM` class to load target model via distributed inference (official Ray setup) with speculative decoding arguments passed in. Without those arguments, the target model is loaded successfully. 

https://github.com/vllm-project/vllm/issues/12841
---
### WhatGhost @ 2025-02-08T03:28:06Z

@luccafong 
Nice work,inspired me a lot!
I have two very small problems that have been bothering me. 
1. I notice you use luccafong/deepseek_mtp_main_random model . I wonder is it only a small size deepseek-like model for testing?
2. How can i run a deepseek-v3 model with mtp？ Just set  speculative_model to deepseek-v3 it own?

Thanks!
---
### LiuXiaoxuanPKU @ 2025-02-10T07:09:11Z

> @luccafong Nice work,inspired me a lot! I have two very small problems that have been bothering me.
> 
> 1. I notice you use luccafong/deepseek_mtp_main_random model . I wonder is it only a small size deepseek-like model for testing?
> 2. How can i run a deepseek-v3 model with mtp？ Just set  speculative_model to deepseek-v3 it own?
> 
> Thanks!

Yes you can just plugin in deepseek model path as shown below, the following is my setup on 8xH200

```
llm = LLM(
    model="/path/to/DeepSeek-R1",
    tensor_parallel_size=8,
    max_model_len=8192, # If you have enough memory with your hardware, you can ignore this
    num_speculative_tokens=1, # only 1 is supported for now
)
```


---
### QualityGN @ 2025-02-16T09:39:50Z

Hi, have you replicated the inference acceleration effect after enabling MTP on multiple nodes？
My envs:
Ray cluster: two nodes of 8 x H20, ray status is right.

It turns out that I got low throughput (approximately 8.5 requests/sec on average) and high inference latency (100+ ms).
---
### fan-niu @ 2025-02-19T13:31:31Z

@luccafong Hi thanks for your great work!  I ran deepseek-r1 in 2 x 8 H100 ray clusters, but encountered CUDA error: invalid device ordinal, could you help take a look this issue? The ray status is normal, thanks

Run script:
```
vllm serve  deepseek-ai/DeepSeek-R1\
        --host 0.0.0.0 \
        --port 8081 \
        --tensor-parallel-size 16 \
        --pipeline-parallel-size 1 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 16 \
        --max-model-len 32768 \
        --served-model-name deepseek_r1 \
        --device cuda \
        --quantization fp8 \
        --trust-remote-code \
        --num-speculative-tokens 1
```

Error message:
```ERROR 02-19 05:03:47 engine.py:389] ^[[36mray::RayWorkerWrapper.execute_method()^[[39m (pid=5505, ip=10.0.0.40, actor_id=8d2af607296694817b9e0c1504000000, repr=<vllm.executor.ray_utils.RayWorkerWrapper object at 0x7f8791323590>)^M
ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 582, in execute_method^M
ERROR 02-19 05:03:47 engine.py:389]     raise e^M
ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 573, in execute_method^M
ERROR 02-19 05:03:47 engine.py:389]     return run_method(target, method, args, kwargs)^M
ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2196, in run_method^M
ERROR 02-19 05:03:47 engine.py:389]     return func(*args, **kwargs)^M
ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^M
ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/spec_decode/spec_decode_worker.py", line 368, in init_device^M
ERROR 02-19 05:03:47 engine.py:389]     self.spec_decode_sampler.init_tensors(self.rank,^M
ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/spec_decode_base_sampler.py", line 56, in init_tensors^M
ERROR 02-19 05:03:47 engine.py:389]     self.num_accepted_tokens = torch.tensor(0,^M
ERROR 02-19 05:03:47 engine.py:389]                                ^^^^^^^^^^^^^^^^M
ERROR 02-19 05:03:47 engine.py:389] RuntimeError: CUDA error: invalid device ordinal^M
ERROR 02-19 05:03:47 engine.py:389] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.^M
ERROR 02-19 05:03:47 engine.py:389] For debugging consider passing CUDA_LAUNCH_BLOCKING=1^M
ERROR 02-19 05:03:47 engine.py:389] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.^M
```


---
### Neo9061 @ 2025-02-19T14:00:00Z

> @luccafong Hi thanks for your great work! I ran deepseek-r1 in 2 x 8 H100 ray clusters, but encountered CUDA error: invalid device ordinal, could you help take a look this issue? The ray status is normal, thanks
> 
> Run script:
> 
> ```
> vllm serve  deepseek-ai/DeepSeek-R1\
>         --host 0.0.0.0 \
>         --port 8081 \
>         --tensor-parallel-size 16 \
>         --pipeline-parallel-size 1 \
>         --gpu-memory-utilization 0.8 \
>         --max-num-seqs 16 \
>         --max-model-len 32768 \
>         --served-model-name deepseek_r1 \
>         --device cuda \
>         --quantization fp8 \
>         --trust-remote-code \
>         --num-speculative-tokens 1
> ```
> 
> Error message:
> 
> ```
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 582, in execute_method^M
> ERROR 02-19 05:03:47 engine.py:389]     raise e^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 573, in execute_method^M
> ERROR 02-19 05:03:47 engine.py:389]     return run_method(target, method, args, kwargs)^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2196, in run_method^M
> ERROR 02-19 05:03:47 engine.py:389]     return func(*args, **kwargs)^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/spec_decode/spec_decode_worker.py", line 368, in init_device^M
> ERROR 02-19 05:03:47 engine.py:389]     self.spec_decode_sampler.init_tensors(self.rank,^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/spec_decode_base_sampler.py", line 56, in init_tensors^M
> ERROR 02-19 05:03:47 engine.py:389]     self.num_accepted_tokens = torch.tensor(0,^M
> ERROR 02-19 05:03:47 engine.py:389]                                ^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389] RuntimeError: CUDA error: invalid device ordinal^M
> ERROR 02-19 05:03:47 engine.py:389] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.^M
> ERROR 02-19 05:03:47 engine.py:389] For debugging consider passing CUDA_LAUNCH_BLOCKING=1^M
> ERROR 02-19 05:03:47 engine.py:389] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.^M
> ```

This is a separate bug. Please see suggested fix in here: https://github.com/vllm-project/vllm/issues/12841
---
### Neo9061 @ 2025-02-19T14:01:24Z

@luccafong @LiuXiaoxuanPKU Any suggestion whether it would be an easy hack on your implementation to re-use the single MTP, like EAGLE draft model, for speculation number more than 1? Many thanks
---
### youkaichao @ 2025-02-19T14:14:44Z

@fan-niu please check if #12841 solves the problem!
---
### fan-niu @ 2025-02-19T16:04:55Z

@Neo9061 @youkaichao Thanks a lot, it is normal now! 
---
### yangchou19 @ 2025-02-24T15:40:57Z

> @luccafong Hi thanks for your great work! I ran deepseek-r1 in 2 x 8 H100 ray clusters, but encountered CUDA error: invalid device ordinal, could you help take a look this issue? The ray status is normal, thanks
> 
> Run script:
> 
> ```
> vllm serve  deepseek-ai/DeepSeek-R1\
>         --host 0.0.0.0 \
>         --port 8081 \
>         --tensor-parallel-size 16 \
>         --pipeline-parallel-size 1 \
>         --gpu-memory-utilization 0.8 \
>         --max-num-seqs 16 \
>         --max-model-len 32768 \
>         --served-model-name deepseek_r1 \
>         --device cuda \
>         --quantization fp8 \
>         --trust-remote-code \
>         --num-speculative-tokens 1
> ```
> 
> Error message:
> 
> ```
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 582, in execute_method^M
> ERROR 02-19 05:03:47 engine.py:389]     raise e^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker_base.py", line 573, in execute_method^M
> ERROR 02-19 05:03:47 engine.py:389]     return run_method(target, method, args, kwargs)^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2196, in run_method^M
> ERROR 02-19 05:03:47 engine.py:389]     return func(*args, **kwargs)^M
> ERROR 02-19 05:03:47 engine.py:389]            ^^^^^^^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/spec_decode/spec_decode_worker.py", line 368, in init_device^M
> ERROR 02-19 05:03:47 engine.py:389]     self.spec_decode_sampler.init_tensors(self.rank,^M
> ERROR 02-19 05:03:47 engine.py:389]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/spec_decode_base_sampler.py", line 56, in init_tensors^M
> ERROR 02-19 05:03:47 engine.py:389]     self.num_accepted_tokens = torch.tensor(0,^M
> ERROR 02-19 05:03:47 engine.py:389]                                ^^^^^^^^^^^^^^^^M
> ERROR 02-19 05:03:47 engine.py:389] RuntimeError: CUDA error: invalid device ordinal^M
> ERROR 02-19 05:03:47 engine.py:389] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.^M
> ERROR 02-19 05:03:47 engine.py:389] For debugging consider passing CUDA_LAUNCH_BLOCKING=1^M
> ERROR 02-19 05:03:47 engine.py:389] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.^M
> ```

I encountered an **“out of memory”** error while running DeepSeek-R1 on 2 machines with 8 *H20 GPUs each.  Has anyone successfully run DeepSeek-R1 on H20 GPUs? 

Here is the command I used:
```
 vllm serve  deepseek-ai/DeepSeek-R1\
        --host 0.0.0.0 \
        --port 8081 \
        --tensor-parallel-size 16 \
        --pipeline-parallel-size 1 \
        --gpu-memory-utilization 0.99 \
        --max-model-len 131072 \
        --served-model-name deepseek_r1 \
        --quantization fp8 \
        --trust-remote-code \
        --num-speculative-tokens 1
```
---
### benchislett @ 2025-02-24T16:03:43Z

@yangchou19 Your gpu memory utilization is set too high. As I understand, the weights for speculative decoding are not currently accounted-for in the memory profiler so there must be additional memory leftover from the vllm allocation for it to live in. As a workaround, try decreasing the gpu memory utilization progressively (try 0.95, 0.9, 0.88, 0.85) until it succeeds. If you want to use speculative decoding, you may need to decrease max-model-len to ensure there is enough memory available for a large KV cache.
---
### hxt365 @ 2025-02-27T12:26:45Z

Is this applicable for R1 Distill models? I got this error for deepseek-ai/DeepSeek-R1-Distill-Qwen-32B: ValueError: num_speculative_tokens was provided without speculative_model
---
### mgoin @ 2025-02-27T13:39:46Z

@hxt365 No the Distill models do not have MTP modules like DeepSeek V3/R1
---
### KiroSummer @ 2025-03-01T04:01:53Z

Great work on the analysis! I wanted to clarify one point regarding the baseline vs. MTP performance comparison. Given that the speculative decoding worker implementation doesn't support asynchronous output processing or multi-step scheduling, could you confirm whether these two optimizations were utilized when calculating the TPOT metric for the baseline model? 
---
### JoeyYoung @ 2025-03-03T05:21:26Z

Hi all, I try to use speculative decoding with tp=8 and pp=2 on the 2 x 8H20 testbed, with following command:
```
vllm serve /vllm-workspace/DeepSeek-R1/ \
        --host 0.0.0.0 \
        --port 8081 \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 2 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 16 \
        --max-model-len 32768 \
        --served-model-name deepseek_r1 \
        --device cuda \
        --quantization fp8 \
        --trust-remote-code \
        --num-speculative-tokens 1
```
But it reports the error:
```
INFO 02-28 00:11:01 [config.py:334] Overriding HF config with <function SpeculativeConfig.hf_config_override at 0x7f1bcd2a7880>
Traceback (most recent call last):
  File "/root/anaconda3/envs/vllm/bin/vllm", line 8, in <module>
    sys.exit(main())
  File "/vllm-workspace/vllm/vllm/entrypoints/cli/main.py", line 73, in main
    args.dispatch_function(args)
  File "/vllm-workspace/vllm/vllm/entrypoints/cli/serve.py", line 34, in cmd
    uvloop.run(run_server(args))
  File "/root/anaconda3/envs/vllm/lib/python3.10/site-packages/uvloop/__init__.py", line 82, in run
    return loop.run_until_complete(wrapper())
  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
  File "/root/anaconda3/envs/vllm/lib/python3.10/site-packages/uvloop/__init__.py", line 61, in wrapper
    return await main
  File "/vllm-workspace/vllm/vllm/entrypoints/openai/api_server.py", line 946, in run_server
    async with build_async_engine_client(args) as engine_client:
  File "/root/anaconda3/envs/vllm/lib/python3.10/contextlib.py", line 199, in __aenter__
    return await anext(self.gen)
  File "/vllm-workspace/vllm/vllm/entrypoints/openai/api_server.py", line 138, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
  File "/root/anaconda3/envs/vllm/lib/python3.10/contextlib.py", line 199, in __aenter__
    return await anext(self.gen)
  File "/vllm-workspace/vllm/vllm/entrypoints/openai/api_server.py", line 162, in build_async_engine_client_from_engine_args
    engine_client = AsyncLLMEngine.from_engine_args(
  File "/vllm-workspace/vllm/vllm/engine/async_llm_engine.py", line 639, in from_engine_args
    engine_config = engine_args.create_engine_config(usage_context)
  File "/vllm-workspace/vllm/vllm/engine/arg_utils.py", line 1237, in create_engine_config
    speculative_config = SpeculativeConfig.maybe_create_spec_config(
  File "/vllm-workspace/vllm/vllm/config.py", line 2025, in maybe_create_spec_config
    return SpeculativeConfig(
  File "/vllm-workspace/vllm/vllm/config.py", line 2199, in __init__
    self._verify_args()
  File "/vllm-workspace/vllm/vllm/config.py", line 2207, in _verify_args
    self.draft_model_config.verify_with_parallel_config(
  File "/vllm-workspace/vllm/vllm/config.py", line 762, in verify_with_parallel_config
    raise NotImplementedError(
NotImplementedError: Pipeline parallelism is not supported for this model. Supported models implement the `SupportsPP` interface.
```

Is pipeline parallelism not supported for the draft model?

---
### BoyuanS @ 2025-03-06T05:15:51Z


![11](https://github.com/user-attachments/assets/a39b33bc-f7c1-409b-8ce5-ff6a5f21db47)

When I started deepseek-r1-awq int4 and set --num-speculative-tokens 1, generated contents are all empty after <think>.  Any idea why this should happen? The model works well when num-speculative-tokens is not set
---
### benchislett @ 2025-03-06T15:44:10Z

@BoyuanS It is likely that your AWQ quantization did not include weights for the MTP head, in which case this will not work. 
---
### parambole @ 2025-07-09T03:35:23Z

Hey @luccafong @mgoin @LiuXiaoxuanPKU  I am currently working on [integrating](https://github.com/AI-Hypercomputer/maxtext/pull/1837) Deepseek's Multi-Token Prediction into Maxtext.

**Question:**

As part of this PR, has the team been able to load the open MTP Deepseek V3 weights and analyze the implementation during pre-training and fine-tuning? I am curious to know about the observed behavior?
---
### mgoin @ 2025-07-09T23:59:31Z

I have no experience with training, sorry!
---
### parambole @ 2025-07-10T02:39:59Z

> I have no experience with training, sorry!

Hey @mgoin - thanks for responding. With respect to inference how is the performance of the predicted token with loading the deepseek MTP published weights ?

Specifically I am observing: https://github.com/deepseek-ai/DeepSeek-V3/issues/928
---