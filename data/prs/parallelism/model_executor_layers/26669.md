# PR #26669: support flashinfer_fp4 moe for 5090 gpu

- field: quantization  | area: model_executor/layers
- author: XiaobingSuper
- state: MERGED  created: 2025-10-13T02:46:01Z  merged: 2025-10-15T19:06:47Z
- url: https://github.com/vllm-project/vllm/pull/26669
- labels: ready

## Body

<!-- markdownlint-disable -->

## Purpose
This PR is to enable lashinfer_fp4 to work for 5090 gpu device.

## Test Plan

```python

import numpy
import torch

from vllm import LLM, SamplingParams


prompts = [
    "The Swiss Alps are", 
    "Brad Marchand is",
    "The Boston Bruins are"
]


sampling_params = SamplingParams(temperature=0.90, top_p=0.95, max_tokens=40, min_tokens=10)
llm = LLM("nv-community/Qwen3-30B-A3B-FP4", tensor_parallel_size=1, enforce_eager=True)

responses = llm.generate(prompts, sampling_params)
for output in responses:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```


```bash
VLLM_USE_FLASHINFER_MOE_FP4=1 VLLM_USE_V1=1 python qwen_fp4_test.py
```

## Test Result

```bash 
Prompt: 'The Swiss Alps are', Generated text: ' a mountain range in which country?\n\nThe Swiss Alps are a mountain range located in Switzerland. They are part of the larger Alps mountain system, which stretches across several European countries. The Swiss Alps are particularly'
Prompt: 'Brad Marchand is', Generated text: ' one of the best goal scorers in the NHL. But he’s also one of the best at drawing penalties. For the second straight season, he’s leading the league in penalty minutes. But it'
Prompt: 'The Boston Bruins are', Generated text: ' a professional ice hockey team based in Boston, Massachusetts. They are members of the Atlantic Division of the Eastern Conference in the National Hockey League (NHL). The team was founded in 192'
```


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

### XiaobingSuper @ 2025-10-14T07:45:00Z

@yewentao256 could you help merge it? Thanks!
---
###  @ 2025-10-14T11:37:31Z

I tested it. It only works when enforce_eager=True.

And it's very slow.
---
### XiaobingSuper @ 2025-10-14T11:55:12Z

> I tested it. It only works when enforce_eager=True.
> 
> And it's very slow.

I can work when not set  enforce_eager=True:
```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████████████████████████████████████████| 67/67 [00:08<00:00,  8.18it/s]
Capturing CUDA graphs (decode, FULL): 100%|███████████████████████████████████████████████████████████| 35/35 [00:04<00:00,  7.59it/s]
(EngineCore_DP0 pid=3407874) INFO 10-14 19:49:24 [gpu_model_runner.py:3778] Graph capturing finished in 14 secs, took 1.03 GiB
(EngineCore_DP0 pid=3407874) INFO 10-14 19:49:24 [core.py:240] init engine (profile, create kv cache, warmup model) took 28.01 seconds
(EngineCore_DP0 pid=3407874) INFO 10-14 19:49:25 [gc_utils.py:40] GC Debug Config. enabled:False,top_objects:-1
INFO 10-14 19:49:25 [llm.py:337] Supported tasks: ['generate']
Adding requests: 100%|██████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 93.67it/s]
Processed prompts: 100%|█████████████████████████| 1/1 [00:00<00:00,  2.64it/s, est. speed input: 10.58 toks/s, output: 105.79 toks/s]
Prompt: 'The Swiss Alps are', Generated text: ' a mountain range in the southern part of Switzerland, forming a natural border between the country and Italy. The Alps are the highest and most extensive mountain range in Europe, stretching across several countries including Switzerland,'
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2272.10it/s]
Processed prompts: 100%|█████████████████████████| 1/1 [00:00<00:00,  2.84it/s, est. speed input: 11.36 toks/s, output: 113.62 toks/s]
Prompt: 'Brad Marchand is', Generated text: ' a professional ice hockey player who has played for the Boston Bruins, a team that has won the Stanley Cup in 2011, 2013, and 2019'
Adding requests: 100%|████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2493.64it/s]
Processed prompts: 100%|█████████████████████████| 1/1 [00:00<00:00,  2.85it/s, est. speed input: 11.42 toks/s, output: 114.16 toks/s]
Prompt: 'The Boston Bruins are', Generated text: ' a professional ice hockey team based in Boston, Massachusetts, and they are a member of the National Hockey League (NHL). The team was founded in 1924 and is one of the'
```

Do you meet any errors?
---
###  @ 2025-10-14T12:02:27Z

>Do you meet any errors?

I get 

 NotImplementedError: No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for CUDA device capability: 120. Required capability: 90 or 100
 
 if enforce_eager=True is not set
---
###  @ 2025-10-14T12:06:54Z

By the way, I'm using 

huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4


and nv-community/Qwen3-30B-A3B-FP4 is only on modelscope, not hf, and it's too old.
---
### XiaobingSuper @ 2025-10-14T13:27:54Z

> By the way, I'm using
> 
> huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4
> 
> and nv-community/Qwen3-30B-A3B-FP4 is only on modelscope, not hf, and it's too old.

It is so strange, I also tested huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4, it works well in my side.
---
### voipmonitor @ 2025-10-14T15:05:54Z

the biggest problem is that while it works for me, the speed is only 100 tokens/sec while the FP8 variant is 122 tokens/sec. This looks like the FP4 this way is not using the native NVFP4 on the card but it is probably dequantised to the FP8 or FP16 or whatever. 
---
###  @ 2025-10-14T15:06:52Z

>It is so strange, I also tested huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4, it works well in my side.

Alright, I deleted my vllm folder and completely recompiled it. Then I found that this is working this time! Without --enforce-eager! And without  --enforce-eager it becomes dramatically fast!

So it's just a compile thing. Damn, vllm is so hard to compile correctly... Anyway, thanks!


---
###  @ 2025-10-14T15:08:34Z

> the biggest problem is that while it works for me, the speed is only 100 tokens/sec while the FP8 variant is 122 tokens/sec. This looks like the FP4 this way is not using the native NVFP4 on the card but it is probably dequantised to the FP8 or FP16 or whatever.

I got a speed around 118t/s. But it's just a single generation, how about total throughput?

By the way, can 5090 32g even run FP8? I haven't tested it yet, but I think 32g is a bit low for 30b fp8 model.
---
### BenasdTW @ 2025-10-14T15:12:26Z

> It is so strange, I also tested huggingface.co/NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4, it works well in my side.

It worked for me as well.
I've tested the following models:

1. `nvidia/Qwen3-30B-A3B-FP4` — works!

2. `Benasd/Qwen3-30B-A3B-Instruct-2507-NVFP4` — this model is quantized to NVFP4 A4 using `llm-compressor`. It works with this PR, but uses 25 GB of VRAM, which is much more than `nvidia/Qwen3-30B-A3B-FP4`, though still less than FP8. (Its model file size is the same as `nvidia/Qwen3-30B-A3B-FP4`.)

3. `NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4` — works!

4. `NVFP4/Qwen3-30B-A3B-Thinking-2507-FP4` — this one failed.
---
### voipmonitor @ 2025-10-14T16:43:08Z

> > the biggest problem is that while it works for me, the speed is only 100 tokens/sec while the FP8 variant is 122 tokens/sec. This looks like the FP4 this way is not using the native NVFP4 on the card but it is probably dequantised to the FP8 or FP16 or whatever.
> 
> I got a speed around 118t/s. But it's just a single generation, how about total throughput?
> 
> By the way, can 5090 32g even run FP8? I haven't tested it yet, but I think 32g is a bit low for 30b fp8 model.

the problem is that I got exactly the same speed in FP4 as FP8 which means native NVFP4 is not used. It must be theoretically 2x faster - practically at least faster. I have RTX 6000 PRO so the FP8 full model fits 
---
###  @ 2025-10-15T02:02:39Z

>the problem is that I got exactly the same speed in FP4 as FP8 which means native NVFP4 is not used. It must be theoretically 2x faster - practically at least faster. I have RTX 6000 PRO so the FP8 full model fits

Maybe this is related?
https://github.com/NVIDIA/TensorRT-LLM/issues/4412

they say in TensorRT-LLM,
>Compared to FP8, FP4 shows faster performance during the prefill stage, but FP4 is actually slightly slower during the decode stage.

So maybe we should test the prefill speed as well
---
###  @ 2025-10-15T02:48:46Z

I tested it again and this time I looked at the response content. It responses nonsense sometimes, and sometimes it loops forever. It's prompt and random seed related. For some prompt it's ok, but sometimes it's not.

Model: NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4

By the way, TensorRT-LLM's implementation also has this behavior. Something must be wrong. Maybe it should be another issue.
---
### ReinForce-II @ 2025-10-15T03:09:55Z

> the problem is that I got exactly the same speed in FP4 as FP8 which means native NVFP4 is not used. It must be theoretically 2x faster - practically at least faster. I have RTX 6000 PRO so the FP8 full model fits

I have tested under various conditions. When `concurrency=1`, NVFP4 shows a slight slowdown in decode. However, during prefill and with higher concurrency during decode, a significant speed improvement can be observed.

```
vllm bench throughput --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 1 --dataset-name random --num-prompts 100 --input-len 4096 --output-len 1
Throughput: 8.61 requests/s, 35223.80 total tokens/s, 8.61 output tokens/s
Total num prompt tokens:  409099
Total num output tokens:  100

vllm bench throughput --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 1 --dataset-name random --num-prompts 100 --input-len 1 --output-len 128
Throughput: 1.13 requests/s, 145.88 total tokens/s, 144.75 output tokens/s
Total num prompt tokens:  100
Total num output tokens:  12800

vllm bench throughput --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --max-model-len 32768 --max-num-seqs 32 --dataset-name random --num-prompts 100 --input-len 1 --output-len 128
Throughput: 11.00 requests/s, 1419.26 total tokens/s, 1408.25 output tokens/s
Total num prompt tokens:  100
Total num output tokens:  12800

VLLM_USE_FLASHINFER_MOE_FP4=1 vllm bench throughput --model RedHatAI/Qwen3-30B-A3B-NVFP4 --max-model-len 32768 --max-num-seqs 1 --dataset-name random --num-prompts 100 --input-len 4096 --output-len 1
Throughput: 13.17 requests/s, 53880.81 total tokens/s, 13.17 output tokens/s
Total num prompt tokens:  409099
Total num output tokens:  100

VLLM_USE_FLASHINFER_MOE_FP4=1 vllm bench throughput --model RedHatAI/Qwen3-30B-A3B-NVFP4 --max-model-len 32768 --max-num-seqs 1 --dataset-name random --num-prompts 100 --input-len 1 --output-len 128
Throughput: 0.97 requests/s, 125.04 total tokens/s, 124.07 output tokens/s
Total num prompt tokens:  100
Total num output tokens:  12800

VLLM_USE_FLASHINFER_MOE_FP4=1 vllm bench throughput --model RedHatAI/Qwen3-30B-A3B-NVFP4 --max-model-len 32768 --max-num-seqs 32 --dataset-name random --num-prompts 100 --input-len 1 --output-len 128
Throughput: 14.36 requests/s, 1852.99 total tokens/s, 1838.62 output tokens/s
Total num prompt tokens:  100
Total num output tokens:  12800
```
---