# PR #5908: [Model] Add Gemma 2

- field: attention  | area: model_executor/layers
- author: WoosukKwon
- state: MERGED  created: 2024-06-27T15:03:57Z  merged: 2024-06-27T20:33:57Z
- url: https://github.com/vllm-project/vllm/pull/5908
- labels: new-model

## Body

This PR adds Gemma 2, a new family of open LLMs from Google.

Two major issues to note:
1. Attention logit soft-capping: Gemma 2 models soft-cap the attention logits. This requires changes to all the attention kernels vLLM is using, so **this PR removes soft-capping as a temporary workaround**. While this makes the model different from the original implementation, it does not significantly affect the model's generation capability.
2. Sliding window attention: Gemma 2 uses sliding window attention for every other layer. vLLM currently ignores it and uses global attention for all layers. This might affect the model's behavior when the context length is larger than the sliding window size (4K). Therefore, **this PR temporarily truncates the model's maximum length to 4K**.

These issues will also be explicitly mentioned in warning messages.

## Comments

### WoosukKwon @ 2024-06-27T15:08:08Z

~~Currently blocked by transformers 4.42 release.~~
---
### robertgshaw2-redhat @ 2024-06-27T16:21:30Z

@WoosukKwon running with `--disable-sliding-window` should resolve this SLA issue. We could also set this as the default if gemma2 arch is detected
---
### WoosukKwon @ 2024-06-27T16:23:38Z

@robertgshaw2-neuralmagic Yes I did a similar thing in `config.py` Could you please take a look?
---
### robertgshaw2-redhat @ 2024-06-27T16:28:57Z

> @robertgshaw2-neuralmagic Yes I did a similar thing in `config.py` Could you please take a look?

The only issue with what you have done is that `model_max_len` will get set to`max_position_embedding` rather than `sliding_window_size` 

If you set `disable_sliding_window=True` in https://github.com/vllm-project/vllm/blob/df2c007974b09777259f5bc55ba4b94092f9f475/vllm/config.py#L137, then you all of the logic associated with selecting the max_seq_len will be capped at sliding window size

If we do not want this as the default behavior, we could instead let the user know about this flag in the `print_warning_once`
---
### WoosukKwon @ 2024-06-27T16:36:46Z

@robertgshaw2-neuralmagic Thanks for letting me know! Update the PR. PTAL.
---
### zifeitong @ 2024-06-28T18:27:30Z

> Attention logit soft-capping: Gemma 2 models soft-cap the attention logits. This requires changes to all the attention kernels vLLM is using, so this PR removes soft-capping as a temporary workaround. While this makes the model different from the original implementation, it does not significantly affect the model's generation capability.

For the 27b model, logit soft-capping seems to be very important: https://github.com/huggingface/transformers/pull/31698. 9b model works fine without it.

---
### WoosukKwon @ 2024-06-28T19:04:09Z

@zifeitong Thanks for letting us know! We are planning to use the latest version of FlashInfer to include the soft-capping. Please stay tuned.
---
### Rallio67 @ 2024-06-29T02:10:33Z

> @zifeitong Thanks for letting us know! We are planning to use the latest version of FlashInfer to include the soft-capping. Please stay tuned.

Would also like to report that install from source of current vllm with gemma2 support is working fine for gemma-2-9b and completely failing for gemma-2-27b. Exact same code, same prompts, etc and it all is fitting on single H100. Gemma-2-27B only outputs 0 tokens, while Gemma-2-9B outputs as expected. Working with the base models downloaded from huggingface repos.
---
### Hi-archers @ 2024-07-01T02:33:58Z

> > @zifeitong Thanks for letting us know! We are planning to use the latest version of FlashInfer to include the soft-capping. Please stay tuned.
> 
> Would also like to report that install from source of current vllm with gemma2 support is working fine for gemma-2-9b and completely failing for gemma-2-27b. Exact same code, same prompts, etc and it all is fitting on single H100. Gemma-2-27B only outputs 0 tokens, while Gemma-2-9B outputs as expected. Working with the base models downloaded from huggingface repos.


I obtained results similar to  @Rallio67 . When using vLLM on A800 with bf16 for generation, Gemma2-9B generates normally, but sometimes Gemma2-27B cannot complete the generation properly and continuously produces a large number of meaningless words.
---
### timbmg @ 2024-07-08T09:28:31Z

Thanks for the PR and supporting Gemma 2! Will the 8k context length be supported in the future?
---
### jvlinsta @ 2024-07-08T13:03:25Z

Multiple sources (e.g., https://www.reddit.com/r/LocalLLaMA/comments/1dusu3s/gemma_2_finetuning_2x_faster_63_less_memory_best/) have confirmed that softcapping is an absolute necessity for the 27b checkpoint. Are there any plans of making this available in vllm? Otherwise, the generation of the 27b checkpoint is useless...  
---
### robertgshaw2-redhat @ 2024-07-08T22:42:18Z

> Multiple sources (e.g., https://www.reddit.com/r/LocalLLaMA/comments/1dusu3s/gemma_2_finetuning_2x_faster_63_less_memory_best/) have confirmed that softcapping is an absolute necessity for the 27b checkpoint. Are there any plans of making this available in vllm? Otherwise, the generation of the 27b checkpoint is useless...

Release v0.5.1 from the weekend supports logits soft capping with the FLASHINFER attention backend
---
### lonngxiang @ 2024-07-09T00:59:47Z

> > Multiple sources (e.g., https://www.reddit.com/r/LocalLLaMA/comments/1dusu3s/gemma_2_finetuning_2x_faster_63_less_memory_best/) have confirmed that softcapping is an absolute necessity for the 27b checkpoint. Are there any plans of making this available in vllm? Otherwise, the generation of the 27b checkpoint is useless...
> 
> Release v0.5.1 from the weekend supports logits soft capping with the FLASHINFER attention backend
run error
![image](https://github.com/vllm-project/vllm/assets/40717349/070c0103-938d-422b-8dea-21613606d90e)
![image](https://github.com/vllm-project/vllm/assets/40717349/08a5bb91-d8de-4245-a622-032bcd7e4243)

---
### robertgshaw2-redhat @ 2024-07-09T01:02:30Z

- Please do not post screenshots of error messages, they are difficult to parse and are not searchable
- You have the wrong versions of FlashInfer installed.

Make sure that you match the proper torch and CUDA versions (torch 2.3 and likely cuda 12.1 is what you want)
---
### Hi-archers @ 2024-07-09T01:07:22Z

> > Multiple sources (e.g., https://www.reddit.com/r/LocalLLaMA/comments/1dusu3s/gemma_2_finetuning_2x_faster_63_less_memory_best/) have confirmed that softcapping is an absolute necessity for the 27b checkpoint. Are there any plans of making this available in vllm? Otherwise, the generation of the 27b checkpoint is useless...
> 
> Release v0.5.1 from the weekend supports logits soft capping with the FLASHINFER attention backend

Thank you very much for your contribution, but @Hi-archers in https://github.com/vllm-project/vllm/issues/6166#issuecomment-2214219533 currently has several users experiencing a "Segmentation fault (core dumped)" error after performing extensive Gemma2 inferences using the FlashInfer backend. My current environment is Torch 2.3.0, Cuda 12.1, and FlashInfer 0.08. I hope you can address this issue. Thank you.

---
### robertgshaw2-redhat @ 2024-07-09T01:12:00Z

FlashInfer is built for specific CUDA versions and PyTorch versions. So you can have CUDA 12.1 and Torch2.3.0 but you may have installed FlashInfer built with Torch 2.2.0.

When this happens, we will get the error `BatchDecodeWithPagedKVCacheWrapper`

The default whl for vllm is Python 2.3 and CUDA 12.1. So you likely want to install the following FlashInfer whl:

```bash
PYTHON_VERSION=310
wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.8/flashinfer-0.0.8+cu121torch2.3-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl
```
---
### lonngxiang @ 2024-07-09T01:34:33Z

> FlashInfer is built for specific CUDA versions and PyTorch versions. So you can have CUDA 12.1 and Torch2.3.0 but you may have installed FlashInfer built with Torch 2.2.0.
> 
> When this happens, we will get the error `BatchDecodeWithPagedKVCacheWrapper`
> 
> The default whl for vllm is Python 2.3 and CUDA 12.1. So you likely want to install the following FlashInfer whl:
> 
> ```shell
> PYTHON_VERSION=310
> wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.8/flashinfer-0.0.8+cu121torch2.3-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl
> ```

TKS, Has been running, but the openai interface call answer has been empty.

![image](https://github.com/vllm-project/vllm/assets/40717349/9ec9d309-893d-4506-90d6-d588c02466b1)

---
### robertgshaw2-redhat @ 2024-07-09T01:38:00Z

Please refrain from posting images of your errors. Rather, paste the the test so that I can copy/paste and so that it is searchable

Can you please try the instruction model rather than the pretrained model with the chat interface?
---
### Hi-archers @ 2024-07-09T01:47:26Z

> FlashInfer is built for specific CUDA versions and PyTorch versions. So you can have CUDA 12.1 and Torch2.3.0 but you may have installed FlashInfer built with Torch 2.2.0.
> 
> When this happens, we will get the error `BatchDecodeWithPagedKVCacheWrapper`
> 
> The default whl for vllm is Python 2.3 and CUDA 12.1. So you likely want to install the following FlashInfer whl:
> 
> ```shell
> PYTHON_VERSION=310
> wget https://github.com/flashinfer-ai/flashinfer/releases/download/v0.0.8/flashinfer-0.0.8+cu121torch2.3-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl
> ```


Thank you for your response. However, even after following your instructions to install FLASHINFER, I still encountered a Segmentation fault (core dumped). This time it occurred at 3729/3822 lines, whereas previously it happened at 3708/3822 lines. Should I open a new issue to address this problem?

---
### robertgshaw2-redhat @ 2024-07-09T01:51:06Z

Can you do `pip show vllm` and tell me what you see?
---
### Hi-archers @ 2024-07-09T01:58:05Z

> Can you do `pip show vllm` and tell me what you see?

pip show vllm:

Name: vllm
Version: 0.5.1
Summary: A high-throughput and memory-efficient inference and serving engine for LLMs
Home-page: https://github.com/vllm-project/vllm
Author: vLLM Team
Author-email: 
License: Apache 2.0
Location: /home/weizihao/miniconda3/envs/gemma/lib/python3.10/site-packages
Requires: aiohttp, cmake, fastapi, filelock, lm-format-enforcer, ninja, numpy, nvidia-ml-py, openai, outlines, pillow, prometheus-client, prometheus-fastapi-instrumentator, psutil, py-cpuinfo, pydantic, ray, requests, sentencepiece, tiktoken, tokenizers, torch, torchvision, tqdm, transformers, typing-extensions, uvicorn, vllm-flash-attn, xformers
Required-by: 

My Code:

```Python
import json
import time

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import argparse

from tqdm import tqdm

import os
import sys

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
os.environ["HF_TOKEN"] = "<TOKEN>"

parser = argparse.ArgumentParser()
parser.add_argument('--top', type=str, default="3")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--utili', type=float, default=1)
parser.add_argument('--model_name', type=str, default="/data1/**/Gemma2/gemma-2-9b-it")
parser.add_argument('--title', type=int, default=1)
parser.add_argument('--temperature', type=float, default=0.)

args = parser.parse_args()

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
sys.path.append(os.path.abspath("../../"))

from system_prompt import system_prompts, demonstration, instruction

sampling_params = SamplingParams(
    temperature=args.temperature,
    seed=args.seed,
    max_tokens=100,
    )

print(sampling_params)

model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(
    model=model_name,
    seed=args.seed,
    gpu_memory_utilization=args.utili,
    )

if __name__ == "__main__":


    answer = []
    prompts = []

    for i, line in tqdm(enumerate(que), total=len(que)):
        for i_line in tmp:
            prompts.append(get_template(i_line["que"], i_line['A'], i_line["B"]))

    print(len(prompts)) # 3822

    outputs = llm.generate(prompts, sampling_params)
```
I removed the irrelevant code from my code.
---
### lonngxiang @ 2024-07-09T02:03:11Z

> Please refrain from posting images of your errors. Rather, paste the the test so that I can copy/paste and so that it is searchable
> 
> Can you please try the instruction model rather than the pretrained model with the chat interface?

I downloaded the wrong version. It's working fine now.

---
### jvlinsta @ 2024-07-09T11:56:08Z

I can get it to run with flashinfer as the attention backend, but results are still abysmal. 
Any idea what could still be different from the original implementation?
---
### noamgai21 @ 2024-07-09T13:13:15Z

Hi! I'm using the latest vLLM image on docker, on GCP using A100 GPUs. I'm getting the following error when making many requests to the OpenAI server, using the 27GB instruction tuned model:

```
2024-07-09 13:06:08.540 | NotImplementedError |  
-- | -- | --
  |   | 2024-07-09 13:06:08.540 | raise NotImplementedError |  
  |   | 2024-07-09 13:06:08.540 | self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts) |  
  |   | 2024-07-09 13:06:08.540 | self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy) |  
  |   | 2024-07-09 13:06:08.540 | return func(*args, **kwargs) |  
  |   | 2024-07-09 13:06:08.540 | self.execute_worker(worker_input) |  
  |   | 2024-07-09 13:06:08.540 | raise result |  
  |   | 2024-07-09 13:06:08.540 | async for request_output in stream: |  
  |   | 2024-07-09 13:06:08.540 | raise e |  
  |   | 2024-07-09 13:06:08.540 | async for output in self._process_request( |  
  |   | 2024-07-09 13:06:08.540 | async for res in result_generator: |  
  |   | 2024-07-09 13:06:08.540 | return await self.chat_completion_full_generator( |  
  |   | 2024-07-09 13:06:08.540 | return await dependant.call(**values) |  
  |   | 2024-07-09 13:06:08.540 | await app(scope, receive, sender) |  
  |   | 2024-07-09 13:06:08.540 | raise exc |  
  |   | 2024-07-09 13:06:08.540 | await wrap_app_handling_exceptions(app, request)(scope, receive, send) |  
  |   | 2024-07-09 13:06:08.540 | await self.app(scope, receive, send) |  
  |   | 2024-07-09 13:06:08.539 | await route.handle(scope, receive, send) |  
  |   | 2024-07-09 13:06:08.539 | await self.middleware_stack(scope, receive, send) |  
  |   | 2024-07-09 13:06:08.539 | await app(scope, receive, sender) |  
  |   | 2024-07-09 13:06:08.539 | raise exc
```

This is with vLLM 0.5.1 with
`VLLM_ATTENTION_BACKEND=FLASHINFER` and `--disable-sliding-window` flags. I sshed to the machine and saw that the correct version of FlashInfer is installed (0.0.8, torch 2.3).

What am I doing wrong?
Indeed this seems to be the case:
https://github.com/vllm-project/vllm/blob/08c5bdecae5c5186c39a1d1ff444c3bf01f7fa0e/vllm/attention/backends/flashinfer.py#L56
---
### orellavie1212 @ 2024-07-09T14:43:29Z

> Hi! I'm using the latest vLLM image on docker, on GCP using A100 GPUs. I'm getting the following error when making many requests to the OpenAI server, using the 27GB instruction tuned model:
> 
> ```
> 2024-07-09 13:06:08.540 | NotImplementedError |  
> -- | -- | --
>   |   | 2024-07-09 13:06:08.540 | raise NotImplementedError |  
>   |   | 2024-07-09 13:06:08.540 | self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts) |  
>   |   | 2024-07-09 13:06:08.540 | self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy) |  
>   |   | 2024-07-09 13:06:08.540 | return func(*args, **kwargs) |  
>   |   | 2024-07-09 13:06:08.540 | self.execute_worker(worker_input) |  
>   |   | 2024-07-09 13:06:08.540 | raise result |  
>   |   | 2024-07-09 13:06:08.540 | async for request_output in stream: |  
>   |   | 2024-07-09 13:06:08.540 | raise e |  
>   |   | 2024-07-09 13:06:08.540 | async for output in self._process_request( |  
>   |   | 2024-07-09 13:06:08.540 | async for res in result_generator: |  
>   |   | 2024-07-09 13:06:08.540 | return await self.chat_completion_full_generator( |  
>   |   | 2024-07-09 13:06:08.540 | return await dependant.call(**values) |  
>   |   | 2024-07-09 13:06:08.540 | await app(scope, receive, sender) |  
>   |   | 2024-07-09 13:06:08.540 | raise exc |  
>   |   | 2024-07-09 13:06:08.540 | await wrap_app_handling_exceptions(app, request)(scope, receive, send) |  
>   |   | 2024-07-09 13:06:08.540 | await self.app(scope, receive, send) |  
>   |   | 2024-07-09 13:06:08.539 | await route.handle(scope, receive, send) |  
>   |   | 2024-07-09 13:06:08.539 | await self.middleware_stack(scope, receive, send) |  
>   |   | 2024-07-09 13:06:08.539 | await app(scope, receive, sender) |  
>   |   | 2024-07-09 13:06:08.539 | raise exc
> ```
> 
> This is with vLLM 0.5.1 with `VLLM_ATTENTION_BACKEND=FLASHINFER` and `--disable-sliding-window` flags. I sshed to the machine and saw that the correct version of FlashInfer is installed (0.0.8, torch 2.3).
> 
> What am I doing wrong? Indeed this seems to be the case:
> 
> https://github.com/vllm-project/vllm/blob/08c5bdecae5c5186c39a1d1ff444c3bf01f7fa0e/vllm/attention/backends/flashinfer.py#L56

can you please share your code sample? how you load the gemma2 27b with its params? 
---
### noamgai21 @ 2024-07-09T14:55:52Z

> > Hi! I'm using the latest vLLM image on docker, on GCP using A100 GPUs. I'm getting the following error when making many requests to the OpenAI server, using the 27GB instruction tuned model:
> > ```
> > 2024-07-09 13:06:08.540 | NotImplementedError |  
> > -- | -- | --
> >   |   | 2024-07-09 13:06:08.540 | raise NotImplementedError |  
> >   |   | 2024-07-09 13:06:08.540 | self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts) |  
> >   |   | 2024-07-09 13:06:08.540 | self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy) |  
> >   |   | 2024-07-09 13:06:08.540 | return func(*args, **kwargs) |  
> >   |   | 2024-07-09 13:06:08.540 | self.execute_worker(worker_input) |  
> >   |   | 2024-07-09 13:06:08.540 | raise result |  
> >   |   | 2024-07-09 13:06:08.540 | async for request_output in stream: |  
> >   |   | 2024-07-09 13:06:08.540 | raise e |  
> >   |   | 2024-07-09 13:06:08.540 | async for output in self._process_request( |  
> >   |   | 2024-07-09 13:06:08.540 | async for res in result_generator: |  
> >   |   | 2024-07-09 13:06:08.540 | return await self.chat_completion_full_generator( |  
> >   |   | 2024-07-09 13:06:08.540 | return await dependant.call(**values) |  
> >   |   | 2024-07-09 13:06:08.540 | await app(scope, receive, sender) |  
> >   |   | 2024-07-09 13:06:08.540 | raise exc |  
> >   |   | 2024-07-09 13:06:08.540 | await wrap_app_handling_exceptions(app, request)(scope, receive, send) |  
> >   |   | 2024-07-09 13:06:08.540 | await self.app(scope, receive, send) |  
> >   |   | 2024-07-09 13:06:08.539 | await route.handle(scope, receive, send) |  
> >   |   | 2024-07-09 13:06:08.539 | await self.middleware_stack(scope, receive, send) |  
> >   |   | 2024-07-09 13:06:08.539 | await app(scope, receive, sender) |  
> >   |   | 2024-07-09 13:06:08.539 | raise exc
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
> > This is with vLLM 0.5.1 with `VLLM_ATTENTION_BACKEND=FLASHINFER` and `--disable-sliding-window` flags. I sshed to the machine and saw that the correct version of FlashInfer is installed (0.0.8, torch 2.3).
> > What am I doing wrong? Indeed this seems to be the case:
> > https://github.com/vllm-project/vllm/blob/08c5bdecae5c5186c39a1d1ff444c3bf01f7fa0e/vllm/attention/backends/flashinfer.py#L56
> 
> can you please share your code sample? how you load the gemma2 27b with its params?

We use vLLM through k8s, this is the relevant snippet from the yaml:

```
containers:
        - name: main
          command:
          # https://github.com/huggingface/text-generation-inference/issues/1330
          # Without this weird ldconfig trick, we will get a "libcuda.so not found" error.
            - bash
            - -c
            - |
              ldconfig
              echo "Setting VLLM_ATTENTION_BACKEND to FLASHINFER to use the FlashInfer backend. Required for gemma2 logit capping"
              export VLLM_ATTENTION_BACKEND=FLASHINFER
              echo "Starting vLLM server with --disable-sliding-window as version 0.5.1 requires it with FlashInfer backend."
              python3 -m vllm.entrypoints.openai.api_server --model google/gemma-2-27b-it --disable-sliding-window
          image: vllm/vllm-openai:latest
```
---
### bks5881 @ 2024-07-12T13:10:42Z

When I am running gemm2-9b-it on h100, 80GB. The speed is very slow for me, like 20 TPS, any idea why? I launched with Lora adapter. its like much faster on sglang, with 43TPS.
---
### yukavio @ 2024-07-15T10:52:34Z

Is there currently a plan to support sliding windows?
---
### renjie0 @ 2024-10-16T18:43:34Z

Has it been changed since then? Is SLA supported without cap on the context length?
---