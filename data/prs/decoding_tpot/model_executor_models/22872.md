# PR #22872: [Multimodal][Speculative Decoding]Eagle Eagle3 mm support, enablement on qwen2.5vl

- field: decoding  | area: model_executor/models
- author: david6666666
- state: MERGED  created: 2025-08-14T03:28:49Z  merged: 2025-09-27T03:35:48Z
- url: https://github.com/vllm-project/vllm/pull/22872
- labels: performance, new-model, speculative-decoding, ready, v1, llama, qwen

## Body

## Purpose
follow #20788 Eagle Eagle3 mm support, enablement on qwen2.5vl
model is https://huggingface.co/Rayzl/qwen2.5-vl-7b-eagle3-sgl
## Test Plan

## Test Result
```
vllm serve \
    /workspace/models/Qwen2.5-VL-7B-Instruct \
    --port 5580 --host 0.0.0.0 \
    --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192  \
    --no-enable-prefix-caching --trust-remote-code -tp 1\
    --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' \
    --num-lookahead-slots=3 \
    --gpu-memory-utilization=0.93

client

vllm bench serve \
--backend openai-chat \
--endpoint-type openai-chat \
--port 5580 \
--host 127.0.0.1 \
--endpoint /v1/chat/completions \
--model /home/models/Qwen2.5-VL-7B-Instruct \
--dataset-name hf \
--dataset-path Lin-Chen/MMStar \
--hf-split val \
--num-prompts 50 \
--max-concurrency 2 \
--temperature 0 \
--top-p 1.0 \
--seed 123

e2e speed up 1.3x
# eagle3
============ Serving Benchmark Result ============
Successful requests:                     50        
Maximum request concurrency:             2         
Benchmark duration (s):                  32.11     
Total input tokens:                      1226      
Total generated tokens:                  3985      
Request throughput (req/s):              1.56      
Output token throughput (tok/s):         124.09    
Peak output token throughput (tok/s):    82.00     
Peak concurrent requests:                6.00      
Total Token throughput (tok/s):          162.26    
---------------Time to First Token----------------
Mean TTFT (ms):                          50.02     
Median TTFT (ms):                        49.18     
P99 TTFT (ms):                           71.19     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.07     
Median TPOT (ms):                        14.27     
P99 TPOT (ms):                           19.15     
---------------Inter-token Latency----------------
Mean ITL (ms):                           24.34     
Median ITL (ms):                         24.75     
P99 ITL (ms):                            25.99     
==================================================


# without eagle3
tip: install termplotlib and gnuplot to plot the metrics
============ Serving Benchmark Result ============
Successful requests:                     50        
Maximum request concurrency:             2         
Benchmark duration (s):                  44.76     
Total input tokens:                      1226      
Total generated tokens:                  4237      
Request throughput (req/s):              1.12      
Output token throughput (tok/s):         94.66     
Peak output token throughput (tok/s):    99.00     
Peak concurrent requests:                5.00      
Total Token throughput (tok/s):          122.05    
---------------Time to First Token----------------
Mean TTFT (ms):                          43.05     
Median TTFT (ms):                        41.76     
P99 TTFT (ms):                           62.82     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.48     
Median TPOT (ms):                        20.49     
P99 TPOT (ms):                           20.64     
---------------Inter-token Latency----------------
Mean ITL (ms):                           20.24     
Median ITL (ms):                         20.46     
P99 ITL (ms):                            21.60     
==================================================
```




## (Optional) Documentation Update

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### github-actions @ 2025-08-14T03:54:38Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### 22quinn @ 2025-08-16T06:43:27Z

> cc @22quinn @morgendave can you help review?

cc spec decode experts @zixi-qi @charlotte12l 
---
### mergify @ 2025-08-20T20:14:32Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @david6666666.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### LJH-LBJ @ 2025-09-05T09:15:52Z

The draft acceptance rate is still low. I have already tried using both Llama and Qwen as different decode layers for the Eagle3 implementation, and I have also compared the code of Qwen2.5-VL Eagle3 in sglang.(https://github.com/sgl-project/sglang/pull/8801) I haven’t found anything missing so far, and I feel like I need some help.@zixi-qi @charlotte12l @DarkLight1337 


---
### mergify @ 2025-09-10T04:34:44Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @david6666666.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### ggg-s @ 2025-09-18T01:54:53Z

> The draft acceptance rate is still low. I have already tried using both Llama and Qwen as different decode layers for the Eagle3 implementation, and I have also compared the code of Qwen2.5-VL Eagle3 in sglang.([sgl-project/sglang#8801](https://github.com/sgl-project/sglang/pull/8801)) I haven’t found anything missing so far, and I feel like I need some help.@zixi-qi @charlotte12l @DarkLight1337

Because the test dataset is different？
---
### ggg-s @ 2025-09-19T07:14:46Z

Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ 
---
### LJH-LBJ @ 2025-09-21T14:54:33Z

> Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ

In https://github.com/sgl-project/SpecForge/pull/102, Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s 
---
### LJH-LBJ @ 2025-09-21T15:28:20Z

> However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model.

Do you mean the code below?
```
inputs_embeds = self.model.get_input_embeddings(
                input_ids,
                multimodal_embeddings=mm_embeds or None,
            )
```
If I do not merge the image embedding to text embedding , how can the model get image information? @ggg-s 
---
### ggg-s @ 2025-09-22T01:49:28Z

> > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> 
> In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s

in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
---
### LJH-LBJ @ 2025-09-22T03:51:09Z

> > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > 
> > 
> > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> 
> in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.

Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens
`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)`
Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s 
---
### ggg-s @ 2025-09-22T07:01:15Z

> > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > 
> > > 
> > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > 
> > 
> > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> 
> Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens `def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)` Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s

it‘s here
<img width="861" height="773" alt="image" src="https://github.com/user-attachments/assets/b8513b55-ac26-4ee0-9aed-c4eb87bf4b40" />

---
### LJH-LBJ @ 2025-09-22T09:55:05Z

> > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > 
> > > > 
> > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > 
> > > 
> > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > 
> > 
> > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens `def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)` Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s
> 
> it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">

Thanks, I ignore this. but I try like this
`
if self.is_multimodal_model:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = self.model.get_input_embeddings(
                input_ids,
                None,
            )
`
I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
**dataset：** lmarena-ai/VisionArena-Chat
**result：**
(APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
**dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
**result：**
(APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
(APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
(APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
(APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
@ggg-s 


---
### ggg-s @ 2025-09-22T13:53:47Z

> > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> 
> > > > > 
> 
> > > > > 
> 
> > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> 
> > > > 
> 
> > > > 
> 
> > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> 
> > > 
> 
> > > 
> 
> > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens `def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)` Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s
> 
> > 
> 
> > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> 
> 
> 
> Thanks, I ignore this. but I try like this
> 
> `
> 
> if self.is_multimodal_model:
> 
>             input_ids = self.input_ids[:num_tokens]
> 
>             inputs_embeds = self.model.get_input_embeddings(
> 
>                 input_ids,
> 
>                 None,
> 
>             )
> 
> `
> 
> I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> 
> **dataset：** lmarena-ai/VisionArena-Chat
> 
> **result：**
> 
> (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> 
> (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> 
> **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> 
> **result：**
> 
> (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> 
> (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> 
> (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> 
> (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> 
> @ggg-s 
> 
> 
> 
> 

Hiii  I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?

---
### LJH-LBJ @ 2025-09-22T14:29:04Z

> > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > 
> > 
> > > > > >
> > 
> > 
> > > > > >
> > 
> > 
> > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > 
> > 
> > > > >
> > 
> > 
> > > > >
> > 
> > 
> > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > 
> > 
> > > >
> > 
> > 
> > > >
> > 
> > 
> > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens `def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)` Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s
> > 
> > 
> > > 
> > 
> > 
> > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > 
> > 
> > Thanks, I ignore this. but I try like this
> > `
> > if self.is_multimodal_model:
> > ```
> >         input_ids = self.input_ids[:num_tokens]
> > 
> >         inputs_embeds = self.model.get_input_embeddings(
> > 
> >             input_ids,
> > 
> >             None,
> > 
> >         )
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
> > `
> > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > **dataset：** lmarena-ai/VisionArena-Chat
> > **result：**
> > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > **result：**
> > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > @ggg-s
> 
> Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?

Do you test with my code?
But I tested  on the mmstar dataset, it still  gets low accuracy.
This is my server cmd
`
vllm serve     /workspace/models/Qwen2.5-VL-7B-Instruct     --port 5580 --host 0.0.0.0     --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192      --no-enable-prefix-caching --trust-remote-code -tp 1    --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}'     --num-lookahead-slots=3   --gpu-memory-utilization=0.93
`
09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
(APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086
@ggg-s 
---
### ggg-s @ 2025-09-23T03:24:51Z

> > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > 
> > > 
> > > > > > >
> > > 
> > > 
> > > > > > >
> > > 
> > > 
> > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > 
> > > 
> > > > > >
> > > 
> > > 
> > > > > >
> > > 
> > > 
> > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > 
> > > 
> > > > >
> > > 
> > > 
> > > > >
> > > 
> > > 
> > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens `def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)` Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s
> > > 
> > > 
> > > > 
> > > 
> > > 
> > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > 
> > > 
> > > Thanks, I ignore this. but I try like this
> > > `
> > > if self.is_multimodal_model:
> > > ```
> > >         input_ids = self.input_ids[:num_tokens]
> > > 
> > >         inputs_embeds = self.model.get_input_embeddings(
> > > 
> > >             input_ids,
> > > 
> > >             None,
> > > 
> > >         )
> > > ```
> > > 
> > > 
> > >     
> > >       
> > >     
> > > 
> > >       
> > >     
> > > 
> > >     
> > >   
> > > `
> > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > **dataset：** lmarena-ai/VisionArena-Chat
> > > **result：**
> > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > **result：**
> > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > @ggg-s
> > 
> > 
> > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> 
> Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd `vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93` 09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s

This is my server cmd

 CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct \
  --port 18000   --max-model-len 8080 --tensor-parallel-size 1  --speculative-config '{
    "method": "eagle3",
    "model": "/models/qwen2.5-vl-7b-eagle3-sgl",
    "num_speculative_tokens": 3
  }' 2>&1 | tee vllm_output.log
  
  
(APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0%
(APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241

When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.

---
### LJH-LBJ @ 2025-09-23T03:43:23Z

Can I attach to you with slack？

> > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > 
> > > > 
> > > > > > > >
> > > > 
> > > > 
> > > > > > > >
> > > > 
> > > > 
> > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > 
> > > > 
> > > > > > >
> > > > 
> > > > 
> > > > > > >
> > > > 
> > > > 
> > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > 
> > > > 
> > > > > >
> > > > 
> > > > 
> > > > > >
> > > > 
> > > > 
> > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > 
> > > > 
> > > > > 
> > > > 
> > > > 
> > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > 
> > > > 
> > > > Thanks, I ignore this. but I try like this
> > > > `
> > > > if self.is_multimodal_model:
> > > > ```
> > > >         input_ids = self.input_ids[:num_tokens]
> > > > 
> > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > 
> > > >             input_ids,
> > > > 
> > > >             None,
> > > > 
> > > >         )
> > > > ```
> > > > 
> > > > 
> > > >     
> > > >       
> > > >     
> > > > 
> > > >       
> > > >     
> > > > 
> > > >     
> > > >   
> > > > `
> > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > **result：**
> > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > **result：**
> > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > @ggg-s
> > > 
> > > 
> > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > 
> > 
> > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> 
> This is my server cmd
> 
> CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> 
> (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> 
> When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.

Do you test on my code? @ggg-s 
---
### ggg-s @ 2025-09-23T04:04:11Z

> Can I attach to you with slack？
> 
> > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > 
> > > > > 
> > > > > > > > >
> > > > > 
> > > > > 
> > > > > > > > >
> > > > > 
> > > > > 
> > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > 
> > > > > 
> > > > > > > >
> > > > > 
> > > > > 
> > > > > > > >
> > > > > 
> > > > > 
> > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > 
> > > > > 
> > > > > > >
> > > > > 
> > > > > 
> > > > > > >
> > > > > 
> > > > > 
> > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > 
> > > > > 
> > > > > > 
> > > > > 
> > > > > 
> > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > 
> > > > > 
> > > > > Thanks, I ignore this. but I try like this
> > > > > `
> > > > > if self.is_multimodal_model:
> > > > > ```
> > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > 
> > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > 
> > > > >             input_ids,
> > > > > 
> > > > >             None,
> > > > > 
> > > > >         )
> > > > > ```
> > > > > 
> > > > > 
> > > > >     
> > > > >       
> > > > >     
> > > > > 
> > > > >       
> > > > >     
> > > > > 
> > > > >     
> > > > >   
> > > > > `
> > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > **result：**
> > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > **result：**
> > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > @ggg-s
> > > > 
> > > > 
> > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > 
> > > 
> > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > 
> > 
> > This is my server cmd
> > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> 
> Do you test on my code? @ggg-s

yep the same result
your server cmd：
 CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct \
  --port 18000  --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1  --speculative-config '{
    "method": "eagle3",
    "model": "/models/qwen2.5-vl-7b-eagle3-sgl",
    "num_speculative_tokens": 3,
    "prefill_token_shift": false,
    "num_speculative_tokens":3,
    "draft_tensor_parallel_size": 1, "max_model_len": 8192
  }' 
  --num-lookahead-slots 3 --gpu-memory-utilization 0.93 \
  2>&1 | tee vllm_output.log
  
  logger：
  
(APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0%
(APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206



  
---
### LJH-LBJ @ 2025-09-23T04:25:51Z

> > Can I attach to you with slack？
> > > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > > 
> > > > > > 
> > > > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > > 
> > > > > > 
> > > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > > 
> > > > > > 
> > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > >
> > > > > > 
> > > > > > 
> > > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > > 
> > > > > > 
> > > > > > > 
> > > > > > 
> > > > > > 
> > > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > > 
> > > > > > 
> > > > > > Thanks, I ignore this. but I try like this
> > > > > > `
> > > > > > if self.is_multimodal_model:
> > > > > > ```
> > > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > > 
> > > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > > 
> > > > > >             input_ids,
> > > > > > 
> > > > > >             None,
> > > > > > 
> > > > > >         )
> > > > > > ```
> > > > > > 
> > > > > > 
> > > > > >     
> > > > > >       
> > > > > >     
> > > > > > 
> > > > > >       
> > > > > >     
> > > > > > 
> > > > > >     
> > > > > >   
> > > > > > `
> > > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > > **result：**
> > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > > **result：**
> > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > > @ggg-s
> > > > > 
> > > > > 
> > > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > > 
> > > > 
> > > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > > 
> > > 
> > > This is my server cmd
> > > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> > 
> > 
> > Do you test on my code? @ggg-s
> 
> yep the same result your server cmd： CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3, "prefill_token_shift": false, "num_speculative_tokens":3, "draft_tensor_parallel_size": 1, "max_model_len": 8192 }' --num-lookahead-slots 3 --gpu-memory-utilization 0.93 2>&1 | tee vllm_output.log
> 
> logger：
> 
> (APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0% (APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206

Sorry, I mean the vllm-code, not the server cmd. Do you use the code add in this PR? @ggg-s 
---
### ggg-s @ 2025-09-23T04:37:11Z

> > > Can I attach to you with slack？
> > > > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > > > 
> > > > > > > 
> > > > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > > > 
> > > > > > > 
> > > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > > > 
> > > > > > > 
> > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > >
> > > > > > > 
> > > > > > > 
> > > > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > > > 
> > > > > > > 
> > > > > > > > 
> > > > > > > 
> > > > > > > 
> > > > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > > > 
> > > > > > > 
> > > > > > > Thanks, I ignore this. but I try like this
> > > > > > > `
> > > > > > > if self.is_multimodal_model:
> > > > > > > ```
> > > > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > > > 
> > > > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > > > 
> > > > > > >             input_ids,
> > > > > > > 
> > > > > > >             None,
> > > > > > > 
> > > > > > >         )
> > > > > > > ```
> > > > > > > 
> > > > > > > 
> > > > > > >     
> > > > > > >       
> > > > > > >     
> > > > > > > 
> > > > > > >       
> > > > > > >     
> > > > > > > 
> > > > > > >     
> > > > > > >   
> > > > > > > `
> > > > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > > > **result：**
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > > > **result：**
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > > > @ggg-s
> > > > > > 
> > > > > > 
> > > > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > > > 
> > > > > 
> > > > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > > > 
> > > > 
> > > > This is my server cmd
> > > > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > > > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > > > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> > > 
> > > 
> > > Do you test on my code? @ggg-s
> > 
> > 
> > yep the same result your server cmd： CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3, "prefill_token_shift": false, "num_speculative_tokens":3, "draft_tensor_parallel_size": 1, "max_model_len": 8192 }' --num-lookahead-slots 3 --gpu-memory-utilization 0.93 2>&1 | tee vllm_output.log
> > logger：
> > (APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0% (APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206
> 
> Sorry, I mean the vllm-code, not the server cmd. Do you use the code add in this PR? @ggg-s

/vllm# git rev-parse --abbrev-ref HEAD
pr-22872

---
### LJH-LBJ @ 2025-09-23T04:51:32Z

> > > > Can I attach to you with slack？
> > > > > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > >
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > 
> > > > > > > > 
> > > > > > > > 
> > > > > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > > > > 
> > > > > > > > 
> > > > > > > > Thanks, I ignore this. but I try like this
> > > > > > > > `
> > > > > > > > if self.is_multimodal_model:
> > > > > > > > ```
> > > > > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > > > > 
> > > > > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > > > > 
> > > > > > > >             input_ids,
> > > > > > > > 
> > > > > > > >             None,
> > > > > > > > 
> > > > > > > >         )
> > > > > > > > ```
> > > > > > > > 
> > > > > > > > 
> > > > > > > >     
> > > > > > > >       
> > > > > > > >     
> > > > > > > > 
> > > > > > > >       
> > > > > > > >     
> > > > > > > > 
> > > > > > > >     
> > > > > > > >   
> > > > > > > > `
> > > > > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > > > > **result：**
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > > > > **result：**
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > > > > @ggg-s
> > > > > > > 
> > > > > > > 
> > > > > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > > > > 
> > > > > > 
> > > > > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > > > > 
> > > > > 
> > > > > This is my server cmd
> > > > > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > > > > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > > > > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> > > > 
> > > > 
> > > > Do you test on my code? @ggg-s
> > > 
> > > 
> > > yep the same result your server cmd： CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3, "prefill_token_shift": false, "num_speculative_tokens":3, "draft_tensor_parallel_size": 1, "max_model_len": 8192 }' --num-lookahead-slots 3 --gpu-memory-utilization 0.93 2>&1 | tee vllm_output.log
> > > logger：
> > > (APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0% (APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206
> > 
> > Sorry, I mean the vllm-code, not the server cmd. Do you use the code add in this PR? @ggg-s
> 
> /vllm# git rev-parse --abbrev-ref HEAD
> pr-22872
> 

Great , with mm-star , I also get 1.3x speedup. The biggest promble is dataset I think. @ggg-s 

---
### ggg-s @ 2025-09-23T09:37:52Z

> > > > > Can I attach to you with slack？
> > > > > > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > >
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > Thanks, I ignore this. but I try like this
> > > > > > > > > `
> > > > > > > > > if self.is_multimodal_model:
> > > > > > > > > ```
> > > > > > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > > > > > 
> > > > > > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > > > > > 
> > > > > > > > >             input_ids,
> > > > > > > > > 
> > > > > > > > >             None,
> > > > > > > > > 
> > > > > > > > >         )
> > > > > > > > > ```
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > >     
> > > > > > > > >       
> > > > > > > > >     
> > > > > > > > > 
> > > > > > > > >       
> > > > > > > > >     
> > > > > > > > > 
> > > > > > > > >     
> > > > > > > > >   
> > > > > > > > > `
> > > > > > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > > > > > **result：**
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > > > > > **result：**
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > > > > > @ggg-s
> > > > > > > > 
> > > > > > > > 
> > > > > > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > > > > > 
> > > > > > > 
> > > > > > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > > > > > 
> > > > > > 
> > > > > > This is my server cmd
> > > > > > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > > > > > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > > > > > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> > > > > 
> > > > > 
> > > > > Do you test on my code? @ggg-s
> > > > 
> > > > 
> > > > yep the same result your server cmd： CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3, "prefill_token_shift": false, "num_speculative_tokens":3, "draft_tensor_parallel_size": 1, "max_model_len": 8192 }' --num-lookahead-slots 3 --gpu-memory-utilization 0.93 2>&1 | tee vllm_output.log
> > > > logger：
> > > > (APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0% (APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206
> > > 
> > > 
> > > Sorry, I mean the vllm-code, not the server cmd. Do you use the code add in this PR? @ggg-s
> > 
> > 
> > /vllm# git rev-parse --abbrev-ref HEAD
> > pr-22872
> 
> Great , with mm-star , I also get 1.3x speedup. The biggest promble is dataset I think. @ggg-s

Got it, so it does look like a dataset issue. By the way, didn’t you test with the MMStar dataset before? If it’s easy, we can talk on Slack.

---
### LJH-LBJ @ 2025-09-23T10:43:07Z

> > > > > > Can I attach to you with slack？
> > > > > > > > > > > > > > > Hi, thank you for your excellent work! I have a couple of questions about the implementation. Firstly, the qwen2.5-vl-7b-sgl model was trained in SpecForge without including image tokens. However, in your branch, you concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. Could this approach impact the model's performance? Additionally, I noticed that the inference for this model on SGLang does not incorporate image tokens. Secondly, regarding the @support_torch_compile decorator with the configuration dynamic_arg_dims={"input_ids": 0, "positions": -1, "inputs_embeds": 0}, it triggers a runtime assertion error due to inconsistent sequence lengths, which prevents the service from starting. @david6666666 @LJH-LBJ
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > > > In [sgl-project/SpecForge#102](https://github.com/sgl-project/SpecForge/pull/102), Rayzl/qwen2.5-vl-7b-eagle3-sgl model uses https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V for training, I think there are many request includes images . Why do you think " without including image tokens" @ggg-s
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > > in SpecForge’s eagle3.py（Line 699）, the draft branch doesn’t take raw image tokens or run a vision encoder. Instead, it consumes the hidden states from the language model. For Qwen2.5-VL, those hidden states already encode the visual context after the model has fused image embeddings into its sequence. So when I said “without including image tokens,” what I meant is that the draft never processes vision tokens directly — it only sees the fused LM hidden states. The same is true in sglang’s speculative decoding for qwen2.5-vl-7b-eagle3: the draft/verify pipeline operates purely on LM states, not on image tokens. That’s why the draft can leverage visual context implicitly, but it’s not explicitly vision-aware.
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > >
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > > Yes, I understand. In [vllm's eagle.py]After combining three-layer hidden states in combine_hidden_states(L147), it gets inputs_embeds by get_input_embeddings(L180). I think get_input_embeddings in llama.py doesnot use image tokens  Therefore， I wonder where I concatenate the image tokens with the three-layer hidden states from Qwen2 before feeding them into the draft model. @ggg-s`def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor: return self.model.get_input_embeddings(input_ids)`
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > > it‘s here <img alt="image" width="861" height="773" src="https://private-user-images.githubusercontent.com/71912051/492170817-b8513b55-ac26-4ee0-9aed-c4eb87bf4b40.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTg1MzIyMTIsIm5iZiI6MTc1ODUzMTkxMiwicGF0aCI6Ii83MTkxMjA1MS80OTIxNzA4MTctYjg1MTNiNTUtYWMyNi00ZWUwLTlhZWQtYzRlYjg3YmY0YjQwLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA5MjIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwOTIyVDA5MDUxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZlMjFiNTc4ODUzMjVhMjQxNWEwZjRiYzRhN2ZlYmQyYmEzMWMyYjMzMTI1YjA3NGYzOWFmMzMwYjM3YjA0ZjQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.OZMffOCjyXp53g03yCxKk-y4er8zpw0j2DCsa2j3P40">
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > > Thanks, I ignore this. but I try like this
> > > > > > > > > > `
> > > > > > > > > > if self.is_multimodal_model:
> > > > > > > > > > ```
> > > > > > > > > >         input_ids = self.input_ids[:num_tokens]
> > > > > > > > > > 
> > > > > > > > > >         inputs_embeds = self.model.get_input_embeddings(
> > > > > > > > > > 
> > > > > > > > > >             input_ids,
> > > > > > > > > > 
> > > > > > > > > >             None,
> > > > > > > > > > 
> > > > > > > > > >         )
> > > > > > > > > > ```
> > > > > > > > > > 
> > > > > > > > > > 
> > > > > > > > > >     
> > > > > > > > > >       
> > > > > > > > > >     
> > > > > > > > > > 
> > > > > > > > > >       
> > > > > > > > > >     
> > > > > > > > > > 
> > > > > > > > > >     
> > > > > > > > > >   
> > > > > > > > > > `
> > > > > > > > > > I get the same result. I also try text-only benchmark dataset, but it still has low accurancy
> > > > > > > > > > **dataset：** lmarena-ai/VisionArena-Chat
> > > > > > > > > > **result：**
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [loggers.py:123] Engine 000: Avg prompt throughput: 150.8 tokens/s, Avg generation throughput: 283.1 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0%
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:29:05 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **_9.9%_**, Mean acceptance length: 1.30, Accepted: 654 tokens, Drafted: 6582 tokens, Per-position acceptance rate: 0.214, 0.064, 0.020
> > > > > > > > > > **dataset：** ALLaVA/allava_text/Evol-Instruct-GPT4-Turbo-143K.json
> > > > > > > > > > **result：**
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [loggers.py:123] Engine 000: Avg prompt throughput: 344.3 tokens/s, Avg generation throughput: 900.8 tokens/s, Running: 128 reqs, Waiting: 726 reqs, GPU KV cache usage: 90.7%, Prefix cache hit rate: 0.0%
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 7.9%, Mean acceptance length: 1.24, Accepted: 1720 tokens, Drafted: 21804 tokens, Per-position acceptance rate: 0.202, 0.031, 0.004
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [loggers.py:123] Engine 000: Avg prompt throughput: 202.5 tokens/s, Avg generation throughput: 930.9 tokens/s, Running: 128 reqs, Waiting: 706 reqs, GPU KV cache usage: 94.7%, Prefix cache hit rate: 0.0%
> > > > > > > > > > (APIServer pid=1748893) INFO 09-22 13:35:35 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: **7.8%**, Mean acceptance length: 1.24, Accepted: 1773 tokens, Drafted: 22596 tokens, Per-position acceptance rate: 0.198, 0.032, 0.005
> > > > > > > > > > @ggg-s
> > > > > > > > > 
> > > > > > > > > 
> > > > > > > > > Hiii I tested on the VisionArena-Chat dataset and got results similar to yours—I couldn’t reproduce their reported results. When I tested on the mmstar dataset, I got the same results as sglang "qwen2.5 vl infer" PR. Could it be due to the dataset?
> > > > > > > > 
> > > > > > > > 
> > > > > > > > Do you test with my code? But I tested on the mmstar dataset, it still gets low accuracy. This is my server cmd  09-23 02:21:39 [loggers.py:123] Engine 000: Avg prompt throughput: 1029.8 tokens/s, Avg generation throughput: 16.2 tokens/s, Running: 0 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 0.0% (APIServer pid=1766187) INFO 09-23 02:21:39 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 21.4%, Mean acceptance length: 1.64, Accepted: 52 tokens, Drafted: 243 tokens, Per-position acceptance rate: 0.395, 0.160, 0.086 @ggg-s`vllm serve /workspace/models/Qwen2.5-VL-7B-Instruct --port 5580 --host 0.0.0.0 --max-num-seqs 128 --dtype bfloat16 --max-model-len=8192 --no-enable-prefix-caching --trust-remote-code -tp 1 --speculative-config '{"method": "eagle3", "model": "/workspace/models/qwen2.5-vl-7b-eagle3-sgl", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1, "max_model_len": 8192}' --num-lookahead-slots=3 --gpu-memory-utilization=0.93`
> > > > > > > 
> > > > > > > 
> > > > > > > This is my server cmd
> > > > > > > CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --max-model-len 8080 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3 }' 2>&1 | tee vllm_output.log
> > > > > > > (APIServer pid=90549) INFO 09-23 03:15:10 [loggers.py:123] Engine 000: Avg prompt throughput: 416.7 tokens/s, Avg generation throughput: 97.5 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 1.9%, Prefix cache hit rate: 0.0% (APIServer pid=90549) INFO 09-23 03:15:10 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 44.1%, Mean acceptance length: 2.32, Accepted: 554 tokens, Drafted: 1257 tokens, Per-position acceptance rate: 0.654, 0.427, 0.241
> > > > > > > When I tested with 2-way concurrency and 32 requests, the average acceptance length was about 2.15.
> > > > > > 
> > > > > > 
> > > > > > Do you test on my code? @ggg-s
> > > > > 
> > > > > 
> > > > > yep the same result your server cmd： CUDA_VISIBLE_DEVICES=5 vllm serve /models/Qwen2.5-VL-7B-Instruct --port 18000 --trust-remote-code --max-model-len 8192 --no-enable-prefix-caching --max-num-seqs 128 --dtype bfloat16 --tensor-parallel-size 1 --speculative-config '{ "method": "eagle3", "model": "/models/qwen2.5-vl-7b-eagle3-sgl", "num_speculative_tokens": 3, "prefill_token_shift": false, "num_speculative_tokens":3, "draft_tensor_parallel_size": 1, "max_model_len": 8192 }' --num-lookahead-slots 3 --gpu-memory-utilization 0.93 2>&1 | tee vllm_output.log
> > > > > logger：
> > > > > (APIServer pid=94751) INFO 09-23 04:02:25 [loggers.py:123] Engine 000: Avg prompt throughput: 581.3 tokens/s, Avg generation throughput: 149.9 tokens/s, Running: 2 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.0%, Prefix cache hit rate: 0.0% (APIServer pid=94751) INFO 09-23 04:02:25 [metrics.py:87] SpecDecoding metrics: Draft acceptance rate: 40.7%, Mean acceptance length: 2.22, Accepted: 825 tokens, Drafted: 2025 tokens, Per-position acceptance rate: 0.636, 0.381, 0.206
> > > > 
> > > > 
> > > > Sorry, I mean the vllm-code, not the server cmd. Do you use the code add in this PR? @ggg-s
> > > 
> > > 
> > > /vllm# git rev-parse --abbrev-ref HEAD
> > > pr-22872
> > 
> > 
> > Great , with mm-star , I also get 1.3x speedup. The biggest promble is dataset I think. @ggg-s
> 
> Got it, so it does look like a dataset issue. By the way, didn’t you test with the MMStar dataset before? If it’s easy, we can talk on Slack.

I didn’t test with the MMStar dataset before. Sure, you can find me by searching for Junhong Liu
---
### mergify @ 2025-09-25T04:59:07Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @david6666666.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### LJH-LBJ @ 2025-09-25T09:20:37Z

@benchislett @luccafong @DarkLight1337  I think this PR is ok, please check
---
### LJH-LBJ @ 2025-09-26T06:43:44Z

@DarkLight1337 @Isotr0py Is there anything else need to modify?
---
### mergify @ 2025-09-26T16:03:35Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @david6666666.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### DarkLight1337 @ 2025-09-27T00:06:01Z

PTAL at the failing test
---
### mergify @ 2025-09-27T01:30:09Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @david6666666.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### LJH-LBJ @ 2025-09-27T01:57:26Z

> PTAL at the failing test

skip the test_eagle_correctness, because this flash attention build does not support headdim not being a multiple of 32
---
### taoyun951753 @ 2025-09-29T10:15:02Z

The online service, deployed with the Qwen2.5-VL-7B model, encountered a corner case that caused the service to crash. The input_ids processed by the propose function in eagle.py generated an image_token_id, resulting in a shape mismatch with mm_embeds.


---
### taoyun951753 @ 2025-09-29T10:19:26Z

> mm embds shape: [torch.Size([546, 3584])]
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m before propose input ids: target token ids: tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151653,  53481,  41683,  17254,   9370,  45930,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102073,  43815,   8997, 101140,  37945, 104317, 102650,  15946,  64471,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         101114,   1773,  62244, 102650,  15946, 101475,  37945,  53481, 102650,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         101047, 102122,  24968,  62244, 102650,  15946, 101114,  37945,  53481,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102650,  15946, 104013, 109318, 101070,   8997,  14880,  23031,   2236,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          68805,  99553,      7, 104346,   3837, 100714,  53481,   3837,  99226,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          98237,  53481,      8, 101124,  66017,   3837,  90919, 104346,  14880,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          11622,   2073, 106468,   5373, 101977,   5373,  42140,  99605,    854,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         104787,   1773, 101475, 102122,  15946,  99226,  98237,  53481,  91680,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102298, 104766,   5373,  44091,   3837, 100714,  53481,  46448, 108593,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102298, 102122, 106336,  49567, 104449,   1773, 101114, 102122,  15946,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          99226,  98237,  53481,  91680, 102298, 104013,   5373, 104766,   5373,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102196,   3837, 100714,  53481,  46448, 108593, 102298, 104013, 104363,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           5373, 104766,  99719,  49567, 100405, 104449,   1773, 151645,    198,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151644,  77091,    198,  57218, 112531,  99194,  38035,  17447,  24562,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         100544,   3837, 104013, 109612,  18830,  17323,     71,   3022,     43,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          46944, 112531,  99194,  38035,  17447], device='cuda:0',
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m        dtype=torch.int32), next token ids: tensor([  4913, 111413, 110607, 100010, 105611], device='cuda:0',
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m        dtype=torch.int32) ,**mm_embeds shape: [torch.Size([546, 3584])]**
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m propose input ids: tensor([  8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m            198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         151655, 151655, 151653,  53481,  41683,  17254,   9370,  45930, 102073,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          43815,   8997, 101140,  37945, 104317, 102650,  15946,  64471, 101114,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           1773,  62244, 102650,  15946, 101475,  37945,  53481, 102650, 101047,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102122,  24968,  62244, 102650,  15946, 101114,  37945,  53481, 102650,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          15946, 104013, 109318, 101070,   8997,  14880,  23031,   2236,  68805,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          99553,      7, 104346,   3837, 100714,  53481,   3837,  99226,  98237,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          53481,      8, 101124,  66017,   3837,  90919, 104346,  14880,  11622,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           2073, 106468,   5373, 101977,   5373,  42140,  99605,    854, 104787,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           1773, 101475, 102122,  15946,  99226,  98237,  53481,  91680, 102298,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         104766,   5373,  44091,   3837, 100714,  53481,  46448, 108593, 102298,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         102122, 106336,  49567, 104449,   1773, 101114, 102122,  15946,  99226,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          98237,  53481,  91680, 102298, 104013,   5373, 104766,   5373, 102196,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           3837, 100714,  53481,  46448, 108593, 102298, 104013, 104363,   5373,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         104766,  99719,  49567, 100405, 104449,   1773, 151645,    198, 151644,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m          77091,    198,   4913, 112531, 111413,  38035,  17447,  24562, 100544,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m           3837, 110607, 109612,  18830, 100010,     71,   3022,     43,  46944,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m         105611,  99194,  38035,  17447, **151655**], device='cuda:0',
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m        dtype=torch.int32), mm embeds  shape :[torch.Size([546, 3584])]
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m **utils.py: is_multimodal sum: 547, emb shape 0:546**
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:402: masked_scatter_size_check: block: [0,0,0], thread: [0,0,0] Assertion `totalElements <= srcSize` failed.
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:69] Dumping input data for V1 LLM engine (v0.11.0rc2.dev0+g0d9fe260d.d20250926) with config: model='/lhf-model/Ipc/Qwen2.5-VL-7B-Instruct-20250829', speculative_config=SpeculativeConfig(method='eagle3', model='/mnt/sharedLLMs/qwen2.5-vl-7b-eagle3-specforge-v5-epoch1', num_spec_tokens=4), tokenizer='/lhf-model/Ipc/Qwen2.5-VL-7B-Instruct-20250829', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=base_model, enable_prefix_caching=True, chunked_prefill_enabled=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"/root/.cache/vllm/torch_compile_cache/6cd3220e9b","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer","vllm.gdn_attention"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"use_inductor_graph_partition":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":"/root/.cache/vllm/torch_compile_cache/6cd3220e9b/rank_0_0/backbone"}, 
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id=chatcmpl-c0150128df3f4637864541d650a02fb9,prompt_token_ids_len=687,mm_features=[MultiModalFeatureSpec(data={'image_grid_thw': MultiModalFieldElem(modality='image', key='image_grid_thw', data=tensor([ 1, 42, 52]), field=MultiModalBatchedField()), 'pixel_values': MultiModalFieldElem(modality='image', key='pixel_values', data=tensor([[1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         ...,
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406]],
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]        dtype=torch.bfloat16), field=MultiModalFlatField(slices=[[slice(0, 2184, None)]], dim=0))}, modality='image', identifier='089654d81821ee9d7bbe74606f03e9df82789e90e7bb59dbf5bf3b1df03c527f', mm_position=PlaceholderRange(offset=15, length=546, is_embed=None))],sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.05, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None),block_ids=([1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341],),num_computed_tokens=0,lora_request=None,prompt_embeds_shape=None)], scheduled_cached_reqs=CachedRequestData(req_ids=['chatcmpl-0721c0b185fd419c9a4594955b125766', 'chatcmpl-c123b1c2a6124ba4b6751f9295fedb60', 'chatcmpl-aa2319efc98c4342838ea42e267ab5d2', 'chatcmpl-029d33be6527448bb50c1417a1d02b0f'], resumed_from_preemption=[false, false, false, false], new_token_ids=[], new_block_ids=[[[1297]], null, null, [[1298]]], num_computed_tokens=[3880, 899, 679, 633]), num_scheduled_tokens={chatcmpl-c0150128df3f4637864541d650a02fb9: 687, chatcmpl-0721c0b185fd419c9a4594955b125766: 5, chatcmpl-c123b1c2a6124ba4b6751f9295fedb60: 5, chatcmpl-aa2319efc98c4342838ea42e267ab5d2: 5, chatcmpl-029d33be6527448bb50c1417a1d02b0f: 5}, total_num_scheduled_tokens=707, scheduled_spec_decode_tokens={chatcmpl-029d33be6527448bb50c1417a1d02b0f: [112531, 99194, 38035, 17447], chatcmpl-0721c0b185fd419c9a4594955b125766: [100544, 3837, 104013, 109612], chatcmpl-c123b1c2a6124ba4b6751f9295fedb60: [17323, 71, 3022, 43], chatcmpl-aa2319efc98c4342838ea42e267ab5d2: [112531, 99194, 38035, 17447]}, scheduled_encoder_inputs={chatcmpl-c0150128df3f4637864541d650a02fb9: [0]}, num_common_prefix_blocks=[0], finished_req_ids=['chatcmpl-e4e8edd6366042429bfaebe5479951bb'], free_encoder_mm_hashes=['dd8c6601b7dec1227a8dbc366c398edb0c05d6a568ab4153729d1df02fe14e88'], structured_output_request_ids={}, grammar_bitmask=null, kv_connector_metadata=null)
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [dump_input.py:79] Dumping scheduler stats: SchedulerStats(num_running_reqs=5, num_waiting_reqs=0, step_counter=0, current_wave=0, kv_cache_usage=0.26257668711656446, prefix_cache_stats=PrefixCacheStats(reset=False, requests=1, queries=687, hits=0), spec_decoding_stats=None, kv_connector_stats=None, num_corrupted_reqs=0)
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710] EngineCore encountered a fatal error.
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710] Traceback (most recent call last):
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 409, in _merge_multimodal_embeddings
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710]     inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1),
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710] torch.AcceleratorError: CUDA error: device-side assert triggered
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
^[[1;36m(EngineCore_DP0 pid=996225)^[[0;0m ERROR 09-29 03:07:32 [core.py:710] 
---
### ggg-s @ 2025-09-29T10:24:21Z

> > mm embds shape: [torch.Size([546, 3584])]
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m before propose input ids: target token ids: tensor([151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151653,  53481,  41683,  17254,   9370,  45930,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102073,  43815,   8997, 101140,  37945, 104317, 102650,  15946,  64471,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         101114,   1773,  62244, 102650,  15946, 101475,  37945,  53481, 102650,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         101047, 102122,  24968,  62244, 102650,  15946, 101114,  37945,  53481,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102650,  15946, 104013, 109318, 101070,   8997,  14880,  23031,   2236,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          68805,  99553,      7, 104346,   3837, 100714,  53481,   3837,  99226,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          98237,  53481,      8, 101124,  66017,   3837,  90919, 104346,  14880,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          11622,   2073, 106468,   5373, 101977,   5373,  42140,  99605,    854,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         104787,   1773, 101475, 102122,  15946,  99226,  98237,  53481,  91680,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102298, 104766,   5373,  44091,   3837, 100714,  53481,  46448, 108593,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102298, 102122, 106336,  49567, 104449,   1773, 101114, 102122,  15946,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          99226,  98237,  53481,  91680, 102298, 104013,   5373, 104766,   5373,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102196,   3837, 100714,  53481,  46448, 108593, 102298, 104013, 104363,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           5373, 104766,  99719,  49567, 100405, 104449,   1773, 151645,    198,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151644,  77091,    198,  57218, 112531,  99194,  38035,  17447,  24562,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         100544,   3837, 104013, 109612,  18830,  17323,     71,   3022,     43,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          46944, 112531,  99194,  38035,  17447], device='cuda:0',
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m        dtype=torch.int32), next token ids: tensor([  4913, 111413, 110607, 100010, 105611], device='cuda:0',
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m        dtype=torch.int32) ,**mm_embeds shape: [torch.Size([546, 3584])]**
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m propose input ids: tensor([  8948,    198,   2610,    525,    264,  10950,  17847,     13, 151645,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m            198, 151644,    872,    198, 151652, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         151655, 151655, 151653,  53481,  41683,  17254,   9370,  45930, 102073,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          43815,   8997, 101140,  37945, 104317, 102650,  15946,  64471, 101114,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           1773,  62244, 102650,  15946, 101475,  37945,  53481, 102650, 101047,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102122,  24968,  62244, 102650,  15946, 101114,  37945,  53481, 102650,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          15946, 104013, 109318, 101070,   8997,  14880,  23031,   2236,  68805,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          99553,      7, 104346,   3837, 100714,  53481,   3837,  99226,  98237,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          53481,      8, 101124,  66017,   3837,  90919, 104346,  14880,  11622,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           2073, 106468,   5373, 101977,   5373,  42140,  99605,    854, 104787,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           1773, 101475, 102122,  15946,  99226,  98237,  53481,  91680, 102298,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         104766,   5373,  44091,   3837, 100714,  53481,  46448, 108593, 102298,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         102122, 106336,  49567, 104449,   1773, 101114, 102122,  15946,  99226,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          98237,  53481,  91680, 102298, 104013,   5373, 104766,   5373, 102196,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           3837, 100714,  53481,  46448, 108593, 102298, 104013, 104363,   5373,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         104766,  99719,  49567, 100405, 104449,   1773, 151645,    198, 151644,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m          77091,    198,   4913, 112531, 111413,  38035,  17447,  24562, 100544,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m           3837, 110607, 109612,  18830, 100010,     71,   3022,     43,  46944,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m         105611,  99194,  38035,  17447, **151655**], device='cuda:0',
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m        dtype=torch.int32), mm embeds  shape :[torch.Size([546, 3584])]
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m **utils.py: is_multimodal sum: 547, emb shape 0:546**
> > /pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:402: masked_scatter_size_check: block: [0,0,0], thread: [0,0,0] Assertion `totalElements <= srcSize` failed.
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:69] Dumping input data for V1 LLM engine (v0.11.0rc2.dev0+g0d9fe260d.d20250926) with config: model='/lhf-model/Ipc/Qwen2.5-VL-7B-Instruct-20250829', speculative_config=SpeculativeConfig(method='eagle3', model='/mnt/sharedLLMs/qwen2.5-vl-7b-eagle3-specforge-v5-epoch1', num_spec_tokens=4), tokenizer='/lhf-model/Ipc/Qwen2.5-VL-7B-Instruct-20250829', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=8192, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=base_model, enable_prefix_caching=True, chunked_prefill_enabled=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"/root/.cache/vllm/torch_compile_cache/6cd3220e9b","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2","vllm.mamba_mixer","vllm.short_conv","vllm.linear_attention","vllm.plamo2_mamba_mixer","vllm.gdn_attention"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"cudagraph_mode":1,"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"use_inductor_graph_partition":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":"/root/.cache/vllm/torch_compile_cache/6cd3220e9b/rank_0_0/backbone"},
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76] Dumping scheduler output for model execution: SchedulerOutput(scheduled_new_reqs=[NewRequestData(req_id=chatcmpl-c0150128df3f4637864541d650a02fb9,prompt_token_ids_len=687,mm_features=[MultiModalFeatureSpec(data={'image_grid_thw': MultiModalFieldElem(modality='image', key='image_grid_thw', data=tensor([ 1, 42, 52]), field=MultiModalBatchedField()), 'pixel_values': MultiModalFieldElem(modality='image', key='pixel_values', data=tensor([[1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         ...,
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]         [1.9297, 1.9297, 1.9297,  ..., 2.1406, 2.1406, 2.1406]],
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:76]        dtype=torch.bfloat16), field=MultiModalFlatField(slices=[[slice(0, 2184, None)]], dim=0))}, modality='image', identifier='089654d81821ee9d7bbe74606f03e9df82789e90e7bb59dbf5bf3b1df03c527f', mm_position=PlaceholderRange(offset=15, length=546, is_embed=None))],sampling_params=SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.05, temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, seed=None, stop=[], stop_token_ids=[151643], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=128, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None),block_ids=([1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341],),num_computed_tokens=0,lora_request=None,prompt_embeds_shape=None)], scheduled_cached_reqs=CachedRequestData(req_ids=['chatcmpl-0721c0b185fd419c9a4594955b125766', 'chatcmpl-c123b1c2a6124ba4b6751f9295fedb60', 'chatcmpl-aa2319efc98c4342838ea42e267ab5d2', 'chatcmpl-029d33be6527448bb50c1417a1d02b0f'], resumed_from_preemption=[false, false, false, false], new_token_ids=[], new_block_ids=[[[1297]], null, null, [[1298]]], num_computed_tokens=[3880, 899, 679, 633]), num_scheduled_tokens={chatcmpl-c0150128df3f4637864541d650a02fb9: 687, chatcmpl-0721c0b185fd419c9a4594955b125766: 5, chatcmpl-c123b1c2a6124ba4b6751f9295fedb60: 5, chatcmpl-aa2319efc98c4342838ea42e267ab5d2: 5, chatcmpl-029d33be6527448bb50c1417a1d02b0f: 5}, total_num_scheduled_tokens=707, scheduled_spec_decode_tokens={chatcmpl-029d33be6527448bb50c1417a1d02b0f: [112531, 99194, 38035, 17447], chatcmpl-0721c0b185fd419c9a4594955b125766: [100544, 3837, 104013, 109612], chatcmpl-c123b1c2a6124ba4b6751f9295fedb60: [17323, 71, 3022, 43], chatcmpl-aa2319efc98c4342838ea42e267ab5d2: [112531, 99194, 38035, 17447]}, scheduled_encoder_inputs={chatcmpl-c0150128df3f4637864541d650a02fb9: [0]}, num_common_prefix_blocks=[0], finished_req_ids=['chatcmpl-e4e8edd6366042429bfaebe5479951bb'], free_encoder_mm_hashes=['dd8c6601b7dec1227a8dbc366c398edb0c05d6a568ab4153729d1df02fe14e88'], structured_output_request_ids={}, grammar_bitmask=null, kv_connector_metadata=null)
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [dump_input.py:79] Dumping scheduler stats: SchedulerStats(num_running_reqs=5, num_waiting_reqs=0, step_counter=0, current_wave=0, kv_cache_usage=0.26257668711656446, prefix_cache_stats=PrefixCacheStats(reset=False, requests=1, queries=687, hits=0), spec_decoding_stats=None, kv_connector_stats=None, num_corrupted_reqs=0)
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710] EngineCore encountered a fatal error.
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710] Traceback (most recent call last):
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 409, in _merge_multimodal_embeddings
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710]     inputs_embeds.masked_scatter_(is_multimodal.unsqueeze(-1),
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710] torch.AcceleratorError: CUDA error: device-side assert triggered
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> > �[1;36m(EngineCore_DP0 pid=996225)�[0;0m ERROR 09-29 03:07:32 [core.py:710]

Please print the error logs from the very beginning.

make <image> placeholders and vision patches the same length (trim to match) inside merge_multimodal_embeddings

---
### DarkLight1337 @ 2025-09-29T10:48:08Z

This corner case should be mitigated by https://github.com/vllm-project/vllm/pull/16229, which was merged recently
---
### benchislett @ 2025-10-08T13:51:06Z

Does anyone know why this PR removes torch compile support for llama_eagle3?
---
### taoyun951753 @ 2025-10-09T08:28:52Z

The qwen2.5-vl eagle3 model encounters an error when using @support_torch_compile 

> Traceback (most recent call last):
> (EngineCore_DP0 pid=8139)   File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
> (EngineCore_DP0 pid=8139)     self.run()
> (EngineCore_DP0 pid=8139)   File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
> (EngineCore_DP0 pid=8139)     self._target(*self._args, **self._kwargs)
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 716, in run_engine_core
> (EngineCore_DP0 pid=8139)     raise e
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 703, in run_engine_core
> (EngineCore_DP0 pid=8139)     engine_core = EngineCoreProc(*args, **kwargs)
> (EngineCore_DP0 pid=8139)                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 499, in __init__
> (EngineCore_DP0 pid=8139)     super().__init__(vllm_config, executor_class, log_stats,
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 93, in __init__
> (EngineCore_DP0 pid=8139)     self._initialize_kv_caches(vllm_config)
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py", line 208, in _initialize_kv_caches
> (EngineCore_DP0 pid=8139)     self.model_executor.initialize_from_config(kv_cache_configs)
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/abstract.py", line 75, in initialize_from_config
> (EngineCore_DP0 pid=8139)     self.collective_rpc("compile_or_warm_up_model")
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 83, in collective_rpc
> (EngineCore_DP0 pid=8139)     return [run_method(self.driver_worker, method, args, kwargs)]
> (EngineCore_DP0 pid=8139)             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 3121, in run_method
> (EngineCore_DP0 pid=8139)     return func(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 344, in compile_or_warm_up_model
> (EngineCore_DP0 pid=8139)     cuda_graph_memory_bytes = self.model_runner.capture_model()
> (EngineCore_DP0 pid=8139)                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 3520, in capture_model
> (EngineCore_DP0 pid=8139)     self._capture_cudagraphs(
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 3597, in _capture_cudagraphs
> (EngineCore_DP0 pid=8139)     self._dummy_run(num_tokens,
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
> (EngineCore_DP0 pid=8139)     return func(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 3240, in _dummy_run
> (EngineCore_DP0 pid=8139)     self.drafter.dummy_run(num_tokens,use_cudagraphs=cudagraph_runtime_mode!=CUDAGraphMode.NONE)
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 120, in decorate_context
> (EngineCore_DP0 pid=8139)     return func(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py", line 966, in dummy_run
> (EngineCore_DP0 pid=8139)     self.model(
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
> (EngineCore_DP0 pid=8139)     return self._call_impl(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
> (EngineCore_DP0 pid=8139)     return forward_call(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl_eagle3.py", line 258, in forward
> (EngineCore_DP0 pid=8139)     return self.model(input_ids, positions, hidden_states, inputs_embeds)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 317, in __call__
> (EngineCore_DP0 pid=8139)     model_output = self.forward(*args, **kwargs)
> (EngineCore_DP0 pid=8139)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen2_5_vl_eagle3.py", line 154, in forward
> (EngineCore_DP0 pid=8139)     def forward(
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py", line 375, in __call__
> (EngineCore_DP0 pid=8139)     return super().__call__(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
> (EngineCore_DP0 pid=8139)     return self._call_impl(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
> (EngineCore_DP0 pid=8139)     return forward_call(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py", line 929, in _fn
> (EngineCore_DP0 pid=8139)     return fn(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/fx/graph_module.py", line 848, in call_wrapped
> (EngineCore_DP0 pid=8139)     return self._wrapped_call(self, *args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/fx/graph_module.py", line 424, in __call__
> (EngineCore_DP0 pid=8139)     raise e
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/fx/graph_module.py", line 411, in __call__
> (EngineCore_DP0 pid=8139)     return super(self.cls, obj).__call__(*args, **kwargs)  # type: ignore[misc]
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
> (EngineCore_DP0 pid=8139)     return self._call_impl(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
> (EngineCore_DP0 pid=8139)     return forward_call(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "<eval_with_key>.271", line 18, in forward
> (EngineCore_DP0 pid=8139)     submod_0 = self.submod_0(l_input_ids_, s72, l_self_modules_embed_tokens_parameters_weight_, l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_, l_hidden_states_, l_self_modules_layers_modules_0_modules_hidden_norm_parameters_weight_, l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_, l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_, l_positions_, s18);  l_input_ids_ = l_self_modules_embed_tokens_parameters_weight_ = l_self_modules_layers_modules_0_modules_input_layernorm_parameters_weight_ = l_self_modules_layers_modules_0_modules_hidden_norm_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_qkv_proj_parameters_weight_ = l_self_modules_layers_modules_0_modules_self_attn_modules_rotary_emb_buffers_cos_sin_cache_ = l_positions_ = s18 = None
> (EngineCore_DP0 pid=8139)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/cuda_graph.py", line 121, in __call__
> (EngineCore_DP0 pid=8139)     return self.runnable(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/piecewise_backend.py", line 96, in __call__
> (EngineCore_DP0 pid=8139)     return self.compiled_graph_for_general_shape(*args)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py", line 929, in _fn
> (EngineCore_DP0 pid=8139)     return fn(*args, **kwargs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_functorch/aot_autograd.py", line 1241, in forward
> (EngineCore_DP0 pid=8139)     return compiled_fn(full_args)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 384, in runtime_wrapper
> (EngineCore_DP0 pid=8139)     all_outs = call_func_at_runtime_with_args(
> (EngineCore_DP0 pid=8139)                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/utils.py", line 126, in call_func_at_runtime_with_args
> (EngineCore_DP0 pid=8139)     out = normalize_as_list(f(args))
> (EngineCore_DP0 pid=8139)                             ^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 750, in inner_fn
> (EngineCore_DP0 pid=8139)     outs = compiled_fn(args)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 556, in wrapper
> (EngineCore_DP0 pid=8139)     return compiled_fn(runtime_args)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/output_code.py", line 584, in __call__
> (EngineCore_DP0 pid=8139)     return self.current_callable(inputs)
> (EngineCore_DP0 pid=8139)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/usr/local/lib/python3.12/dist-packages/torch/_inductor/utils.py", line 2716, in run
> (EngineCore_DP0 pid=8139)     out = model(new_inputs)
> (EngineCore_DP0 pid=8139)           ^^^^^^^^^^^^^^^^^
> (EngineCore_DP0 pid=8139)   File "/root/.cache/vllm/torch_compile_cache/8304a885d2/rank_0_0/inductor_cache/2i/c2iggmzrq2sxtsm37f72hc22obqnk3swtexiwdxv5crnyca2zvan.py", line 622, in call
> (EngineCore_DP0 pid=8139)     assert_size_stride(arg8_1, (3, s18), (s18, 1))
> (EngineCore_DP0 pid=8139) AssertionError: expected size 3==3, stride 2048==1536 at dim=0
> (EngineCore_DP0 pid=8139) This error most often comes from a incorrect fake (aka meta) kernel for a custom op.


---