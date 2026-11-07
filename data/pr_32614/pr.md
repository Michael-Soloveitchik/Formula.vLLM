# PR #32614: fix: Add glm4_moe_lite to MLA detection

- field: attention  | area: model_executor/layers
- author: marksverdhei
- state: MERGED  created: 2026-01-19T20:00:25Z  merged: 2026-01-23T20:38:57Z
- url: https://github.com/vllm-project/vllm/pull/32614
- labels: ready, v1, nvidia

## Body

## Summary

- Add `glm4_moe_lite` and `glm4_moe_lite_mtp` to `is_deepseek_mla()` check in `model_arch_config_convertor.py`

GLM-4.7-Flash (`glm4_moe_lite`) uses Multi-head Latent Attention (MLA) via `Glm4MoeLiteMLAAttention` (which inherits from `DeepseekV2MLAAttention`) but was missing from the MLA detection.

Without this fix, vLLM falls back to standard KV caching instead of efficient MLA caching, resulting in ~4x higher KV cache memory usage.

Note: `glm4_moe` is intentionally NOT included as it uses standard attention (`Glm4MoeAttention` with `vllm.attention.layer.Attention`).

Co-authored with @mgoin 

NOTE: SM100 has issues with support for this model with various MLA decode and prefill kernels, so the following changes were made to support default inference there:
* Disable trtllm prefill and flashinfer prefill if we didn't find DeepSeek R1 compatible MLA dimensions
* Disable flashinfer mla if we didn't find DeepSeek R1 compatible MLA dimensions
* Explicitly enable cutlass mla so the block_size=128 gets enforced
* This means that on SM100 we will run with CUTLASS_MLA decode and FA2 prefill by default for this model, I also tested that TRITON_MLA works.

## Test Plan

- [x] Tested with `marksverdhei/GLM-4.7-Flash-fp8` on 2x RTX 3090
- [x] Verified MLA is detected and efficient KV caching is used
- [x] Model runs with 14.7 GB VRAM per GPU


## Comments

### mgoin @ 2026-01-20T17:02:08Z

@LucasWilkinson @MatthewBonanni When I run with this PR on B200, I get this error. I think we need to change the kernel support registration
```
(EngineCore_DP0 pid=1261690)   File "/home/mgoin/code/vllm/.venv/lib/python3.12/site-packages/flashinfer/decode.py", line 2491, in _check_trtllm_gen_mla_shape
(EngineCore_DP0 pid=1261690)     raise ValueError(f"Expected qk_nope_head_dim == 128, got {qk_nope_head_dim}")
(EngineCore_DP0 pid=1261690) ValueError: Expected qk_nope_head_dim == 128, got 192
```
---
### marksverdhei @ 2026-01-22T14:02:53Z

> Blocking merge while we investigate which MLA backends are/can be supported for this model. For instance forcing TRITON_MLA on B200 results in 0% on GSM8k

Could it be because of the known looping issues that Lama CPP also had?  Like 0% because it doesn't arrive at an answer and only generates think tokens or did you control for it already? I have never contributed to a vLLM before, so I'm not completely familiar with the testing procedures
---
### mdierolf @ 2026-01-23T17:04:38Z

LGTM!

Works well on RTX 6000 Blackwell, model is unusable without this.

Nobody is going to run this small model on B200 anyway, merge it and fix B200 mañana!
---
### marksverdhei @ 2026-01-24T09:17:48Z

Thank you for merging! i can now call myself a proud contributor of vllm, with those two lines of strings! :laughing: Way to kick off my 2026 new years resolutions!
---
### Xiaojinhua @ 2026-01-27T03:04:52Z

Is there a new version released after this fix, or only be installed through the latest source code now.
---
### esmeetu @ 2026-01-28T00:59:51Z

@Xiaojinhua You can install the nightly version: 
```
uv pip install -U vllm --pre \ 
--extra-index-url https://wheels.vllm.ai/nightly/cu129 \
--extra-index-url https://download.pytorch.org/whl/cu129 \
--index-strategy unsafe-best-match
```
---
### gaby @ 2026-02-18T11:55:34Z

@esmeetu @marksverdhei even with these changes weeks ago, the glm-4.7-flash model does *not* start because `transformers` is pinned to `<5.0.0`. 

Tldr: using nightly docker image
---
### mgoin @ 2026-02-18T13:09:32Z

@gaby yes we still recommend in the recipe for the model to install transformers from source https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html
---
### gaby @ 2026-02-18T13:20:55Z

@mgoin Sadly that doesn't work using vLLM with Docker. Since we rely on the image provided by the vLLM team.
---
### mgoin @ 2026-02-18T13:23:51Z

@gaby you can extend the existing image like this 
![image](https://github.com/user-attachments/assets/d0190831-4498-4632-a5d7-7e7ea0b4d537)

https://x.com/thezachmueller/status/2014354173492432942?s=46&t=jLcDgQXDbYe6HgFmTNYgpg
---
### gaby @ 2026-02-18T13:26:39Z

@mgoin Will give that a try, thanks! 
---
### esmeetu @ 2026-02-18T13:35:33Z

@gaby You can try this image: vllm/vllm-openai:glm5
---
### gaby @ 2026-02-18T13:46:51Z

@esmeetu thanks!
---