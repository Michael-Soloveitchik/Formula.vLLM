# PR #26669: support flashinfer_fp4 moe for 5090 gpu (#26669)

# Issue

## Original PR description (captured source)

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



## Full discussion

See [26669.md](./26669.md) and [26669.json](./26669.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/26669
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/26669.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `0f4783c39d1b98942e6ac52269aa327e72c737672643dfda1958a021466afb83`

Existing semantic notes in [26669.sem.json](./26669.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py`

### Diff

```diff
diff --git a/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py b/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py
index ddb74a27dc12..5ce0188b60ae 100644
--- a/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py
+++ b/vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py
@@ -32,7 +32,7 @@ def is_flashinfer_fp4_cutlass_moe_available() -> bool:
         envs.VLLM_USE_FLASHINFER_MOE_FP4
         and has_flashinfer_cutlass_fused_moe()
         and current_platform.is_cuda()
-        and current_platform.is_device_capability(100)
+        and current_platform.has_device_capability(100)
     )
 
 
```
