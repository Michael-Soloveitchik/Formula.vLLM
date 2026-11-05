# PR #20788: [Meta] Official Eagle mm support, first enablement on llama4 (#20788)

# Issue

## Original PR description (captured source)

## Purpose
Enable MM inference for EAGLE, targeting mllama4 in this PR but generally easy to extend to other models
Issue with this PR:
MM chunked prefill needs to be disabled, or set mbnt to a large number. There will be a follow up PR using unshifting eagle prefill fix this

## Test Plan
```
CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py  --num_spec_tokens 7 --num_prompts 1 --method eagle  --model_dir /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --eagle_dir /home/zhiweiz/local/models/scout_draft_HF_20250605_202942 --tp 4 --custom-mm-prompts
```

vllm serve with benchmark testing
cmd
```
#!/bin/bash
# Configuration of environment variables
export VLLM_USE_MODELSCOPE=False
export VLLM_TORCH_PROFILER_DIR=~/vllm_profile
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_V1=1
export SAFETENSORS_FAST_GPU=1
# Command to run the vllm server
spec_dec_config='{"method": "eagle", "model": "/home/zhiweiz/local/models/scout_draft_HF_20250605_202942", "prefill_token_shift": false, "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4, "max_model_len": 32768}'
vllm serve /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --disable-log-requests \
    -tp 4 \
    --max-num-seqs 128 \
    --max_num_batched_tokens=80000 \
    --max-model-len=32768 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --speculative-config="$spec_dec_config" \
    --num-lookahead-slots=3 \
    2>&1 | tee /data/users/$USER/logs/server/vllm_17b16e_vllm_serving$(date +%Y%m%d_%H%M%S).log
```
baseline without `--speculative-config` flag

benchmark cmd
```
python benchmarks/benchmark_serving.py --backend openai-chat --model /home/zhiweiz/local/models/scout_base_HF_20250605_201140 --dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat --seed 0 --max-concurrency 16 --endpoint /v1/chat/completions 2>&1 | tee /data/users/$USER/tmp/vllm_17b16e_vllm_loadgen$(date +%Y%m%d_%H%M%S).log\n
```

## Test Result
```
--------------------------------------------------
total_num_output_tokens: 155
num_drafts: 45
num_draft_tokens: 315
num_accepted_tokens: 110
mean acceptance length: 3.44
--------------------------------------------------
acceptance at token 0: 0.87
acceptance at token 1: 0.69
acceptance at token 2: 0.47
acceptance at token 3: 0.20
acceptance at token 4: 0.16
acceptance at token 5: 0.04
acceptance at token 6: 0.02
```

Eagle MM benchmark
```
Maximum request concurrency: 16
100%|██████████| 1000/1000 [03:16<00:00,  5.10it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  196.05
Total input tokens:                      87321
Total generated tokens:                  117053
Request throughput (req/s):              5.10
Output token throughput (tok/s):         597.04
Total Token throughput (tok/s):          1042.43
---------------Time to First Token----------------
Mean TTFT (ms):                          201.32
Median TTFT (ms):                        164.53
P99 TTFT (ms):                           797.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          26.02
Median TPOT (ms):                        24.69
P99 TPOT (ms):                           48.73
---------------Inter-token Latency----------------
Mean ITL (ms):                           58.54
Median ITL (ms):                         40.44
P99 ITL (ms):                            160.04
==================================================
```

```
Maximum request concurrency: 16
100%|██████████| 1000/1000 [04:02<00:00,  4.12it/s]
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  242.87
Total input tokens:                      87321
Total generated tokens:                  116716
Request throughput (req/s):              4.12
Output token throughput (tok/s):         480.58
Total Token throughput (tok/s):          840.12
---------------Time to First Token----------------
Mean TTFT (ms):                          187.59
Median TTFT (ms):                        157.39
P99 TTFT (ms):                           826.26
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          32.00
Median TPOT (ms):                        31.85
P99 TPOT (ms):                           37.72
---------------Inter-token Latency----------------
Mean ITL (ms):                           31.55
Median ITL (ms):                         22.52
P99 ITL (ms):                            131.54
==================================================
```


 ## Follow ups
Need to make offline inference work with vision datasets


## Full discussion

See [20788.md](./20788.md) and [20788.json](./20788.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/20788
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/20788.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `ed7d287ae3de0704b3910c1cf8101aef44fff5a06843e1c46cacd61a2ef18cad`

Existing semantic notes in [20788.sem.json](./20788.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `examples/offline_inference/spec_decode.py`
- `tests/v1/e2e/test_spec_decode.py`
- `vllm/model_executor/models/llama4.py`
- `vllm/model_executor/models/llama4_eagle.py`
- `vllm/model_executor/models/llama_eagle.py`
- `vllm/model_executor/models/llama_eagle3.py`
- `vllm/v1/spec_decode/eagle.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff

```diff
diff --git a/examples/offline_inference/spec_decode.py b/examples/offline_inference/spec_decode.py
index ce735f3b27df..184c30891eca 100644
--- a/examples/offline_inference/spec_decode.py
+++ b/examples/offline_inference/spec_decode.py
@@ -13,6 +13,38 @@
     from argparse import ArgumentParser as FlexibleArgumentParser
 
 
+QUESTION = "What is the content of each image?"
+IMAGE_URLS = [
+    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
+    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1920px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/3/30/George_the_amazing_guinea_pig.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1f/Oryctolagus_cuniculus_Rcdo.jpg/1920px-Oryctolagus_cuniculus_Rcdo.jpg",
+    "https://upload.wikimedia.org/wikipedia/commons/9/98/Horse-and-pony.jpg",
+]
+
+
+def get_custom_mm_prompts(num_prompts):
+    prompts = []
+    for url in IMAGE_URLS:
+        prompts.append(
+            [
+                {"type": "image_url", "image_url": {"url": url}},
+                {"type": "text", "text": QUESTION},
+            ]
+        )
+    if num_prompts > len(IMAGE_URLS):
+        prompts = prompts * (num_prompts // len(IMAGE_URLS) + 1)
+
+    return [[{"role": "user", "content": prompt}] for prompt in prompts[:num_prompts]]
+
+
 def parse_args():
     parser = FlexibleArgumentParser()
     add_dataset_parser(parser)
@@ -35,6 +67,7 @@ def parse_args():
     parser.add_argument("--output-len", type=int, default=256)
     parser.add_argument("--model-dir", type=str, default=None)
     parser.add_argument("--eagle-dir", type=str, default=None)
+    parser.add_argument("--custom-mm-prompts", action="store_true")
     return parser.parse_args()
 
 
@@ -44,14 +77,26 @@ def main():
 
     model_dir = args.model_dir
     if args.model_dir is None:
+        if args.custom_mm_prompts:
+            raise ValueError(
+                "custom_mm_prompts requires mm based models"
+                "default llama3.1-8b-instruct is not mm based"
+                "please specify model_dir to give a mm based model"
+            )
         model_dir = "meta-llama/Llama-3.1-8B-Instruct"
     tokenizer = AutoTokenizer.from_pretrained(model_dir)
-
-    prompts = get_samples(args, tokenizer)
-    # add_special_tokens is False to avoid adding bos twice when using chat templates
-    prompt_ids = [
-        tokenizer.encode(prompt.prompt, add_special_tokens=False) for prompt in prompts
-    ]
+    args.custom_skip_chat_template = True
+
+    if not args.custom_mm_prompts:
+        prompts = get_samples(args, tokenizer)
+        # add_special_tokens is False to avoid adding bos twice
+        # when using chat templates
+        prompt_ids = [
+            tokenizer.encode(prompt.prompt, add_special_tokens=False)
+            for prompt in prompts
+        ]
+    else:
+        prompts = get_custom_mm_prompts(args.num_prompts)
 
     if args.method == "eagle" or args.method == "eagle3":
         eagle_dir = args.eagle_dir
@@ -85,10 +130,17 @@ def main():
         speculative_config=speculative_config,
         disable_log_stats=False,
         max_model_len=16384,
+        limit_mm_per_prompt={"image": 5},
+        disable_chunked_mm_input=True,
     )
 
     sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.output_len)
-    outputs = llm.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
+    if not args.custom_mm_prompts:
+        outputs = llm.generate(
+            prompt_token_ids=prompt_ids, sampling_params=sampling_params
+        )
+    else:
+        outputs = llm.chat(prompts, sampling_params=sampling_params)
 
     # print the generated text
     if args.print_output:
diff --git a/tests/v1/e2e/test_spec_decode.py b/tests/v1/e2e/test_spec_decode.py
index 2423f966acfa..31f25e94c5b4 100644
--- a/tests/v1/e2e/test_spec_decode.py
+++ b/tests/v1/e2e/test_spec_decode.py
@@ -3,29 +3,34 @@
 from __future__ import annotations
 
 import random
-from typing import Any
+from typing import Any, Union
 
 import pytest
 import torch
 
 from vllm import LLM, SamplingParams
+from vllm.assets.base import VLLM_S3_BUCKET_URL
+from vllm.assets.image import VLM_IMAGES_DIR
 from vllm.distributed import cleanup_dist_env_and_memory
 
 
-@pytest.fixture
-def test_prompts():
+def get_test_prompts(mm_enabled: bool):
     prompt_types = ["repeat", "sentence"]
+    if mm_enabled:
+        prompt_types.append("mm")
     num_prompts = 100
     prompts = []
 
     random.seed(0)
     random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)
+    print(f"Prompt types: {random_prompt_type_choices}")
 
     # Generate a mixed batch of prompts, some of which can be easily
     # predicted by n-gram matching and some which likely cannot.
     for kind in random_prompt_type_choices:
         word_choices = ["test", "temp", "hello", "where"]
         word = random.choice(word_choices)
+        prompt: Union[str, list[dict[str, Any]]] = ""
         if kind == "repeat":
             prompt = f"""
             please repeat the word '{word}' 10 times.
@@ -38,6 +43,21 @@ def test_prompts():
             uses the word {word} at least once.
             give no other output than that simple sentence without quotes.
             """
+        elif kind == "mm":
+            placeholders = [{
+                "type": "image_url",
+                "image_url": {
+                    "url":
+                    f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
+                },
+            }]
+            prompt = [
+                *placeholders,
+                {
+                    "type": "text",
+                    "text": "The meaning of the image is"
+                },
+            ]
         else:
             raise ValueError(f"Unknown prompt type: {kind}")
         prompts.append([{"role": "user", "content": prompt}])
@@ -57,7 +77,6 @@ def model_name():
 
 def test_ngram_correctness(
     monkeypatch: pytest.MonkeyPatch,
-    test_prompts: list[list[dict[str, Any]]],
     sampling_config: SamplingParams,
     model_name: str,
 ):
@@ -67,6 +86,7 @@ def test_ngram_correctness(
     '''
     with monkeypatch.context() as m:
         m.setenv("VLLM_USE_V1", "1")
+        test_prompts = get_test_prompts(mm_enabled=False)
 
         ref_llm = LLM(model=model_name, max_model_len=1024)
         ref_outputs = ref_llm.chat(test_prompts, sampling_config)
@@ -103,23 +123,32 @@ def test_ngram_correctness(
         cleanup_dist_env_and_memory()
 
 
-@pytest.mark.parametrize("model_setup", [
-    ("eagle", "meta-llama/Llama-3.1-8B-Instruct",
-     "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", 1),
-    ("eagle3", "meta-llama/Llama-3.1-8B-Instruct",
-     "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1),
-    pytest.param(
-        ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
-         "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
-        marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
-],
-                         ids=["llama3_eagle", "llama3_eagle3", "llama4_eagle"])
+@pytest.mark.parametrize(
+    ["model_setup", "mm_enabled"], [
+        (("eagle", "meta-llama/Llama-3.1-8B-Instruct",
+          "yuhuili/EAGLE-LLaMA3.1-Instruct-8B", 1), False),
+        (("eagle3", "meta-llama/Llama-3.1-8B-Instruct",
+          "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", 1), False),
+        pytest.param(
+            ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
+             "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
+            False,
+            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
+        pytest.param(
+            ("eagle", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
+             "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct", 4),
+            True,
+            marks=pytest.mark.skip(reason="Skipping due to CI OOM issues")),
+    ],
+    ids=["llama3_eagle", "llama3_eagle3", "llama4_eagle", "llama4_eagle_mm"])
 def test_eagle_correctness(
     monkeypatch: pytest.MonkeyPatch,
-    test_prompts: list[list[dict[str, Any]]],
     sampling_config: SamplingParams,
     model_setup: tuple[str, str, str, int],
+    mm_enabled: bool,
 ):
+    # Generate test prompts inside the function instead of using fixture
+    test_prompts = get_test_prompts(mm_enabled)
     '''
     Compare the outputs of a original LLM and a speculative LLM
     should be the same when using eagle speculative decoding.
diff --git a/vllm/model_executor/models/llama4.py b/vllm/model_executor/models/llama4.py
index 470e701d9801..60098209c39a 100644
--- a/vllm/model_executor/models/llama4.py
+++ b/vllm/model_executor/models/llama4.py
@@ -256,6 +256,7 @@ def __init__(
         super().__init__()
 
         self.layer_idx = extract_layer_index(prefix)
+        self.global_layer = config.no_rope_layers[self.layer_idx] == 0
         self.hidden_size = config.hidden_size
         rope_theta = config.rope_theta
         rope_scaling = config.rope_scaling
diff --git a/vllm/model_executor/models/llama4_eagle.py b/vllm/model_executor/models/llama4_eagle.py
index 222ab5dfaee4..ece490ff2f2a 100644
--- a/vllm/model_executor/models/llama4_eagle.py
+++ b/vllm/model_executor/models/llama4_eagle.py
@@ -37,8 +37,9 @@
 from vllm.model_executor.models.llama4 import (Llama4DecoderLayer,
                                                Llama4ForCausalLM)
 from vllm.model_executor.models.utils import extract_layer_index
+from vllm.multimodal.inputs import NestedTensors
 
-from .utils import AutoWeightsLoader, maybe_prefix
+from .utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings
 
 logger = init_logger(__name__)
 
@@ -78,15 +79,23 @@ def __init__(
         self.norm = RMSNorm(self.config.hidden_size,
                             eps=self.config.rms_norm_eps)
 
+    def get_input_embeddings(
+        self,
+        input_ids: torch.Tensor,
+    ) -> torch.Tensor:
+        return self.embed_tokens(input_ids)
+
     def forward(
         self,
         input_ids: Optional[torch.Tensor],
         positions: torch.Tensor,
         hidden_states: torch.Tensor,
+        inputs_embeds: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, torch.Tensor]:
-        input_embeds = self.embed_tokens(input_ids)
+        if inputs_embeds is None:
+            inputs_embeds = self.get_input_embeddings(input_ids)
         hidden_states = self.fc(
-            torch.cat((input_embeds, hidden_states), dim=-1))
+            torch.cat((inputs_embeds, hidden_states), dim=-1))
         residual = None
         for layer in self.layers:
             hidden_states, residual = layer(
@@ -190,8 +199,9 @@ def forward(
         input_ids: torch.Tensor,
         positions: torch.Tensor,
         hidden_states: torch.Tensor,
+        inputs_embeds: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, torch.Tensor]:
-        return self.model(input_ids, positions, hidden_states)
+        return self.model(input_ids, positions, hidden_states, inputs_embeds)
 
     def load_weights(self, weights: Iterable[tuple[str,
                                                    torch.Tensor]]) -> None:
@@ -212,3 +222,20 @@ def load_weights(self, weights: Iterable[tuple[str,
             model_weights[name] = loaded_weight
 
         loader.load_weights(model_weights.items())
+
+    def get_input_embeddings(
+        self,
+        input_ids: torch.Tensor,
+        multimodal_embeddings: Optional[NestedTensors] = None,
+    ) -> torch.Tensor:
+        inputs_embeds = self.model.get_input_embeddings(input_ids)
+
+        if multimodal_embeddings is not None:
+            inputs_embeds = merge_multimodal_embeddings(
+                input_ids,
+                inputs_embeds,
+                multimodal_embeddings,
+                self.config.image_token_index,
+            )
+
+        return inputs_embeds
diff --git a/vllm/model_executor/models/llama_eagle.py b/vllm/model_executor/models/llama_eagle.py
index c7690604c1d0..a4933b77e3a5 100644
--- a/vllm/model_executor/models/llama_eagle.py
+++ b/vllm/model_executor/models/llama_eagle.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 from collections.abc import Iterable
+from typing import Optional
 
 import torch
 import torch.nn as nn
@@ -148,7 +149,12 @@ def forward(
         input_ids: torch.Tensor,
         positions: torch.Tensor,
         hidden_states: torch.Tensor,
+        inputs_embeds: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        if inputs_embeds is not None:
+            raise NotImplementedError(
+                f"{type(self).__name__} does not support multimodal inputs yet."
+            )
         return self.model(input_ids, positions, hidden_states)
 
     def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
diff --git a/vllm/model_executor/models/llama_eagle3.py b/vllm/model_executor/models/llama_eagle3.py
index 7fc9fe2ebb6f..71275f0d5857 100644
--- a/vllm/model_executor/models/llama_eagle3.py
+++ b/vllm/model_executor/models/llama_eagle3.py
@@ -202,7 +202,12 @@ def forward(
         input_ids: torch.Tensor,
         positions: torch.Tensor,
         hidden_states: torch.Tensor,
+        inputs_embeds: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        if inputs_embeds is not None:
+            raise NotImplementedError(
+                f"{type(self).__name__} does not support multimodal inputs yet."
+            )
         return self.model(input_ids, positions, hidden_states)
 
     def compute_logits(
diff --git a/vllm/v1/spec_decode/eagle.py b/vllm/v1/spec_decode/eagle.py
index 63f6fc276189..302126dbe3d5 100644
--- a/vllm/v1/spec_decode/eagle.py
+++ b/vllm/v1/spec_decode/eagle.py
@@ -1,5 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+from typing import Optional
+
 import numpy as np
 import torch
 import torch.nn as nn
@@ -51,6 +53,9 @@ def __init__(
         # hidden size (e.g., Llama 3.3 70B).
         self.hidden_size = self.draft_model_config.get_hidden_size()
 
+        self.is_multimodal_model = vllm_config.model_config \
+            .is_multimodal_model
+
         self.use_cuda_graph = (self.vllm_config.compilation_config.level
                                == CompilationLevel.PIECEWISE and
                                not self.vllm_config.model_config.enforce_eager)
@@ -76,6 +81,11 @@ def __init__(
                                    device=device,
                                    dtype=torch.int32)
 
+        self.inputs_embeds = torch.zeros(
+            (self.max_num_tokens, self.hidden_size),
+            dtype=self.dtype,
+            device=device)
+
     def propose(
         self,
         # [num_tokens]
@@ -88,6 +98,7 @@ def propose(
         next_token_ids: torch.Tensor,
         common_attn_metadata: CommonAttentionMetadata,
         sampling_metadata: SamplingMetadata,
+        mm_embeds: Optional[list[torch.Tensor]] = None,
     ) -> torch.Tensor:
         num_tokens = target_token_ids.shape[0]
         batch_size = next_token_ids.shape[0]
@@ -128,14 +139,27 @@ def propose(
         # copy inputs to buffer for cudagraph
         self.positions[:num_tokens] = target_positions
         self.hidden_states[:num_tokens] = target_hidden_states
+        if self.is_multimodal_model:
+            input_ids = self.input_ids[:num_tokens]
+            inputs_embeds = self.model.get_input_embeddings(
+                input_ids,
+                multimodal_embeddings=mm_embeds or None,
+            )
+            self.inputs_embeds[:num_tokens] = inputs_embeds
+            inputs_embeds = self.inputs_embeds[:num_input_tokens]
+            input_ids = None
+        else:
+            inputs_embeds = None
+            input_ids = self.input_ids[:num_input_tokens]
 
         with set_forward_context(per_layer_attn_metadata,
                                  self.vllm_config,
                                  num_tokens=num_input_tokens):
             ret_hidden_states = self.model(
-                self.input_ids[:num_input_tokens],
-                self.positions[:num_input_tokens],
-                self.hidden_states[:num_input_tokens],
+                input_ids=input_ids,
+                positions=self.positions[:num_input_tokens],
+                hidden_states=self.hidden_states[:num_input_tokens],
+                inputs_embeds=inputs_embeds,
             )
             if self.method == "deepseek_mtp":
                 last_hidden_states = ret_hidden_states
@@ -218,15 +242,24 @@ def propose(
             self.input_ids[:batch_size] = input_ids
             self.positions[:batch_size] = clamped_positions
             self.hidden_states[:batch_size] = hidden_states
+            if self.is_multimodal_model:
+                inputs_embeds = self.model.get_input_embeddings(input_ids)
+                self.inputs_embeds[:batch_size] = inputs_embeds
+                inputs_embeds = self.inputs_embeds[:input_batch_size]
+                input_ids = None
+            else:
+                inputs_embeds = None
+                input_ids = self.input_ids[:input_batch_size]
 
             # Run the model.
             with set_forward_context(per_layer_attn_metadata,
                                      self.vllm_config,
                                      num_tokens=input_batch_size):
                 last_hidden_states, hidden_states = self.model(
-                    self.input_ids[:input_batch_size],
-                    self.positions[:input_batch_size],
-                    self.hidden_states[:input_batch_size],
+                    input_ids=input_ids,
+                    positions=self.positions[:input_batch_size],
+                    hidden_states=self.hidden_states[:input_batch_size],
+                    inputs_embeds=inputs_embeds,
                 )
             hidden_states = hidden_states[:batch_size]
             logits = self.model.compute_logits(last_hidden_states[:batch_size],
@@ -391,10 +424,18 @@ def dummy_run(
     ) -> None:
         with set_forward_context(None, self.vllm_config,
                                  num_tokens=num_tokens):
+            if self.is_multimodal_model:
+                input_ids = None
+                inputs_embeds = self.inputs_embeds[:num_tokens]
+            else:
+                input_ids = self.input_ids[:num_tokens]
+                inputs_embeds = None
+
             self.model(
-                self.input_ids[:num_tokens],
-                self.positions[:num_tokens],
-                self.hidden_states[:num_tokens],
+                input_ids=input_ids,
+                positions=self.positions[:num_tokens],
+                hidden_states=self.hidden_states[:num_tokens],
+                inputs_embeds=inputs_embeds,
             )
 
     def validate_same_kv_cache_group(self,
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 987ef22a1b7f..29cda4d837bf 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -1205,13 +1205,15 @@ def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
     def _gather_mm_embeddings(
         self,
         scheduler_output: "SchedulerOutput",
+        shift_computed_tokens: int = 0,
     ) -> list[torch.Tensor]:
         mm_embeds: list[torch.Tensor] = []
         for req_id in self.input_batch.req_ids:
             num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                 req_id]
             req_state = self.requests[req_id]
-            num_computed_tokens = req_state.num_computed_tokens
+            num_computed_tokens = \
+                req_state.num_computed_tokens + shift_computed_tokens
             mm_positions = req_state.mm_positions
             for i, pos_info in enumerate(mm_positions):
                 start_pos = pos_info.offset
@@ -1858,6 +1860,11 @@ def propose_draft_token_ids(
                         [h[token_indices] for h in aux_hidden_states], dim=-1)
                 else:
                     target_hidden_states = hidden_states[token_indices]
+            mm_embeds = None
+            if self.is_multimodal_model:
+                mm_embeds = self._gather_mm_embeddings(scheduler_output,
+                                                       shift_computed_tokens=1)
+
             draft_token_ids = self.drafter.propose(
                 target_token_ids=target_token_ids,
                 target_positions=target_positions,
@@ -1865,6 +1872,7 @@ def propose_draft_token_ids(
                 next_token_ids=next_token_ids,
                 sampling_metadata=sampling_metadata,
                 common_attn_metadata=common_attn_metadata,
+                mm_embeds=mm_embeds,
             )
             spec_token_ids = draft_token_ids.tolist()
         return spec_token_ids
```
