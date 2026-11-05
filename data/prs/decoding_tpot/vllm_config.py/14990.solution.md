# PR #14990: [Feature] Enhance EAGLE Architecture with Proper RMS Norms (#14990)

# Issue

## Original PR description (captured source)

In this PR, we allow the EAGLE architecture to optionally use proper RMS normalizations similar to the DeepSeek MTP module. It has two main benefits.

1) It improves acceptance length by ~5% when pre-attention norm, the output norm, and parallel norms (highlighted in red in the diagram below, taken from the DeepSeek V3 technical report) are used during training. Through ablation studies, we found all these RMS norms contribute to the improvement.
<img width="381" alt="Screenshot 2025-03-17 at 5 31 23 PM" src="https://github.com/user-attachments/assets/12c5d2b1-4998-42a2-ad02-5c2ed59e9e77" />

2) It alleviates the approximated KV cache bug currently in the vLLM EAGLE implementation as it reduces the caused performance degradation from 15% to 9% https://github.com/vllm-project/vllm/issues/14649 . Essentially, these additional RMS norms make sure the approximated KV cache is not too off, since the hidden states, from which those KV cache are computed, are normalized. The following table show the acceptance length for Llama 3 8B on GSM8K using this [example script](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/eagle.py).

| Number of Speculated Tokens | 1    | 2    | 3    | 4    | 5    |
|----------------------------|------|------|------|------|------|
| EAGLE Repo                | 1.84 | 2.45 | 2.83 | 3.07 | 3.17 |
| vLLM                      | 1.84 | 2.37 | 2.65 | 2.84 | 2.89 |
| Acceptance Length Drop      | 0%   | 3%   | 6%   | 7%   | 9%   |

With these normalizations, the performance degradation due to the approximated KV cache is now 0 ~ 9%, compared to the 8% ~ 15% drop without these normalizations. However, I want to emphasize that I do not intend to use this PR as a fix of https://github.com/vllm-project/vllm/issues/14649, but rather to show the community how to alleviate this bug temporarily while the bug is being fixed. 

cc @LiuXiaoxuanPKU Would appreciate your review. Thanks!



## Full discussion

See [14990.md](./14990.md) and [14990.json](./14990.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/14990
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/14990.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `589ed68a32ba7b6871d3be4d9da7da8cc81c982d0f81eb152882d1c2c570ee2c`

Existing semantic notes in [14990.sem.json](./14990.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/config.py`
- `vllm/model_executor/models/eagle.py`
- `vllm/transformers_utils/configs/eagle.py`

### Diff

```diff
diff --git a/vllm/config.py b/vllm/config.py
index 70cc0affe998..53ce603c6996 100644
--- a/vllm/config.py
+++ b/vllm/config.py
@@ -803,10 +803,18 @@ def get_hidden_size(self) -> int:
 
     @property
     def is_deepseek_mla(self) -> bool:
-        return (hasattr(self.hf_text_config, "model_type")) \
-                and (self.hf_text_config.model_type in \
-                    ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp'))\
-                and (self.hf_text_config.kv_lora_rank is not None)
+        if not hasattr(self.hf_text_config, "model_type"):
+            return False
+        elif self.hf_text_config.model_type in \
+            ('deepseek_v2', 'deepseek_v3', 'deepseek_mtp'):
+            return self.hf_text_config.kv_lora_rank is not None
+        elif self.hf_text_config.model_type == 'eagle':
+            # if the model is an EAGLE module, check for the
+            # underlying architecture
+            return self.hf_text_config.model.model_type in \
+                    ('deepseek_v2', 'deepseek_v3') \
+                and self.hf_text_config.kv_lora_rank is not None
+        return False
 
     def get_head_size(self) -> int:
         # TODO remove hard code
diff --git a/vllm/model_executor/models/eagle.py b/vllm/model_executor/models/eagle.py
index 010e51a3b9f2..3e4a5040b7c8 100644
--- a/vllm/model_executor/models/eagle.py
+++ b/vllm/model_executor/models/eagle.py
@@ -7,6 +7,7 @@
 
 from vllm.config import VllmConfig
 from vllm.logger import init_logger
+from vllm.model_executor.layers.layernorm import RMSNorm
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.sampler import SamplerOutput
 from vllm.model_executor.layers.vocab_parallel_embedding import (
@@ -59,7 +60,15 @@ class EAGLE(nn.Module):
        truncated_vocab_size < vocab_size. To use this technique, one has to find
        the top-k most frequent tokens in target dataset and add that as a tensor
        in the draft checkpoint (using key token_map). Also, the draft config
-       needs to have truncated_vocab_size (=k) as an attribute."""
+       needs to have truncated_vocab_size (=k) as an attribute.
+    4. We allow an enhanced EAGLE architecture similar to the DeepSeek MTP 
+       module with regards to the use of additional RMS norms. The original 
+       EAGLE architecture 1) skips the pre-attention norm in its first 
+       transformer block, and 2) skips the final output norm, both of which we 
+       found to be suboptimal. We also add the support for separate norms
+       applying to both the token embedding and hidden states before projection
+       as in DeepSeek MTP, which we found to improve performance as well.
+    """
 
     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         super().__init__()
@@ -81,9 +90,22 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         # While weights and biases are generally not needed,
         # they are retained here to support certain unit tests
         # (e.g., spec_decode/e2e/test_eagle_correctness.py).
-        self.model.model.layers[0].input_layernorm = DummyInputLayerNorm(
-            weight=self.model.model.layers[0].input_layernorm.weight)
-        self.model.model.norm = DummyOutputNorm()
+        if not hasattr(self.config.model,
+                       "skip_prenorm") or self.config.model.skip_prenorm:
+            self.model.model.layers[0].input_layernorm = DummyInputLayerNorm(
+                weight=self.model.model.layers[0].input_layernorm.weight)
+
+        if not hasattr(
+                self.config.model,
+                "skip_output_norm") or self.config.model.skip_output_norm:
+            self.model.model.norm = DummyOutputNorm()
+
+        self.add_para_norm = False
+        if hasattr(self.config.model,
+                   "add_para_norm") and self.config.model.add_para_norm:
+            self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+            self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+            self.add_para_norm = True
 
         self.orig_vocab_size = config.vocab_size
         self.truncated_vocab_size = config.truncated_vocab_size
@@ -128,8 +150,17 @@ def forward(
         if inputs_embeds is None:
             inputs_embeds = self.get_input_embeddings(input_ids)
 
-        inputs_embeds = self.fc(
-            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
+        if self.add_para_norm:
+            inputs_embeds = torch.cat([
+                self.enorm(inputs_embeds),
+                self.hnorm(previous_hidden_states)
+            ],
+                                      dim=-1)
+        else:
+            inputs_embeds = torch.cat([inputs_embeds, previous_hidden_states],
+                                      dim=-1)
+
+        inputs_embeds = self.fc(inputs_embeds)
 
         inputs_embeds[positions == 0] = 0  # masking inputs at position=0
 
@@ -190,6 +221,14 @@ def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
                 else:
                     logger.warning_once("Found bias in the loaded weights but "
                                         "the model config doesn't have bias.")
+            elif name.startswith("enorm.weight"):
+                weight_loader = getattr(self.enorm.weight, "weight_loader",
+                                        default_weight_loader)
+                weight_loader(self.enorm.weight, loaded_weight)
+            elif name.startswith("hnorm.weight"):
+                weight_loader = getattr(self.hnorm.weight, "weight_loader",
+                                        default_weight_loader)
+                weight_loader(self.hnorm.weight, loaded_weight)
             elif name.startswith("model.lm_head.") or name.startswith(
                     "model.model."):
                 model_weights[name.split("model.", 1)[-1]] = loaded_weight
diff --git a/vllm/transformers_utils/configs/eagle.py b/vllm/transformers_utils/configs/eagle.py
index b26aba66699f..dd806061ff58 100644
--- a/vllm/transformers_utils/configs/eagle.py
+++ b/vllm/transformers_utils/configs/eagle.py
@@ -5,6 +5,8 @@
 
 from transformers import AutoConfig, PretrainedConfig
 
+from vllm.transformers_utils.configs.deepseek_vl2 import DeepseekV2Config
+
 
 class EAGLEConfig(PretrainedConfig):
     model_type = "eagle"
@@ -14,8 +16,17 @@ def __init__(self,
                  truncated_vocab_size: Optional[int] = None,
                  **kwargs):
 
-        model_config = None if model is None else (AutoConfig.for_model(
-            **model) if isinstance(model, dict) else model)
+        model_config: Union[PretrainedConfig, DeepseekV2Config, None]
+        if isinstance(model, dict):
+            archs = model.get("architectures", [])
+            target_archs = ["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"]
+            if any(target_arch in archs for target_arch in target_archs):
+                # AutoConfig does not support DeepSeek MoE models yet
+                model_config = DeepseekV2Config(**model)
+            else:
+                model_config = AutoConfig.for_model(**model)
+        else:
+            model_config = model
 
         for k, v in kwargs.items():
             if k != "architectures" and k != "model_type" and hasattr(
```
