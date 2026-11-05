# PR #24243: Eagle3 that supports the Minicpm3 model (#24243)

# Issue

## Original PR description (captured source)

## Purpose
Eagle3 that supports the Minicpm3 model
## Test Plan
————————————————
200 samples, 10 concurrency, single RTX 4090:​​
Total output tokens: 106,678
Number of draft proposals: 43,221
Number of draft tokens generated: 259,326
Number of accepted tokens: 63,669
Mean acceptance length: 2.47

Acceptance rate at token 0: 0.58
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.22
Acceptance rate at token 3: 0.15
Acceptance rate at token 4: 0.11

Baseline (non-speculative): 215.84 s
Speculative decoding model: 133.24 s
Speedup: 1.62×

​20 samples, single concurrency, single RTX 4090:​​
Total output tokens: 21,690
Number of draft proposals: 8,914
Number of draft tokens generated: 53,484
Number of accepted tokens: 12,814
Mean acceptance length: 2.44

Acceptance rate at token 0: 0.57
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.21
Acceptance rate at token 3: 0.14
Acceptance rate at token 4: 0.10

Baseline (non-speculative): 382.29 s
Speculative decoding model: 224.20 s
Speedup: 1.71×
Average acceptance length: 2.44

​100 samples, 5 concurrency, single RTX 4090:​​
Total output tokens: 103,920
Number of draft proposals: 42,227
Number of draft tokens generated: 253,362
Number of accepted tokens: 61,896
Mean acceptance length: 2.47

Acceptance rate at token 0: 0.58
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.22
Acceptance rate at token 3: 0.15
Acceptance rate at token 4: 0.10

Baseline (non-speculative): 412.49 s
Speculative decoding model: 238.08 s
Speedup: 1.73×
Average acceptance length: 2.47
## Test Result

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [24243.md](./24243.md) and [24243.json](./24243.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/24243
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/24243.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `dd11d91b2a2d3e8871391f15bc73328b9f8e9f7694ab579001a89a1c9d1fe676`

Existing semantic notes in [24243.sem.json](./24243.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/config/speculative.py`
- `vllm/model_executor/models/minicpm.py`

### Diff

```diff
diff --git a/vllm/config/speculative.py b/vllm/config/speculative.py
index d533930e1c7a..fd7e73a5ae8b 100644
--- a/vllm/config/speculative.py
+++ b/vllm/config/speculative.py
@@ -527,7 +527,7 @@ def _verify_args(self) -> Self:
                              "speculative decoding is > 1, but got "
                              f"{self.disable_by_batch_size=}")
 
-        eagle3_target_supported = ["llama", "qwen", "gpt_oss"]
+        eagle3_target_supported = ["llama", "qwen", "minicpm", "gpt_oss"]
         if self.method == "eagle3" and self.target_model_config and not any(
                 supported_model in
                 self.target_model_config.hf_text_config.model_type
diff --git a/vllm/model_executor/models/minicpm.py b/vllm/model_executor/models/minicpm.py
index 0986ea07406a..55fe3e2ae3ae 100644
--- a/vllm/model_executor/models/minicpm.py
+++ b/vllm/model_executor/models/minicpm.py
@@ -55,7 +55,7 @@
 from vllm.platforms import current_platform
 from vllm.sequence import IntermediateTensors
 
-from .interfaces import SupportsLoRA, SupportsPP
+from .interfaces import SupportsEagle3, SupportsLoRA, SupportsPP
 from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                     make_empty_intermediate_tensors_factory, make_layers,
                     maybe_prefix)
@@ -381,6 +381,9 @@ def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         self.num_experts = getattr(self.config, "num_experts", 0)
         self._init_layers(prefix, config, cache_config, quant_config)
         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+
+        self.aux_hidden_state_layers = tuple[int, ...]()
+
         self.make_empty_intermediate_tensors = (
             make_empty_intermediate_tensors_factory(
                 ["hidden_states", "residual"], self.config.hidden_size))
@@ -408,7 +411,8 @@ def forward(
         positions: torch.Tensor,
         intermediate_tensors: Optional[IntermediateTensors] = None,
         inputs_embeds: Optional[torch.Tensor] = None,
-    ) -> Union[torch.Tensor, IntermediateTensors]:
+    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor,
+                                                        list[torch.Tensor]]]:
         if get_pp_group().is_first_rank:
             if inputs_embeds is not None:
                 hidden_states = inputs_embeds
@@ -419,18 +423,29 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in islice(self.layers, self.start_layer, self.end_layer):
+        aux_hidden_states = []
+        for idx, layer in enumerate(
+                islice(self.layers, self.start_layer, self.end_layer)):
+            if idx in self.aux_hidden_state_layers:
+                aux_hidden_states.append(
+                    hidden_states +
+                    residual if residual is not None else hidden_states)
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
                 residual,
             )
+
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
                 "hidden_states": hidden_states,
                 "residual": residual
             })
+
         hidden_states = self.norm(hidden_states)
+
+        if len(aux_hidden_states) > 0:
+            return hidden_states, aux_hidden_states
         return hidden_states
 
     def load_weights(self, weights: Iterable[tuple[str,
@@ -502,7 +517,7 @@ def load_weights(self, weights: Iterable[tuple[str,
         return loaded_params
 
 
-class MiniCPMForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
+class MiniCPMForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
     packed_modules_mapping = {
         "qkv_proj": [
             "q_proj",
@@ -568,16 +583,36 @@ def _init_model(self, *, vllm_config: VllmConfig, prefix: str = ""):
     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
         return self.model.get_input_embeddings(input_ids)
 
+    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        self.model.aux_hidden_state_layers = layers
+
+    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
+        num_layers = len(self.model.layers)
+        return (2, num_layers // 2, num_layers - 3)
+
     def forward(
         self,
         input_ids: torch.Tensor,
         positions: torch.Tensor,
         intermediate_tensors: Optional[IntermediateTensors] = None,
         inputs_embeds: Optional[torch.Tensor] = None,
-    ) -> Union[torch.Tensor, IntermediateTensors]:
-        hidden_states = self.model(input_ids, positions, intermediate_tensors,
-                                   inputs_embeds) / self.scale_width
-        return hidden_states
+    ) -> Union[torch.Tensor, IntermediateTensors, tuple[torch.Tensor,
+                                                        list[torch.Tensor]]]:
+        model_output = self.model(input_ids, positions, intermediate_tensors,
+                                  inputs_embeds)
+
+        if isinstance(model_output, tuple) and len(model_output) == 2:
+            # Aux hidden states are present.
+            hidden_states, aux_hidden_states = model_output
+            hidden_states = hidden_states / self.scale_width
+            return hidden_states, aux_hidden_states
+        else:
+            # Only hidden states or IntermediateTensors
+            if isinstance(model_output, IntermediateTensors):
+                return model_output
+            else:
+                hidden_states = model_output / self.scale_width
+                return hidden_states
 
     def compute_logits(
         self,
```
