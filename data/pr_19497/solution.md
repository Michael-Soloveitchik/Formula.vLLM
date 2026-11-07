# PR #19497: [Models] Improve iteration over layers

- field: quantization  | area: model_executor_models
- author: lgeiger
- state: MERGED  created: 2025-06-11T15:46:15Z  merged: 2025-08-29T01:26:35Z
- url: https://github.com/vllm-project/vllm/pull/19497
- labels: ready, llama, qwen, deepseek

# Issue

## Problem

Model `forward()` methods iterate over pipeline-parallel layer subsets by slicing a `torch.nn.ModuleList` via `self.layers[self.start_layer:self.end_layer]`. Slicing a `ModuleList` constructs a brand-new `ModuleList` object each call. This allocation is cheap but not free, and it runs synchronously at the start of `forward()` so it cannot overlap with GPU compute.

## Example

In eager mode, a profile shows the slice op appearing as measurable CPU work at the top of each decoder stack `forward()` before any kernel launches, blocking the start of GPU execution for that step.

## Use case

Standard autoregressive decode/prefill through a transformer decoder stack, especially under pipeline parallelism where `start_layer`/`end_layer` carve out this rank's layer range.

## Resources

CPU time and kernel-launch/dispatch latency on the critical path; the new-ModuleList construction is serial CPU work that delays GPU compute instead of overlapping with it.

## Type of users

All inference users, most visibly those in eager mode and pipeline-parallel (PP) multi-GPU setups where the per-forward overhead repeats every step across many models.

## Kind of inputs

Any request; overhead is per-forward-call and independent of sequence length or batch size, so it recurs on every decode step and prefill.

## Mechanism with the issue

`nn.ModuleList.__getitem__` with a slice builds and returns a new `ModuleList` (copying child module refs) rather than returning a lazy view. Because this executes at the head of `Module.forward()`, it adds CPU latency that is not overlapped with GPU computation. Per PyTorch source, plain iteration over a `ModuleList` does not construct a new list.

# Solution

## Fix in words

Replaces `self.layers[self.start_layer:self.end_layer]` (and equivalent `for i in range(start, end): layer = self.layers[i]` patterns) with `islice(self.layers, self.start_layer, self.end_layer)` across many model files, adding `from itertools import islice` to each. `islice` iterates the existing `ModuleList` lazily without allocating a new one, eliminating the per-forward construction. A couple of spots that indexed via `range(len(self.layers))` are also simplified to `enumerate(self.layers)`. Effect: removes the non-overlappable CPU allocation at the start of `forward()`, allowing layer iteration to overlap with GPU compute.

## Author's solution

Author changed layer iteration from slicing the `ModuleList` to `itertools.islice`, since iteration does not create a new `ModuleList` and can overlap with GPU computation. Reports a very minor but positive effect: 0-0.9% faster time to first token and improved (noisy) throughput, with no readability cost, covered by existing unit tests.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/arcee.py`
- `vllm/model_executor/models/arctic.py`
- `vllm/model_executor/models/baichuan.py`
- `vllm/model_executor/models/bailing_moe.py`
- `vllm/model_executor/models/bamba.py`
- `vllm/model_executor/models/bloom.py`
- `vllm/model_executor/models/chameleon.py`
- `vllm/model_executor/models/chatglm.py`
- `vllm/model_executor/models/commandr.py`
- `vllm/model_executor/models/dbrx.py`
- `vllm/model_executor/models/deepseek.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `vllm/model_executor/models/dots1.py`
- `vllm/model_executor/models/ernie45_moe.py`
- `vllm/model_executor/models/ernie45_vl_moe.py`
- `vllm/model_executor/models/exaone.py`
- `vllm/model_executor/models/exaone4.py`
- `vllm/model_executor/models/falcon.py`
- `vllm/model_executor/models/gemma.py`
- `vllm/model_executor/models/gemma2.py`
- `vllm/model_executor/models/gemma3.py`
- `vllm/model_executor/models/glm4_moe.py`
- `vllm/model_executor/models/gpt2.py`
- `vllm/model_executor/models/gpt_bigcode.py`
- `vllm/model_executor/models/gpt_j.py`
- `vllm/model_executor/models/gpt_neox.py`
- `vllm/model_executor/models/granite.py`
- `vllm/model_executor/models/granitemoe.py`
- `vllm/model_executor/models/granitemoehybrid.py`
- `vllm/model_executor/models/granitemoeshared.py`
- `vllm/model_executor/models/grok1.py`
- `vllm/model_executor/models/internlm2.py`
- `vllm/model_executor/models/internlm2_ve.py`
- `vllm/model_executor/models/jais.py`
- `vllm/model_executor/models/jamba.py`
- `vllm/model_executor/models/lfm2.py`
- `vllm/model_executor/models/llama.py`
- `vllm/model_executor/models/mamba2.py`
- `vllm/model_executor/models/mimo.py`
- `vllm/model_executor/models/minicpm.py`
- `vllm/model_executor/models/minimax_text_01.py`
- `vllm/model_executor/models/mixtral.py`
- `vllm/model_executor/models/mixtral_quant.py`
- `vllm/model_executor/models/molmo.py`
- `vllm/model_executor/models/mpt.py`
- `vllm/model_executor/models/nemotron.py`
- `vllm/model_executor/models/nemotron_h.py`
- `vllm/model_executor/models/nemotron_nas.py`
- `vllm/model_executor/models/olmo.py`
- `vllm/model_executor/models/olmo2.py`
- `vllm/model_executor/models/olmoe.py`
- `vllm/model_executor/models/opt.py`
- `vllm/model_executor/models/orion.py`
- `vllm/model_executor/models/persimmon.py`
- `vllm/model_executor/models/phi.py`
- `vllm/model_executor/models/phimoe.py`
- `vllm/model_executor/models/plamo2.py`
- `vllm/model_executor/models/qwen.py`
- `vllm/model_executor/models/qwen2.py`
- `vllm/model_executor/models/qwen2_moe.py`
- `vllm/model_executor/models/qwen3_moe.py`
- `vllm/model_executor/models/seed_oss.py`
- `vllm/model_executor/models/stablelm.py`
- `vllm/model_executor/models/starcoder2.py`
- `vllm/model_executor/models/step3_text.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/arcee.py b/vllm/model_executor/models/arcee.py
index 4cf73e2e0ea5..13ed4da0602a 100644
--- a/vllm/model_executor/models/arcee.py
+++ b/vllm/model_executor/models/arcee.py
@@ -9,6 +9,7 @@
 # activation.
 
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -243,7 +244,7 @@ def forward(
 
         aux_hidden_states: list[torch.Tensor] = []
         for idx, layer in enumerate(
-                self.layers[self.start_layer:self.end_layer]):
+                islice(self.layers, self.start_layer, self.end_layer)):
             if idx in self.aux_hidden_state_layers:
                 aux_hidden_states.append(
                     hidden_states +
diff --git a/vllm/model_executor/models/arctic.py b/vllm/model_executor/models/arctic.py
index 4693c9487a8b..c566611266af 100644
--- a/vllm/model_executor/models/arctic.py
+++ b/vllm/model_executor/models/arctic.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Inference-only Snowflake Arctic model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -403,7 +404,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/baichuan.py b/vllm/model_executor/models/baichuan.py
index 804a2f1785d5..4563c356666a 100644
--- a/vllm/model_executor/models/baichuan.py
+++ b/vllm/model_executor/models/baichuan.py
@@ -22,6 +22,7 @@
 """Inference-only BaiChuan model compatible with HuggingFace weights."""
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -309,7 +310,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/bailing_moe.py b/vllm/model_executor/models/bailing_moe.py
index 23cab3509ca8..a42640cef9d4 100644
--- a/vllm/model_executor/models/bailing_moe.py
+++ b/vllm/model_executor/models/bailing_moe.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only BailingMoE model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -359,8 +360,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 hidden_states,
                 position_ids,
diff --git a/vllm/model_executor/models/bamba.py b/vllm/model_executor/models/bamba.py
index e2cd31af5390..a72bbdebe531 100644
--- a/vllm/model_executor/models/bamba.py
+++ b/vllm/model_executor/models/bamba.py
@@ -345,8 +345,7 @@ def forward(
 
         residual = None
         num_attn = 0
-        for i in range(len(self.layers)):
-            layer = self.layers[i]
+        for i, layer in enumerate(self.layers):
             if isinstance(layer, BambaAttentionDecoderLayer):
                 num_attn += 1
 
diff --git a/vllm/model_executor/models/bloom.py b/vllm/model_executor/models/bloom.py
index 126404584892..13ecda0122be 100644
--- a/vllm/model_executor/models/bloom.py
+++ b/vllm/model_executor/models/bloom.py
@@ -20,6 +20,7 @@
 """Inference-only BLOOM model compatible with HuggingFace weights."""
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -273,7 +274,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(position_ids, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/chameleon.py b/vllm/model_executor/models/chameleon.py
index e6914ad4c495..28a1a66c2329 100644
--- a/vllm/model_executor/models/chameleon.py
+++ b/vllm/model_executor/models/chameleon.py
@@ -3,6 +3,7 @@
 
 from collections.abc import Iterable, Mapping, Sequence
 from functools import cached_property
+from itertools import islice
 from typing import Annotated, Any, Literal, Optional, Union
 
 import torch
@@ -914,7 +915,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/chatglm.py b/vllm/model_executor/models/chatglm.py
index 5470ff3e8b61..1fc2da3e4d7c 100644
--- a/vllm/model_executor/models/chatglm.py
+++ b/vllm/model_executor/models/chatglm.py
@@ -5,6 +5,7 @@
 """Inference-only ChatGLM model compatible with THUDM weights."""
 import json
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -281,7 +282,7 @@ def forward(
         hidden_states: torch.Tensor,
         position_ids: torch.Tensor,
     ) -> Union[torch.Tensor, IntermediateTensors]:
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(hidden_states=hidden_states,
                                   position_ids=position_ids)
 
diff --git a/vllm/model_executor/models/commandr.py b/vllm/model_executor/models/commandr.py
index 4dd84b8f8fdd..7f87e31abdcd 100644
--- a/vllm/model_executor/models/commandr.py
+++ b/vllm/model_executor/models/commandr.py
@@ -23,6 +23,7 @@
 # This file is based on the LLama model definition file in transformers
 """PyTorch Cohere model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -322,7 +323,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/dbrx.py b/vllm/model_executor/models/dbrx.py
index e74d90e0b1d7..519cd522213b 100644
--- a/vllm/model_executor/models/dbrx.py
+++ b/vllm/model_executor/models/dbrx.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -359,7 +360,7 @@ def forward(
         else:
             assert intermediate_tensors
             hidden_states = intermediate_tensors["hidden_states"]
-        for block in self.blocks[self.start_layer:self.end_layer]:
+        for block in islice(self.blocks, self.start_layer, self.end_layer):
             hidden_states = block(position_ids, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/deepseek.py b/vllm/model_executor/models/deepseek.py
index 2f0202f1e038..e815f13d66dc 100644
--- a/vllm/model_executor/models/deepseek.py
+++ b/vllm/model_executor/models/deepseek.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only Deepseek model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -377,7 +378,7 @@ def forward(
         else:
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
@@ -483,4 +484,4 @@ def compute_logits(
     def load_weights(self, weights: Iterable[tuple[str,
                                                    torch.Tensor]]) -> set[str]:
         loader = AutoWeightsLoader(self)
-        return loader.load_weights(weights)
\ No newline at end of file
+        return loader.load_weights(weights)
diff --git a/vllm/model_executor/models/deepseek_v2.py b/vllm/model_executor/models/deepseek_v2.py
index 7657e7cb003d..ed033954f7c0 100644
--- a/vllm/model_executor/models/deepseek_v2.py
+++ b/vllm/model_executor/models/deepseek_v2.py
@@ -25,6 +25,7 @@
 """Inference-only DeepseekV2/DeepseekV3 model."""
 import typing
 from collections.abc import Callable, Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -712,7 +713,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/dots1.py b/vllm/model_executor/models/dots1.py
index 5f410c0ae5fb..c386f8db9eec 100644
--- a/vllm/model_executor/models/dots1.py
+++ b/vllm/model_executor/models/dots1.py
@@ -25,6 +25,7 @@
 # limitations under the License.
 """Inference-only dots1 model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -391,7 +392,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/ernie45_moe.py b/vllm/model_executor/models/ernie45_moe.py
index 4780ea931ea5..33ec27fc630e 100644
--- a/vllm/model_executor/models/ernie45_moe.py
+++ b/vllm/model_executor/models/ernie45_moe.py
@@ -23,6 +23,7 @@
 # limitations under the License.
 """Inference-only ErineMoE model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -419,8 +420,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/ernie45_vl_moe.py b/vllm/model_executor/models/ernie45_vl_moe.py
index f56c09843515..780974c3b758 100644
--- a/vllm/model_executor/models/ernie45_vl_moe.py
+++ b/vllm/model_executor/models/ernie45_vl_moe.py
@@ -23,6 +23,7 @@
 # limitations under the License.
 """Inference-only Erine VL model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -508,8 +509,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual,
                                             visual_token_mask, **kwargs)
 
diff --git a/vllm/model_executor/models/exaone.py b/vllm/model_executor/models/exaone.py
index 8052b6bb8234..942db0143a45 100644
--- a/vllm/model_executor/models/exaone.py
+++ b/vllm/model_executor/models/exaone.py
@@ -26,6 +26,7 @@
 """Inference-only Exaone model compatible with HuggingFace weights."""
 
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -371,7 +372,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/exaone4.py b/vllm/model_executor/models/exaone4.py
index 827e9014184b..971fcbd2aa27 100644
--- a/vllm/model_executor/models/exaone4.py
+++ b/vllm/model_executor/models/exaone4.py
@@ -22,6 +22,7 @@
 """Inference-only Exaone model compatible with HuggingFace weights."""
 
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -354,7 +355,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/falcon.py b/vllm/model_executor/models/falcon.py
index 62a93dabd5d7..a9fe0924babd 100644
--- a/vllm/model_executor/models/falcon.py
+++ b/vllm/model_executor/models/falcon.py
@@ -22,6 +22,7 @@
 
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -389,7 +390,7 @@ def forward(
                 hidden_states = self.get_input_embeddings(input_ids)
         else:
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/gemma.py b/vllm/model_executor/models/gemma.py
index 59c3102add4c..12eb27503870 100644
--- a/vllm/model_executor/models/gemma.py
+++ b/vllm/model_executor/models/gemma.py
@@ -18,6 +18,7 @@
 """Inference-only Gemma model compatible with HuggingFace weights."""
 from collections.abc import Iterable
 from functools import cache
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -308,7 +309,7 @@ def forward(
         else:
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/gemma2.py b/vllm/model_executor/models/gemma2.py
index 8cfe92c64540..0bdb6c6bf7ae 100644
--- a/vllm/model_executor/models/gemma2.py
+++ b/vllm/model_executor/models/gemma2.py
@@ -17,6 +17,7 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -292,7 +293,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/gemma3.py b/vllm/model_executor/models/gemma3.py
index b762be3c5292..410c715d5241 100644
--- a/vllm/model_executor/models/gemma3.py
+++ b/vllm/model_executor/models/gemma3.py
@@ -16,6 +16,7 @@
 # See the License for the specific language governing permissions and
 # limitations under the License.
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -398,7 +399,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/glm4_moe.py b/vllm/model_executor/models/glm4_moe.py
index fe5e46a99826..fcc63815ac56 100644
--- a/vllm/model_executor/models/glm4_moe.py
+++ b/vllm/model_executor/models/glm4_moe.py
@@ -24,6 +24,7 @@
 """Inference-only GLM-4.5 model compatible with HuggingFace weights."""
 import typing
 from collections.abc import Callable, Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -440,8 +441,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/gpt2.py b/vllm/model_executor/models/gpt2.py
index 98d76337395b..4446b5ab181c 100644
--- a/vllm/model_executor/models/gpt2.py
+++ b/vllm/model_executor/models/gpt2.py
@@ -20,6 +20,7 @@
 # limitations under the License.
 """Inference-only GPT-2 model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -228,7 +229,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
 
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/gpt_bigcode.py b/vllm/model_executor/models/gpt_bigcode.py
index 036ded530f97..d5c2604145ee 100644
--- a/vllm/model_executor/models/gpt_bigcode.py
+++ b/vllm/model_executor/models/gpt_bigcode.py
@@ -21,6 +21,7 @@
 # limitations under the License.
 """Inference-only GPTBigCode model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -246,7 +247,7 @@ def forward(
         else:
             hidden_states = intermediate_tensors["hidden_states"]
 
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/gpt_j.py b/vllm/model_executor/models/gpt_j.py
index bd162a5e57bc..584c7f5d8a2d 100644
--- a/vllm/model_executor/models/gpt_j.py
+++ b/vllm/model_executor/models/gpt_j.py
@@ -19,6 +19,7 @@
 # limitations under the License.
 """Inference-only GPT-J model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -223,7 +224,7 @@ def forward(
                 hidden_states = self.get_input_embeddings(input_ids)
         else:
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(position_ids, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
@@ -336,4 +337,4 @@ def compute_logits(
     def load_weights(self, weights: Iterable[tuple[str,
                                                    torch.Tensor]]) -> set[str]:
         loader = AutoWeightsLoader(self)
-        return loader.load_weights(weights)
\ No newline at end of file
+        return loader.load_weights(weights)
diff --git a/vllm/model_executor/models/gpt_neox.py b/vllm/model_executor/models/gpt_neox.py
index d418d8bb86ce..e97db188e27e 100644
--- a/vllm/model_executor/models/gpt_neox.py
+++ b/vllm/model_executor/models/gpt_neox.py
@@ -19,6 +19,7 @@
 # limitations under the License.
 """Inference-only GPT-NeoX model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -235,7 +236,7 @@ def forward(
                 hidden_states = self.get_input_embeddings(input_ids)
         else:
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(position_ids, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/granite.py b/vllm/model_executor/models/granite.py
index 507a9206c428..f8ba0229210a 100644
--- a/vllm/model_executor/models/granite.py
+++ b/vllm/model_executor/models/granite.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only IBM Granite model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -316,7 +317,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/granitemoe.py b/vllm/model_executor/models/granitemoe.py
index 7d31854dce8d..07ad75bcf166 100644
--- a/vllm/model_executor/models/granitemoe.py
+++ b/vllm/model_executor/models/granitemoe.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only GraniteMoe model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional
 
 import torch
@@ -303,7 +304,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/granitemoehybrid.py b/vllm/model_executor/models/granitemoehybrid.py
index f451e65338b7..79c6d8146ba9 100644
--- a/vllm/model_executor/models/granitemoehybrid.py
+++ b/vllm/model_executor/models/granitemoehybrid.py
@@ -397,8 +397,7 @@ def forward(
             residual = intermediate_tensors["residual"]
 
         num_attn = 0
-        for i in range(len(self.layers)):
-            layer = self.layers[i]
+        for i, layer in enumerate(self.layers):
             if isinstance(layer, GraniteMoeHybridAttentionDecoderLayer):
                 num_attn += 1
 
diff --git a/vllm/model_executor/models/granitemoeshared.py b/vllm/model_executor/models/granitemoeshared.py
index 1e2e8544179c..0b568a4b2268 100644
--- a/vllm/model_executor/models/granitemoeshared.py
+++ b/vllm/model_executor/models/granitemoeshared.py
@@ -6,6 +6,7 @@
 experts.
 """
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional
 
 import torch
@@ -200,8 +201,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/grok1.py b/vllm/model_executor/models/grok1.py
index 3659249cd8bd..a59113438337 100644
--- a/vllm/model_executor/models/grok1.py
+++ b/vllm/model_executor/models/grok1.py
@@ -23,6 +23,7 @@
 # limitations under the License.
 """Inference-only Grok1 model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -347,8 +348,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/internlm2.py b/vllm/model_executor/models/internlm2.py
index 26bc48ffbd9b..320e8d9d480c 100644
--- a/vllm/model_executor/models/internlm2.py
+++ b/vllm/model_executor/models/internlm2.py
@@ -3,6 +3,7 @@
 
 from collections.abc import Iterable
 from functools import partial
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -297,7 +298,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/internlm2_ve.py b/vllm/model_executor/models/internlm2_ve.py
index 4bbb49da0e96..d41ac2b70bc6 100644
--- a/vllm/model_executor/models/internlm2_ve.py
+++ b/vllm/model_executor/models/internlm2_ve.py
@@ -1,6 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -123,7 +124,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/jais.py b/vllm/model_executor/models/jais.py
index bed4a5dff2ef..91a06dd50247 100644
--- a/vllm/model_executor/models/jais.py
+++ b/vllm/model_executor/models/jais.py
@@ -23,6 +23,7 @@
 
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -276,7 +277,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
 
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states = layer(hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/jamba.py b/vllm/model_executor/models/jamba.py
index 3c1a0b68df56..aebd2cbe2e99 100644
--- a/vllm/model_executor/models/jamba.py
+++ b/vllm/model_executor/models/jamba.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Inference-only Jamba model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional
 
 import torch
@@ -350,7 +351,7 @@ def forward(
 
         kv_cache_index = 0
         mamba_cache_index = 0
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             layer_mamba_cache_params = None
             if isinstance(layer, JambaAttentionDecoderLayer):
                 kv_cache_index += 1
diff --git a/vllm/model_executor/models/lfm2.py b/vllm/model_executor/models/lfm2.py
index 5f3148b47ead..927f78c4e4b4 100644
--- a/vllm/model_executor/models/lfm2.py
+++ b/vllm/model_executor/models/lfm2.py
@@ -1,6 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional
 
 import torch
@@ -374,7 +375,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions=positions,
                 hidden_states=hidden_states,
@@ -554,4 +555,4 @@ def load_weights(self, weights: Iterable[tuple[str,
             skip_prefixes=(["lm_head."]
                            if self.config.tie_word_embeddings else None),
         )
-        return loader.load_weights(weights)
\ No newline at end of file
+        return loader.load_weights(weights)
diff --git a/vllm/model_executor/models/llama.py b/vllm/model_executor/models/llama.py
index e39a6df843cd..a22bde194f5d 100644
--- a/vllm/model_executor/models/llama.py
+++ b/vllm/model_executor/models/llama.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only LLaMA model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -383,7 +384,7 @@ def forward(
 
         aux_hidden_states = []
         for idx, layer in enumerate(
-                self.layers[self.start_layer:self.end_layer]):
+                islice(self.layers, self.start_layer, self.end_layer)):
             if idx in self.aux_hidden_state_layers:
                 aux_hidden_states.append(hidden_states + residual)
             hidden_states, residual = layer(positions, hidden_states, residual)
diff --git a/vllm/model_executor/models/mamba2.py b/vllm/model_executor/models/mamba2.py
index 3432cf29feac..81b9a125380a 100644
--- a/vllm/model_executor/models/mamba2.py
+++ b/vllm/model_executor/models/mamba2.py
@@ -164,9 +164,7 @@ def forward(
             # v1 get mamba2_metadata from forward_context
             mamba2_metadata = None
 
-        for i in range(len(self.layers)):
-            layer = self.layers[i]
-
+        for i, layer in enumerate(self.layers):
             hidden_states, residual = layer(
                 positions=positions,
                 hidden_states=hidden_states,
diff --git a/vllm/model_executor/models/mimo.py b/vllm/model_executor/models/mimo.py
index 5b497dd9d89f..ea5292d0df20 100644
--- a/vllm/model_executor/models/mimo.py
+++ b/vllm/model_executor/models/mimo.py
@@ -26,6 +26,7 @@
 # limitations under the License.
 """Inference-only MiMo model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -74,7 +75,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/minicpm.py b/vllm/model_executor/models/minicpm.py
index d398a5d12bbc..5632f8c8cc4f 100644
--- a/vllm/model_executor/models/minicpm.py
+++ b/vllm/model_executor/models/minicpm.py
@@ -25,6 +25,7 @@
 """Inference-only MiniCPM model compatible with HuggingFace weights."""
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -414,7 +415,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/minimax_text_01.py b/vllm/model_executor/models/minimax_text_01.py
index 176a40179bca..93ef13d5d16a 100644
--- a/vllm/model_executor/models/minimax_text_01.py
+++ b/vllm/model_executor/models/minimax_text_01.py
@@ -3,6 +3,7 @@
 """Inference-only MiniMaxText01 model."""
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import TYPE_CHECKING, Optional, Union
 
 if TYPE_CHECKING:
@@ -1019,8 +1020,7 @@ def forward(self,
 
         minimax_cache_index = 0
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             _caches = None
             if not envs.VLLM_USE_V1 and isinstance(
                     layer.self_attn, MiniMaxText01LinearAttention):
diff --git a/vllm/model_executor/models/mixtral.py b/vllm/model_executor/models/mixtral.py
index 30de83da49e0..52fcbbfc58be 100644
--- a/vllm/model_executor/models/mixtral.py
+++ b/vllm/model_executor/models/mixtral.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only Mixtral model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -307,7 +308,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/mixtral_quant.py b/vllm/model_executor/models/mixtral_quant.py
index c8ad358c622d..692267b4d727 100644
--- a/vllm/model_executor/models/mixtral_quant.py
+++ b/vllm/model_executor/models/mixtral_quant.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only Mixtral model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import numpy as np
@@ -346,7 +347,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/molmo.py b/vllm/model_executor/models/molmo.py
index 5fc28ed0e493..b2fc7be1af22 100644
--- a/vllm/model_executor/models/molmo.py
+++ b/vllm/model_executor/models/molmo.py
@@ -5,6 +5,7 @@
 from collections.abc import Iterable, Mapping, Sequence
 from dataclasses import dataclass
 from functools import cached_property, partial
+from itertools import islice
 from typing import Annotated, Optional, Union
 
 import numpy as np
@@ -842,7 +843,7 @@ def forward(
             residual = intermediate_tensors["residual"]
 
         # Apply blocks one-by-one.
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/mpt.py b/vllm/model_executor/models/mpt.py
index 8db52a69924c..48ac91fa6dde 100644
--- a/vllm/model_executor/models/mpt.py
+++ b/vllm/model_executor/models/mpt.py
@@ -4,6 +4,7 @@
 # Adapted from https://huggingface.co/mosaicml/mpt-7b/tree/main
 import math
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -260,7 +261,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
 
-        for block in self.blocks[self.start_layer:self.end_layer]:
+        for block in islice(self.blocks, self.start_layer, self.end_layer):
             hidden_states = block(position_ids, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/nemotron.py b/vllm/model_executor/models/nemotron.py
index eabf47b1aede..10adc62d3de3 100644
--- a/vllm/model_executor/models/nemotron.py
+++ b/vllm/model_executor/models/nemotron.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only Nemotron model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -353,7 +354,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/nemotron_h.py b/vllm/model_executor/models/nemotron_h.py
index 07cd5a4c6e24..8a563288cb4d 100644
--- a/vllm/model_executor/models/nemotron_h.py
+++ b/vllm/model_executor/models/nemotron_h.py
@@ -399,8 +399,7 @@ def forward(
 
         residual = None
         num_non_mamba_layers = 0
-        for i in range(len(self.layers)):
-            layer = self.layers[i]
+        for i, layer in enumerate(self.layers):
             layer_mamba_cache_params = None
             if isinstance(layer,
                           NemotronHMambaDecoderLayer) and mamba_cache_params:
diff --git a/vllm/model_executor/models/nemotron_nas.py b/vllm/model_executor/models/nemotron_nas.py
index a766ed9476a6..f8e38dcd80b5 100644
--- a/vllm/model_executor/models/nemotron_nas.py
+++ b/vllm/model_executor/models/nemotron_nas.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only deci model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -287,8 +288,7 @@ def forward(
             residual = intermediate_tensors["residual"]
 
         kv_cache_index = 0
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             if not layer._is_no_op_attention:
                 hidden_states, residual = layer(positions, hidden_states,
                                                 residual)
diff --git a/vllm/model_executor/models/olmo.py b/vllm/model_executor/models/olmo.py
index 01639d398126..71575989565a 100644
--- a/vllm/model_executor/models/olmo.py
+++ b/vllm/model_executor/models/olmo.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only OLMo model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -280,7 +281,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
 
         # Apply blocks one-by-one.
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             # shape: (batch_size, seq_len, d_model)
             hidden_states = layer(positions, hidden_states)
 
diff --git a/vllm/model_executor/models/olmo2.py b/vllm/model_executor/models/olmo2.py
index 66a0f9115585..bccd1b87043a 100644
--- a/vllm/model_executor/models/olmo2.py
+++ b/vllm/model_executor/models/olmo2.py
@@ -26,6 +26,7 @@
 
 from collections.abc import Iterable
 from functools import partial
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -305,7 +306,7 @@ def forward(
             assert isinstance(hidden_states, torch.Tensor)
 
         # Apply blocks one-by-one.
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             # shape: (batch_size, seq_len, d_model)
             hidden_states = layer(positions, hidden_states)
 
diff --git a/vllm/model_executor/models/olmoe.py b/vllm/model_executor/models/olmoe.py
index a47c3bd41645..9b8525bfadec 100644
--- a/vllm/model_executor/models/olmoe.py
+++ b/vllm/model_executor/models/olmoe.py
@@ -15,6 +15,7 @@
 """Inference-only OLMoE model compatible with HuggingFace weights."""
 from collections.abc import Iterable
 from functools import partial
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -314,7 +315,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/opt.py b/vllm/model_executor/models/opt.py
index 9eaac1e28dcd..b92e586f0bf2 100644
--- a/vllm/model_executor/models/opt.py
+++ b/vllm/model_executor/models/opt.py
@@ -20,6 +20,7 @@
 # limitations under the License.
 """Inference-only OPT model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -269,7 +270,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/orion.py b/vllm/model_executor/models/orion.py
index d121188ba5d4..add751ebf09c 100644
--- a/vllm/model_executor/models/orion.py
+++ b/vllm/model_executor/models/orion.py
@@ -7,6 +7,7 @@
 # LICENSE: https://huggingface.co/OrionStarAI/Orion-14B-Base/blob/main/LICENSE
 """Inference-only Orion-14B model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -252,7 +253,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/persimmon.py b/vllm/model_executor/models/persimmon.py
index f8db99eb92ba..6bdd38d06880 100644
--- a/vllm/model_executor/models/persimmon.py
+++ b/vllm/model_executor/models/persimmon.py
@@ -23,6 +23,7 @@
 # limitations under the License.
 """Inference-only persimmon model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -255,7 +256,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/phi.py b/vllm/model_executor/models/phi.py
index 21d517b3a490..789b24eb0f6b 100644
--- a/vllm/model_executor/models/phi.py
+++ b/vllm/model_executor/models/phi.py
@@ -38,6 +38,7 @@
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 """Inference-only Phi-1.5 model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -240,7 +241,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
 
         if not get_pp_group().is_last_rank:
diff --git a/vllm/model_executor/models/phimoe.py b/vllm/model_executor/models/phimoe.py
index cfe0982204fa..15ae081a9f5f 100644
--- a/vllm/model_executor/models/phimoe.py
+++ b/vllm/model_executor/models/phimoe.py
@@ -24,6 +24,7 @@
 # limitations under the License.
 """Inference-only PhiMoE model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -500,7 +501,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/plamo2.py b/vllm/model_executor/models/plamo2.py
index e5034b536266..7f70e44b10a6 100644
--- a/vllm/model_executor/models/plamo2.py
+++ b/vllm/model_executor/models/plamo2.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Inference-only PLaMo2 model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional
 
 import torch
@@ -614,7 +615,7 @@ def forward(
         mamba2_metadata: Mamba2Metadata,
     ) -> torch.Tensor:
         mamba_cache_index = 0
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             layer_mamba_cache_params = None
             if layer.is_mamba:
                 layer_mamba_cache_params = mamba_cache_params.at_layer_idx(
diff --git a/vllm/model_executor/models/qwen.py b/vllm/model_executor/models/qwen.py
index e804f03e014e..e32dc51f00c0 100644
--- a/vllm/model_executor/models/qwen.py
+++ b/vllm/model_executor/models/qwen.py
@@ -8,6 +8,7 @@
 """Inference-only QWen model compatible with HuggingFace weights."""
 import json
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -234,7 +235,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for layer in self.h[self.start_layer:self.end_layer]:
+        for layer in islice(self.h, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/qwen2.py b/vllm/model_executor/models/qwen2.py
index 27c1e68c6704..54dc0bebd9c5 100644
--- a/vllm/model_executor/models/qwen2.py
+++ b/vllm/model_executor/models/qwen2.py
@@ -25,6 +25,7 @@
 # limitations under the License.
 """Inference-only Qwen2 model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -358,7 +359,7 @@ def forward(
 
         aux_hidden_states = []
         for idx, layer in enumerate(
-                self.layers[self.start_layer:self.end_layer]):
+                islice(self.layers, self.start_layer, self.end_layer)):
             if idx in self.aux_hidden_state_layers:
                 aux_hidden_states.append(hidden_states + residual)
             hidden_states, residual = layer(positions, hidden_states, residual)
diff --git a/vllm/model_executor/models/qwen2_moe.py b/vllm/model_executor/models/qwen2_moe.py
index 5c4ad34246d6..5551ad8c3232 100644
--- a/vllm/model_executor/models/qwen2_moe.py
+++ b/vllm/model_executor/models/qwen2_moe.py
@@ -25,6 +25,7 @@
 # limitations under the License.
 """Inference-only Qwen2MoE model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -381,7 +382,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/qwen3_moe.py b/vllm/model_executor/models/qwen3_moe.py
index 8498f61b35fd..94e6a66bea5c 100644
--- a/vllm/model_executor/models/qwen3_moe.py
+++ b/vllm/model_executor/models/qwen3_moe.py
@@ -24,6 +24,7 @@
 """Inference-only Qwen3MoE model compatible with HuggingFace weights."""
 import typing
 from collections.abc import Callable, Iterable
+from itertools import islice
 from typing import Any, Optional, Union
 
 import torch
@@ -420,8 +421,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({
diff --git a/vllm/model_executor/models/seed_oss.py b/vllm/model_executor/models/seed_oss.py
index 34a87a6a69a3..e3c7c700f8fa 100644
--- a/vllm/model_executor/models/seed_oss.py
+++ b/vllm/model_executor/models/seed_oss.py
@@ -23,6 +23,7 @@
 # limitations under the License.
 """Inference-only SeedOss model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -340,7 +341,7 @@ def forward(
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(
                 positions,
                 hidden_states,
diff --git a/vllm/model_executor/models/stablelm.py b/vllm/model_executor/models/stablelm.py
index d6ec743ce845..9e880ebd5081 100644
--- a/vllm/model_executor/models/stablelm.py
+++ b/vllm/model_executor/models/stablelm.py
@@ -22,6 +22,7 @@
 """Inference-only StabeLM (https://github.com/Stability-AI/StableLM)
 model compatible with HuggingFace weights."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -247,7 +248,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/starcoder2.py b/vllm/model_executor/models/starcoder2.py
index 9d9a2bff0e43..62ff9b618275 100644
--- a/vllm/model_executor/models/starcoder2.py
+++ b/vllm/model_executor/models/starcoder2.py
@@ -21,6 +21,7 @@
 # limitations under the License.
 """ PyTorch Starcoder2 model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Optional, Union
 
 import torch
@@ -250,7 +251,7 @@ def forward(
         else:
             assert intermediate_tensors is not None
             hidden_states = intermediate_tensors["hidden_states"]
-        for layer in self.layers[self.start_layer:self.end_layer]:
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states = layer(positions, hidden_states)
         if not get_pp_group().is_last_rank:
             return IntermediateTensors({"hidden_states": hidden_states})
diff --git a/vllm/model_executor/models/step3_text.py b/vllm/model_executor/models/step3_text.py
index 47d2af5c2a14..97611d3e140e 100644
--- a/vllm/model_executor/models/step3_text.py
+++ b/vllm/model_executor/models/step3_text.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Inference-only Jurassic model."""
 from collections.abc import Iterable
+from itertools import islice
 from typing import Any, Optional
 
 import torch
@@ -346,8 +347,7 @@ def forward(
             hidden_states = intermediate_tensors["hidden_states"]
             residual = intermediate_tensors["residual"]
 
-        for i in range(self.start_layer, self.end_layer):
-            layer = self.layers[i]
+        for layer in islice(self.layers, self.start_layer, self.end_layer):
             hidden_states, residual = layer(positions, hidden_states, residual)
 
         if not get_pp_group().is_last_rank:

```
