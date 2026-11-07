# PR #6455: [Distributed][Model] Rank-based Component Creation for Pipeline Parallelism Memory Optimization

- field: parallelism  | area: model_executor_models
- author: wushidonguc
- state: MERGED  created: 2024-07-15T22:58:42Z  merged: 2024-07-17T02:20:26Z
- url: https://github.com/vllm-project/vllm/pull/6455
- labels: ready

# Issue

## Problem

In Pipeline Parallel (PP) setups, the Llama model instantiated every component on every rank regardless of whether that rank used it. The embedding layer, final RMSNorm, lm_head, logits processor, and sampler were all created on all PP ranks. Only the first rank needs embeddings and only the last rank needs the norm/lm_head/logits/sampler, so intermediate ranks held large unused weight tensors, wasting HBM.

## Example

Serving Llama-3-70B with `--pipeline-parallel-size 8`: each of the 8 ranks allocated a full VocabParallelEmbedding and ParallelLMHead even though most never use them, costing up to 25% extra per-rank memory.

## Use case

Running large LLMs across multiple GPUs via pipeline parallelism to fit models that exceed single-GPU memory.

## Resources

GPU HBM/memory — redundant embedding and lm_head weight tensors consume VRAM that could otherwise serve more KV-cache blocks.

## Type of users

Users serving large models (e.g. 70B) with multi-GPU pipeline parallelism, especially in memory-constrained environments.

## Kind of inputs

Any workload under PP>1; the issue is config-driven (pipeline_parallel_size, vocab_size, hidden_size), not input-shape dependent.

## Mechanism with the issue

In `LlamaModel.__init__` and `LlamaForCausalLM.__init__` (vllm/model_executor/models/llama.py), `embed_tokens`, `norm`, `lm_head`, `logits_processor`, and `sampler` were constructed unconditionally. Since vLLM derives final block count from the minimum free memory across ranks, the unused weights on intermediate ranks raised the memory lower bound and reduced available KV-cache.

# Solution

## Fix in words

The diff gates component creation on PP rank position. `embed_tokens` is now a real `VocabParallelEmbedding` only when `get_pp_group().is_first_rank` (or last rank with `tie_word_embeddings`), else a lightweight `PPMissingLayer()` placeholder. `norm` is a real `RMSNorm` only on `is_last_rank`, else `PPMissingLayer()`. In `LlamaForCausalLM`, `unpadded_vocab_size`, `lm_head`, `logits_processor`, and `sampler` are created only on `is_last_rank`, with `lm_head` set to `PPMissingLayer()` otherwise. This removes unused embedding/lm_head weights from intermediate and non-relevant ranks, lowering the per-rank memory floor (~25% for Llama-3-70B at PP=8) and freeing HBM for KV-cache.

## Author's solution

The author selectively creates components only on the ranks that use them: the embedding layer on the first rank, and the norm layer, lm_head, logits processor, and sampler on the last rank. This cuts redundant per-rank allocations, yielding up to 25% per-rank memory savings for Llama-3-70B at `--pipeline-parallel-size 8` and enabling higher throughput and serving of larger models under constrained memory.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/llama.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/llama.py b/vllm/model_executor/models/llama.py
index f03e34b9e7c9..4c434e54cf74 100644
--- a/vllm/model_executor/models/llama.py
+++ b/vllm/model_executor/models/llama.py
@@ -50,7 +50,7 @@
 from vllm.utils import is_hip
 
 from .interfaces import SupportsLoRA
-from .utils import is_pp_missing_parameter, make_layers
+from .utils import PPMissingLayer, is_pp_missing_parameter, make_layers
 
 
 class LlamaMLP(nn.Module):
@@ -257,17 +257,24 @@ def __init__(
                       (lora_config.max_loras or 1)) if lora_config else 0
         self.vocab_size = config.vocab_size + lora_vocab
         self.org_vocab_size = config.vocab_size
-        self.embed_tokens = VocabParallelEmbedding(
-            self.vocab_size,
-            config.hidden_size,
-            org_num_embeddings=config.vocab_size,
-        )
+        if get_pp_group().is_first_rank or (config.tie_word_embeddings
+                                            and get_pp_group().is_last_rank):
+            self.embed_tokens = VocabParallelEmbedding(
+                self.vocab_size,
+                config.hidden_size,
+                org_num_embeddings=config.vocab_size,
+            )
+        else:
+            self.embed_tokens = PPMissingLayer()
         self.start_layer, self.end_layer, self.layers = make_layers(
             config.num_hidden_layers,
             lambda: LlamaDecoderLayer(config=config,
                                       cache_config=cache_config,
                                       quant_config=quant_config))
-        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+        if get_pp_group().is_last_rank:
+            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+        else:
+            self.norm = PPMissingLayer()
 
     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
         return self.embed_tokens(input_ids)
@@ -360,26 +367,30 @@ def __init__(
                                 cache_config,
                                 quant_config,
                                 lora_config=lora_config)
-        self.unpadded_vocab_size = config.vocab_size
-        if lora_config:
-            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
-        self.lm_head = ParallelLMHead(
-            self.unpadded_vocab_size,
-            config.hidden_size,
-            org_num_embeddings=config.vocab_size,
-            padding_size=DEFAULT_VOCAB_PADDING_SIZE
-            # We need bigger padding if using lora for kernel
-            # compatibility
-            if not lora_config else lora_config.lora_vocab_padding_size,
-            quant_config=quant_config,
-        )
-        if config.tie_word_embeddings:
-            self.lm_head.weight = self.model.embed_tokens.weight
-
-        logit_scale = getattr(config, "logit_scale", 1.0)
-        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
-                                                config.vocab_size, logit_scale)
-        self.sampler = Sampler()
+        if get_pp_group().is_last_rank:
+            self.unpadded_vocab_size = config.vocab_size
+            if lora_config:
+                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
+            self.lm_head = ParallelLMHead(
+                self.unpadded_vocab_size,
+                config.hidden_size,
+                org_num_embeddings=config.vocab_size,
+                padding_size=DEFAULT_VOCAB_PADDING_SIZE
+                # We need bigger padding if using lora for kernel
+                # compatibility
+                if not lora_config else lora_config.lora_vocab_padding_size,
+                quant_config=quant_config,
+            )
+            if config.tie_word_embeddings:
+                self.lm_head.weight = self.model.embed_tokens.weight
+
+            logit_scale = getattr(config, "logit_scale", 1.0)
+            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
+                                                    config.vocab_size,
+                                                    logit_scale)
+            self.sampler = Sampler()
+        else:
+            self.lm_head = PPMissingLayer()
 
     def forward(
         self,

```
