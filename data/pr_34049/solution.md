# PR #34049: [Spec Decode] Reduce TP communication for speculative decoding draft token generation

- field: decoding_tpot  | area: vllm_config
- author: zixi-qi
- state: MERGED  created: 2026-02-07T09:17:20Z  merged: 2026-02-22T22:59:16Z
- url: https://github.com/vllm-project/vllm/pull/34049
- labels: speculative-decoding, ready, v1, llama

# Issue

## Problem

During Eagle speculative decoding draft token generation, vLLM all-gathers the full vocabulary logits tensor across tensor-parallel ranks every draft step, moving O(batch_size × vocab_size) data. For greedy draft selection only the argmax token ID is needed, so gathering the entire vocab distribution is wasted TP communication.

## Example

Llama 4 Maverick (vocab ~200K+ tokens) with EAGLE at TP=8, num_speculative_tokens=5: each of the 5 draft steps all-gathers batch_size × 200K floats across 8 ranks, when only one token ID per sequence is actually consumed.

## Use case

Multi-GPU TP serving with Eagle/EAGLE speculative decoding doing greedy (argmax) draft token proposal in non-tree speculation.

## Resources

Inter-GPU interconnect bandwidth (NVLink/PCIe) for the all-gather collective, which scales with vocab_size and dominates when the vocab is large.

## Type of users

Users running speculative decoding with large-vocabulary draft models across tensor-parallel GPU setups (e.g. Llama 4 + EAGLE, TP=8).

## Kind of inputs

Workloads where batch_size × draft_vocab_size is large — big vocab drafters and/or larger batches; gains marginal when batch small or draft vocab small.

## Mechanism with the issue

In `vllm/v1/spec_decode/eagle.py` `propose()`, draft tokens come from `self.model.compute_logits(...).argmax(dim=-1)`. `compute_logits` path (logits_processor `_get_logits`) uses `tensor_model_parallel_all_gather` to assemble the full vocab logits on every rank before the argmax, so the collective payload is O(vocab_size) per token even though only the max index survives.

# Solution

## Fix in words

Adds a `use_local_argmax_reduction` flag to `SpeculativeConfig` (default False). New `LogitsProcessor.get_top_tokens()` computes logits on each rank's local vocab shard, applies soft_cap/scale, masks padding beyond `org_vocab_size` to -inf, takes a local `max()` over the shard, and converts shard-local indices to global via `org_vocab_start_index`. Instead of all-gathering full logits it all-gathers only the (max_value, global_index) pairs — O(batch × 2 × tp_size) — stacked as float32 to avoid bf16 index precision loss, then `argmax` over rank values and `gather` to pick the winning global token ID. `eagle.py` routes greedy draft selection through a `_greedy_sample()` helper that calls `get_top_tokens()` when enabled (tree path still builds full logits for top-k); `llama4_eagle.py` falls back to full-logits argmax when `draft_id_to_target_id` vocab remapping is active, with validation/warn logs at init.

## Author's solution

Add a `use_local_argmax_reduction` option to speculative_config that computes local argmax on each TP rank's vocab shard, gathers only (max_value, global_index) pairs — O(batch_size × 2 × tp_size) instead of O(batch_size × vocab_size) — and reduces across ranks to find the global argmax. Defaults to False since gains are marginal when batch size or draft vocab is small. Benchmarked on Llama 4 Maverick + EAGLE (TP=8) with on-par acceptance rate and small TPOT improvement (~1.3%).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/config/speculative.py`
- `vllm/model_executor/layers/logits_processor.py`
- `vllm/model_executor/models/llama4_eagle.py`
- `vllm/v1/spec_decode/eagle.py`

### Diff
```diff
diff --git a/vllm/config/speculative.py b/vllm/config/speculative.py
index dcc549c4cf1a..0380fef420d3 100644
--- a/vllm/config/speculative.py
+++ b/vllm/config/speculative.py
@@ -109,6 +109,11 @@ class SpeculativeConfig:
     speculative input batches can contain sequences of different lengths,
     which may only be supported by certain attention backends. This currently
     only affects the EAGLE method of speculation."""
+    use_local_argmax_reduction: bool = False
+    """Use vocab-parallel local argmax instead of all-gathering full logits
+    for draft token generation. Reduces communication from O(vocab_size) to
+    O(2 * tp_size) per token. Only applies to greedy draft selection in
+    non-tree speculation."""
 
     # Ngram proposer configuration
     prompt_lookup_max: int | None = Field(default=None, ge=1)
diff --git a/vllm/model_executor/layers/logits_processor.py b/vllm/model_executor/layers/logits_processor.py
index 38753b0fcc74..dd2a61bc6a2c 100644
--- a/vllm/model_executor/layers/logits_processor.py
+++ b/vllm/model_executor/layers/logits_processor.py
@@ -5,6 +5,7 @@
 import torch
 
 from vllm.distributed import (
+    get_tensor_model_parallel_world_size,
     tensor_model_parallel_all_gather,
     tensor_model_parallel_gather,
 )
@@ -102,6 +103,58 @@ def _get_logits(
             logits = logits[..., : self.org_vocab_size]
         return logits
 
+    def get_top_tokens(
+        self,
+        lm_head: VocabParallelEmbedding,
+        hidden_states: torch.Tensor,
+        embedding_bias: torch.Tensor | None = None,
+    ) -> torch.Tensor:
+        """Vocab-parallel argmax without all-gathering full logits.
+
+        Each TP rank computes local argmax, then only the (value, index) pairs
+        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
+        O(batch * vocab_size).
+        """
+        if self.scale <= 0.0 and self.scale != 1.0:
+            raise ValueError(
+                "The local argmax reduction optimization is not supported for "
+                "non-positive logit scaling factors."
+            )
+        tp_size = get_tensor_model_parallel_world_size()
+
+        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
+        if self.soft_cap is not None:
+            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
+        if self.scale != 1.0:
+            logits = logits * self.scale
+
+        # Mask out padding entries beyond org_vocab_size on this shard.
+        num_pad = lm_head.shard_indices.num_org_vocab_padding
+        if num_pad > 0:
+            logits[..., -num_pad:] = -float("inf")
+
+        local_max_vals, local_max_indices = logits.max(dim=-1)
+
+        # Convert shard-local indices to global vocab indices.
+        vocab_start = lm_head.shard_indices.org_vocab_start_index
+        global_indices = local_max_indices + vocab_start
+
+        if tp_size == 1:
+            return global_indices
+
+        # All-gather (value, index) pairs, then reduce to global argmax.
+        # Use float32 to avoid bf16 precision loss on large vocab indices.
+        local_pair = torch.stack(
+            [local_max_vals.float(), global_indices.float()], dim=-1
+        )
+        # [batch, 2] -> [batch, 2 * tp_size]
+        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
+        # [batch, tp_size, 2] where [:, :, 0]=values, [:, :, 1]=indices
+        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
+        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
+        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
+        return top_tokens.squeeze(-1).to(torch.int64)
+
     def extra_repr(self) -> str:
         s = f"vocab_size={self.vocab_size}"
         s += f", org_vocab_size={self.org_vocab_size}"
diff --git a/vllm/model_executor/models/llama4_eagle.py b/vllm/model_executor/models/llama4_eagle.py
index 02f5b5ff639b..6c7b53d4d525 100644
--- a/vllm/model_executor/models/llama4_eagle.py
+++ b/vllm/model_executor/models/llama4_eagle.py
@@ -208,6 +208,23 @@ def forward(
     ) -> tuple[torch.Tensor, torch.Tensor]:
         return self.model(input_ids, positions, hidden_states, inputs_embeds)
 
+    def get_top_tokens(
+        self,
+        hidden_states: torch.Tensor,
+    ) -> torch.Tensor:
+        """Vocab-parallel argmax without all-gathering full logits.
+
+        Falls back to full logits when draft_id_to_target_id remapping is
+        active, since the shared lm_head covers the full target vocab but
+        the draft model only predicts over a subset (draft_vocab_size).
+        """
+        if (
+            hasattr(self, "draft_id_to_target_id")
+            and self.draft_id_to_target_id is not None
+        ):
+            return self.compute_logits(hidden_states).argmax(dim=-1)
+        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)
+
     def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
         def transform(inputs):
             name, loaded_weight = inputs
diff --git a/vllm/v1/spec_decode/eagle.py b/vllm/v1/spec_decode/eagle.py
index a6e7995bc7cb..5cab1854c747 100644
--- a/vllm/v1/spec_decode/eagle.py
+++ b/vllm/v1/spec_decode/eagle.py
@@ -99,6 +99,9 @@ def __init__(
         self.parallel_drafting_hidden_state_tensor: torch.Tensor | None = None
         if self.parallel_drafting:
             self._init_parallel_drafting_params()
+        self.use_local_argmax_reduction: bool = (
+            self.speculative_config.use_local_argmax_reduction
+        )
 
         # The drafter can get longer sequences than the target model.
         max_batch_size = vllm_config.scheduler_config.max_num_seqs
@@ -372,6 +375,12 @@ def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
 
         self.cudagraph_dispatcher.initialize_cudagraph_keys(eagle_cudagraph_mode)
 
+    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Greedy-sample draft tokens from hidden states."""
+        if self.use_local_argmax_reduction:
+            return self.model.get_top_tokens(hidden_states)
+        return self.model.compute_logits(hidden_states).argmax(dim=-1)
+
     def propose(
         self,
         # [num_tokens]
@@ -494,11 +503,10 @@ def propose(
                 last_hidden_states, hidden_states = ret_hidden_states
 
         sample_hidden_states = last_hidden_states[token_indices_to_sample]
-        logits = self.model.compute_logits(sample_hidden_states)
 
         # Early exit if there is only one draft token to be generated.
         if self.num_speculative_tokens == 1 or self.parallel_drafting:
-            draft_token_ids = logits.argmax(dim=-1)
+            draft_token_ids = self._greedy_sample(sample_hidden_states)
             return draft_token_ids.view(-1, self.num_speculative_tokens)
 
         if self.uses_mrope:
@@ -516,7 +524,8 @@ def propose(
             hidden_states = hidden_states[token_indices_to_sample]
 
         if isinstance(attn_metadata, TreeAttentionMetadata):
-            # Draft using tree attention.
+            # Draft using tree attention - requires full logits for top-k
+            logits = self.model.compute_logits(sample_hidden_states)
             draft_token_ids_list = self.propose_tree(
                 batch_size=batch_size,
                 logits=logits,
@@ -528,7 +537,7 @@ def propose(
             # [batch_size, num_tree_tokens]
             return torch.cat(draft_token_ids_list, dim=1)
 
-        draft_token_ids = logits.argmax(dim=-1)
+        draft_token_ids = self._greedy_sample(sample_hidden_states)
 
         if self.allowed_attn_types is not None and not isinstance(
             attn_metadata, self.allowed_attn_types
@@ -693,8 +702,7 @@ def propose(
                     last_hidden_states, hidden_states = ret_hidden_states
 
             hidden_states = hidden_states[:batch_size]
-            logits = self.model.compute_logits(last_hidden_states[:batch_size])
-            draft_token_ids = logits.argmax(dim=-1)
+            draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
             draft_token_ids_list.append(draft_token_ids)
 
         # [batch_size, num_speculative_tokens]
@@ -1524,6 +1532,31 @@ def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
                             "Shared target model lm_head with MTP shared_head.head."
                         )
 
+        if self.use_local_argmax_reduction:
+            if not hasattr(self.model, "get_top_tokens"):
+                raise ValueError(
+                    "use_local_argmax_reduction is enabled but draft model "
+                    f"{self.model.__class__.__name__} does not implement "
+                    "get_top_tokens()."
+                )
+            # Warn if draft model has vocab remapping, which forces fallback
+            # to the full-logits path (negating the optimization).
+            if (
+                hasattr(self.model, "draft_id_to_target_id")
+                and self.model.draft_id_to_target_id is not None
+            ):
+                logger.warning(
+                    "use_local_argmax_reduction is enabled but draft model "
+                    "uses draft_id_to_target_id vocab remapping. The "
+                    "optimization will be bypassed (falling back to full "
+                    "logits gather + argmax)."
+                )
+            else:
+                logger.info(
+                    "Using local argmax reduction for draft token generation "
+                    "(communication: O(2*tp_size) vs O(vocab_size))."
+                )
+
     @torch.inference_mode()
     def dummy_run(
         self,

```
