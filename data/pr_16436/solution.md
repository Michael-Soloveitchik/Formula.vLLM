# PR #16436: [Core] Speed up decode by remove synchronizing operation in sampler

- field: decoding_tpot  | area: model_executor_layers
- author: chanh
- state: MERGED  created: 2025-04-10T22:31:17Z  merged: 2025-04-21T18:18:23Z
- url: https://github.com/vllm-project/vllm/pull/16436
- labels: ready

# Issue

## Problem

`apply_penalties` in vllm/model_executor/layers/utils.py uses boolean advanced indexing (`logits[logits > 0]` and `logits[logits <= 0]`) to selectively scale logits by the repetition penalty. Boolean mask indexing on a GPU tensor invokes `nonzero`, which forces a CUDA device-to-host synchronization. The sync stalls the CPU until the GPU resolves the mask, serializing the sampling phase against the forward pass and blocking overlap.

## Example

During decode with `torch.cuda.set_sync_debug_mode("warn")`, PyTorch flags lines 52 and 54 of utils.py as synchronizing CUDA operations. On smaller models the removed sync yields an ~8% decode speedup.

## Use case

Online serving where repetition penalty is active and per-step decode latency dominates, so CPU sampling work must overlap with the GPU forward pass.

## Resources

CPU-GPU synchronization / kernel-launch pipelining. The sync blocks CPU scheduling and prevents GPU forward compute from overlapping with CPU sampling prep.

## Type of users

Users serving smaller models at low latency per token, where sampling-phase overhead is a large fraction of step time and repetition penalties are enabled.

## Kind of inputs

Decode-phase requests using repetition_penalty; effect most visible with small models and modest batch where GPU compute is short relative to CPU overhead.

## Mechanism with the issue

`logits[mask]` with a boolean GPU mask triggers `torch.nonzero`, a legitimate sync — CPU needs the nonzero count to size the index tensor and configure subsequent kernel launches (see pytorch/pytorch#12461). Four such masked-index ops per call each force a device sync, preventing CPU/GPU overlap.

# Solution

## Fix in words

Before, the code wrote back into logits via two boolean-masked index assignments (`logits[logits>0] /= ...[logits>0]`, `logits[logits<=0] *= ...[logits<=0]`), each triggering `nonzero` syncs. After, it computes a dense `penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)`, then builds a single dense `scaling = torch.where(logits > 0, 1.0/penalties, penalties)` and applies it with one elementwise `logits *= scaling`. `torch.where` produces a full-shape tensor with no nonzero/mask-indexing, so no CPU sync occurs. Result: sampling CPU work overlaps the GPU forward pass, ~8% decode speedup on small models.

## Author's solution

Author identified that boolean advanced indexing calls `nonzero`, which requires a CPU sync in PyTorch (confirmed via sync debug mode and pytorch/pytorch#12461, where the sync is deemed unavoidable for mask indexing). By refactoring to avoid the indexing, the sync is removed, letting much more sampling-phase CPU work overlap with the GPU forward pass, giving an 8% decode speedup for smaller models.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/utils.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/utils.py b/vllm/model_executor/layers/utils.py
index a9ef973917e1..5e56be0619b5 100644
--- a/vllm/model_executor/layers/utils.py
+++ b/vllm/model_executor/layers/utils.py
@@ -47,10 +47,15 @@ def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
         output_tokens_tensor, vocab_size, num_seqs)
     repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
         1, vocab_size)
-    logits[logits > 0] /= torch.where(prompt_mask | output_mask,
-                                      repetition_penalties, 1.0)[logits > 0]
-    logits[logits <= 0] *= torch.where(prompt_mask | output_mask,
-                                       repetition_penalties, 1.0)[logits <= 0]
+
+    # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
+    penalties = torch.where(prompt_mask | output_mask, repetition_penalties,
+                            1.0)
+
+    # If logits are positive, divide by penalty, otherwise multiply by penalty.
+    scaling = torch.where(logits > 0, 1.0 / penalties, penalties)
+    logits *= scaling
+
     # We follow the definition in OpenAI API.
     # Refer to https://platform.openai.com/docs/api-reference/parameter-details
     logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts

```
