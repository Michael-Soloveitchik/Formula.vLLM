# PR #6442: [Core] Use numpy to speed up padded token processing

- field: scheduling  | area: model_executor_sampling_metadata.py
- author: peng1999
- state: MERGED  created: 2024-07-15T09:31:47Z  merged: 2024-07-16T15:13:26Z
- url: https://github.com/vllm-project/vllm/pull/6442
- labels: ready

# Issue

## Problem

When sampling penalties are enabled, the sampler builds padded prompt/output token matrices using nested Python list comprehensions that concatenate per-sequence token lists with `[vocab_size] * pad` lists, then hands the result to `torch.tensor`. Both the per-row list construction and `torch.tensor` conversion of a Python list-of-lists are slow, scaling poorly with batch size. This adds significant CPU overhead to every sampler call on large batches.

## Example

Benchmarked on an A100 80GB with large batch sizes, the padding path dominated sampler latency; replacing it cut sampler latency ~50% for large batches.

## Use case

High-throughput serving of small models where many requests are batched together and the sampler runs every decode step with penalties active (repetition/presence/frequency).

## Resources

CPU compute and Python interpreter overhead on the host, plus the CPU->tensor conversion cost before the pinned-memory transfer.

## Type of users

Users serving small models at high throughput with large batch sizes and sampling penalties enabled, where per-step CPU sampler overhead is a bottleneck.

## Kind of inputs

Large batches (many sequences) with varying prompt/output token lengths requiring padding to `prompt_max_len`/`output_max_len`, with penalty params set so `do_penalties` is true.

## Mechanism with the issue

`SamplingTensors.from_lists` builds `prompt_padded_tokens`/`output_padded_tokens` as list-of-lists via comprehension (`tokens + [vocab_size] * (max_len - len(tokens))`), then calls `torch.tensor(...)` on them. Allocating many temporary Python pad lists and converting a ragged-then-padded list-of-lists into a torch tensor is O(total_tokens) in slow Python/C-conversion, which grows with batch size.

# Solution

## Fix in words

Before: padded token matrices were built with Python list comprehensions and materialized via `torch.tensor(list_of_lists, dtype=torch.long, pin_memory=...)`. After: it preallocates an `np.full((num_seqs, max_len), vocab_size, dtype=np.int64)` array and fills each row in place with `prompt_padded_tokens[i, :len(tokens)] = tokens`, avoiding per-row temporary pad lists. The tensors are then created zero-copy with `torch.from_numpy(...)` and pinned explicitly via `.pin_memory()` when requested. Result: ~50% faster sampler on large batches by replacing slow Python list ops and list-to-tensor conversion with vectorized NumPy allocation and a direct numpy->tensor view.

## Author's solution

Replace the list comprehension with NumPy operations for padded token processing, yielding roughly a 50% sampler speedup at large batch sizes and improving throughput when serving small models, as shown in the author's A100 80GB benchmarks.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/sampling_metadata.py`

### Diff
```diff
diff --git a/vllm/model_executor/sampling_metadata.py b/vllm/model_executor/sampling_metadata.py
index ad5fb13176ed..c346cd056286 100644
--- a/vllm/model_executor/sampling_metadata.py
+++ b/vllm/model_executor/sampling_metadata.py
@@ -2,6 +2,7 @@
 from dataclasses import dataclass
 from typing import Dict, List, Optional, Tuple
 
+import numpy as np
 import torch
 
 from vllm.model_executor.layers.ops.sample import get_num_triton_sampler_splits
@@ -457,16 +458,20 @@ def from_lists(cls, temperatures: List[float], top_ps: List[float],
         if do_penalties:
             prompt_max_len = max([len(tokens) for tokens in prompt_tokens],
                                  default=0)
-            prompt_padded_tokens = [
-                tokens + [vocab_size] * (prompt_max_len - len(tokens))
-                for tokens in prompt_tokens
-            ]
+            prompt_padded_tokens = np.full(
+                (len(prompt_tokens), prompt_max_len),
+                vocab_size,
+                dtype=np.int64)
+            for i, tokens in enumerate(prompt_tokens):
+                prompt_padded_tokens[i, :len(tokens)] = tokens
             output_max_len = max([len(tokens) for tokens in output_tokens],
                                  default=0)
-            output_padded_tokens = [
-                tokens + [vocab_size] * (output_max_len - len(tokens))
-                for tokens in output_tokens
-            ]
+            output_padded_tokens = np.full(
+                (len(output_tokens), output_max_len),
+                vocab_size,
+                dtype=np.int64)
+            for i, tokens in enumerate(output_tokens):
+                output_padded_tokens[i, :len(tokens)] = tokens
 
         temperatures_t = torch.tensor(
             temperatures,
@@ -517,18 +522,11 @@ def from_lists(cls, temperatures: List[float], top_ps: List[float],
             pin_memory=pin_memory,
         )
         if do_penalties:
-            prompt_tensor = torch.tensor(
-                prompt_padded_tokens,
-                device="cpu",
-                dtype=torch.long,
-                pin_memory=pin_memory,
-            )
-            output_tensor = torch.tensor(
-                output_padded_tokens,
-                device="cpu",
-                dtype=torch.long,
-                pin_memory=pin_memory,
-            )
+            prompt_tensor = torch.from_numpy(prompt_padded_tokens)
+            output_tensor = torch.from_numpy(output_padded_tokens)
+            if pin_memory:
+                prompt_tensor = prompt_tensor.pin_memory()
+                output_tensor = output_tensor.pin_memory()
         else:
             prompt_tensor = None
             output_tensor = None

```
