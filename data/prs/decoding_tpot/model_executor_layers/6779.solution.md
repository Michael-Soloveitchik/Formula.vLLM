# PR #6779: [Core] Use array to speedup padding

- field: decoding_tpot  | area: model_executor_layers
- author: peng1999
- state: MERGED  created: 2024-07-25T09:08:36Z  merged: 2024-07-26T04:31:31Z
- url: https://github.com/vllm-project/vllm/pull/6779
- labels: ready

# Issue

## Problem

`SequenceData` stored prompt and output token IDs as Python `list[int]` objects. Every sampling step, the Sampler rebuilt per-sequence token lists by copying these into fresh Python lists (`list(seq_data.prompt_token_ids)` / `list(seq_data.output_token_ids)`) before feeding them to `make_tensor_with_pad`. Python lists box each integer as a full PyObject, so copying and iterating them is slow and memory-heavy, and the copies happen on the hot path for every request every decode step.

## Example

Micro-benchmark with 1024 input length and 2048 batch size: building the sampling tensors took 79ms before, 22ms after. qwen-1.5-0.5b end-to-end throughput rose from 24.76 to 29.44 it/s (output 3165 to 3765 toks/s).

## Use case

High-batch LLM serving where the Sampler runs once per decode step across the whole batch, so per-sequence token-list handling is on the critical TPOT path.

## Resources

CPU time and host memory — Python-object boxing/copying of token-id lists, plus the host-to-pinned-memory tensor construction overhead.

## Type of users

Users serving with large batch sizes and/or long prompts, where the sampling metadata build dominates per-step CPU cost.

## Kind of inputs

Large batch size (e.g. 2048) with long input sequences (e.g. 1024 tokens); cost scales with batch_size x sequence_length token IDs processed each step.

## Mechanism with the issue

`SamplingTensors.from_sampling_metadata` in `sampling_metadata.py` collected per-seq `prompt_tokens`/`output_tokens` as `List[List[int]]`, copying each sequence's IDs via `list(...)`. The underlying `SequenceData` in `sequence.py` held `_prompt_token_ids`/`_output_token_ids` as Python lists, so both storage and the per-step copy used boxed-int lists that are slow to allocate and iterate for `make_tensor_with_pad`.

# Solution

## Fix in words

`SequenceData` now stores `_prompt_token_ids` and `_output_token_ids` as `array('l', ...)` (C `long` arrays) instead of Python lists, and exposes new `prompt_token_ids_array`/`output_token_ids_array` properties returning the raw arrays. `SamplingTensors.from_sampling_metadata` now appends these arrays directly (`seq_data.prompt_token_ids_array`) instead of copying into fresh lists via `list(...)`, changing the collected type from `List[List[int]]` to `List[array]`. Prefill-logprob padding fills use `array('l')` instead of `[]`, and `_apply_min_tokens_penalty` reads `output_token_ids_array`. `_update_cached_all_tokens` wraps the array concatenation in `list(...)` to keep the cached-all-tokens contract. Result: no per-step Python-list copies and compact contiguous storage, cutting sampling-tensor build latency ~3.6x.

## Author's solution

Use `array.array` in `SequenceData` to store token IDs, which greatly speeds up `make_tensor_with_pad` in the Sampler. The author validated with a micro-benchmark (79ms to 22ms at 1024 len / 2048 batch) and an end-to-end qwen-1.5-0.5b throughput test showing higher it/s and toks/s.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/sampler.py`
- `vllm/model_executor/sampling_metadata.py`
- `vllm/sequence.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/sampler.py b/vllm/model_executor/layers/sampler.py
index 5c376797a054..121458f8156a 100644
--- a/vllm/model_executor/layers/sampler.py
+++ b/vllm/model_executor/layers/sampler.py
@@ -220,7 +220,7 @@ def _apply_min_tokens_penalty(
             seqs_to_penalize: List[int] = []
             for j, seq_id in enumerate(seq_ids):
                 seq_data = seq_group.seq_data[seq_id]
-                if len(seq_data.output_token_ids) < min_tokens:
+                if len(seq_data.output_token_ids_array) < min_tokens:
                     seqs_to_penalize.append(j)
 
             if seqs_to_penalize:
diff --git a/vllm/model_executor/sampling_metadata.py b/vllm/model_executor/sampling_metadata.py
index 390b5d173ebc..27b37a9d5347 100644
--- a/vllm/model_executor/sampling_metadata.py
+++ b/vllm/model_executor/sampling_metadata.py
@@ -1,4 +1,5 @@
 import random
+from array import array
 from dataclasses import dataclass
 from typing import Dict, List, Optional, Tuple
 
@@ -329,8 +330,8 @@ def from_sampling_metadata(
             user-defined seed for each sequence.
         extra_entropy: extra entropy to use when generating seeds.
         """
-        prompt_tokens: List[List[int]] = []
-        output_tokens: List[List[int]] = []
+        prompt_tokens: List[array] = []
+        output_tokens: List[array] = []
         top_ks: List[int] = []
         temperatures: List[float] = []
         top_ps: List[float] = []
@@ -432,13 +433,15 @@ def from_sampling_metadata(
                 if (seq_group.is_prompt
                         and sampling_params.prompt_logprobs is not None):
                     prefill_len = len(seq_group.prompt_logprob_indices)
-                    prompt_tokens.extend([] for _ in range(prefill_len))
-                    output_tokens.extend([] for _ in range(prefill_len))
+                    prompt_tokens.extend(
+                        array('l') for _ in range(prefill_len))
+                    output_tokens.extend(
+                        array('l') for _ in range(prefill_len))
                 if seq_group.do_sample:
                     for seq_id in seq_ids:
                         seq_data = seq_group.seq_data[seq_id]
-                        prompt_tokens.append(list(seq_data.prompt_token_ids))
-                        output_tokens.append(list(seq_data.output_token_ids))
+                        prompt_tokens.append(seq_data.prompt_token_ids_array)
+                        output_tokens.append(seq_data.output_token_ids_array)
 
         sampling_tensors = SamplingTensors.from_lists(
             temperatures, top_ps, top_ks, min_ps, presence_penalties,
@@ -454,9 +457,9 @@ def from_lists(cls, temperatures: List[float], top_ps: List[float],
                    frequency_penalties: List[float],
                    repetition_penalties: List[float],
                    sampling_seeds: List[int], sample_indices: List[int],
-                   prompt_tokens: List[List[int]],
-                   output_tokens: List[List[int]], vocab_size: int,
-                   extra_seeds_to_generate: int, device: torch.device,
+                   prompt_tokens: List[array], output_tokens: List[array],
+                   vocab_size: int, extra_seeds_to_generate: int,
+                   device: torch.device,
                    dtype: torch.dtype) -> "SamplingTensors":
         # Note that the performance will be very bad without
         # pinned memory.
diff --git a/vllm/sequence.py b/vllm/sequence.py
index 0cd4c7e71d78..72821ecea0f4 100644
--- a/vllm/sequence.py
+++ b/vllm/sequence.py
@@ -3,6 +3,7 @@
 import enum
 import math
 from abc import ABC, abstractmethod
+from array import array
 from collections import defaultdict
 from dataclasses import dataclass, field
 from typing import (TYPE_CHECKING, Dict, List, Mapping, Optional, Set, Tuple,
@@ -119,10 +120,10 @@ def __init__(
         prompt_token_ids: List[int],
         output_token_ids: Optional[List[int]] = None,
     ) -> None:
-        self._prompt_token_ids: List[int] = list(prompt_token_ids)
+        self._prompt_token_ids = array('l', prompt_token_ids)
         self._prompt_token_ids_tuple: Tuple[int, ...] = tuple(prompt_token_ids)
-        self._output_token_ids: List[int] = (
-            list(output_token_ids) if output_token_ids is not None else [])
+        self._output_token_ids = array(
+            'l', output_token_ids if output_token_ids is not None else [])
 
         self.cumulative_logprob = 0.0
         # The number of tokens that are computed (that run against the model).
@@ -132,8 +133,8 @@ def __init__(
         self._update_cached_all_tokens()
 
     def _update_cached_all_tokens(self):
-        self._cached_all_token_ids: List[int] = (self._prompt_token_ids +
-                                                 self._output_token_ids)
+        self._cached_all_token_ids: List[int] = list(self._prompt_token_ids +
+                                                     self._output_token_ids)
 
     @property
     def prompt_token_ids(self) -> Tuple[int, ...]:
@@ -141,19 +142,27 @@ def prompt_token_ids(self) -> Tuple[int, ...]:
 
     @prompt_token_ids.setter
     def prompt_token_ids(self, new_prompt_token_ids) -> None:
-        self._prompt_token_ids = list(new_prompt_token_ids)
+        self._prompt_token_ids = array('l', new_prompt_token_ids)
         self._prompt_token_ids_tuple = tuple(new_prompt_token_ids)
         self._update_cached_all_tokens()
 
+    @property
+    def prompt_token_ids_array(self) -> array:
+        return self._prompt_token_ids
+
     @property
     def output_token_ids(self) -> Tuple[int, ...]:
         return tuple(self._output_token_ids)
 
     @output_token_ids.setter
     def output_token_ids(self, new_output_token_ids) -> None:
-        self._output_token_ids = list(new_output_token_ids)
+        self._output_token_ids = array('l', new_output_token_ids)
         self._update_cached_all_tokens()
 
+    @property
+    def output_token_ids_array(self) -> array:
+        return self._output_token_ids
+
     def append_token_id(self, token_id: int, logprob: float) -> None:
         self._output_token_ids.append(token_id)
         self._cached_all_token_ids.append(token_id)

```
