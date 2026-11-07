# PR #7377: [Misc] Add numpy implementation of `compute_slot_mapping`

- field: attention  | area: attention_backends
- author: Yard1
- state: MERGED  created: 2024-08-09T19:57:07Z  merged: 2024-08-09T22:52:29Z
- url: https://github.com/vllm-project/vllm/pull/7377
- labels: ready

# Issue

## Problem

The Python implementation of `compute_slot_mapping` builds the prefill slot mapping by looping element-by-element over potentially large ranges of token positions, appending one slot at a time. For large sequences this pure-Python loop is slow because every iteration incurs interpreter overhead for integer division, modulo, list indexing, and `list.append`. The cost scales linearly with the number of tokens and becomes a measurable bottleneck during prefill.

## Example

A long prefill request with thousands of tokens forces thousands of Python loop iterations to compute each slot individually; the PR reports up to a 3x speedup from vectorizing this with numpy on large lists.

## Use case

Prefill phase of attention metadata construction when scheduling requests for a forward pass, especially long-context prompts processed in a single prefill.

## Resources

CPU time and kernel-launch/metadata-prep overhead on the host; the per-token Python interpreter loop stalls GPU utilization while slot mapping is assembled on CPU.

## Type of users

Users serving long-context prompts or large batched prefills where many token slots must be mapped per request.

## Kind of inputs

Large prefill sequences — high `seq_len` minus `context_len` (many slots). Small slot counts (< 256 elements) do not trigger the issue.

## Mechanism with the issue

`compute_slot_mapping` in `vllm/attention/backends/utils.py` used an inner `add_slot(i)` closure called in a Python `for i in range(...)` loop, computing `block_number * block_size + block_offset` and appending per element. The per-iteration interpreter overhead dominates for large ranges, making it slow versus a vectorized array computation.

# Solution

## Fix in words

The diff splits the slot computation into two helpers: `_compute_slot_mapping_python` (the original per-element loop) and `_compute_slot_mapping_numpy`, which builds the index array with `np.arange`, computes `block_offset` and `idx // block_size` vectorized, gathers `block_table_array[idx]`, then multiplies by `block_size` and adds the offset using numpy ops before `extend`-ing the result. `compute_slot_mapping` now computes `range_start`/`range_end` and the element count `numel`, extends the padding mask separately, and dispatches to numpy when `numel >= _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL` (256) else the python path. The old single-token decode optimization and inline `add_slot` closure are removed. Effect: large prefills use vectorized numpy for up to 3x speedup while small slot counts keep the python path to avoid numpy array-creation overhead.

## Author's solution

Add a numpy implementation of `compute_slot_mapping` that leverages vectorized instructions for up to 3x speedup on large lists, since the Python loop over large lists is inefficient. Keep the Python implementation for small slot counts, because numpy array-creation overhead causes a slowdown there.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/attention/backends/utils.py`

### Diff
```diff
diff --git a/vllm/attention/backends/utils.py b/vllm/attention/backends/utils.py
index 3ca668cb4e02..e6b5f820c5fa 100644
--- a/vllm/attention/backends/utils.py
+++ b/vllm/attention/backends/utils.py
@@ -1,6 +1,7 @@
 """Attention backend utils"""
 from typing import TYPE_CHECKING, Dict, List, Type, TypeVar, Union
 
+import numpy as np
 import torch
 
 from vllm.attention import AttentionMetadata, AttentionMetadataBuilder
@@ -13,6 +14,10 @@
 
 PAD_SLOT_ID = -1
 
+# Switch to numpy implementation of compute_slot_mapping
+# if we have at least this many elements. Could be tuned further.
+_COMPUTE_SLOT_MAPPING_NUMPY_NUMEL = 256
+
 if TYPE_CHECKING:
     from vllm.worker.model_runner import ModelInputForGPUBuilder
 
@@ -46,6 +51,29 @@ def compute_slot_mapping_start_idx(is_prompt: bool, query_len: int,
     return start_idx
 
 
+def _compute_slot_mapping_python(slot_mapping: List[int],
+                                 block_table: List[int], range_start: int,
+                                 range_end: int, block_size: int):
+    for i in range(range_start, range_end):
+        block_number = block_table[i // block_size]
+        block_offset = i % block_size
+        slot = block_number * block_size + block_offset
+        slot_mapping.append(slot)
+
+
+def _compute_slot_mapping_numpy(slot_mapping: List[int],
+                                block_table: List[int], range_start: int,
+                                range_end: int, block_size: int):
+    block_table_array = np.array(block_table)
+    idx = np.arange(range_start, range_end)
+    block_offset = idx % block_size
+    idx //= block_size
+    seq_slot_mapping_array = block_table_array[idx]
+    seq_slot_mapping_array *= block_size
+    seq_slot_mapping_array += block_offset
+    slot_mapping.extend(seq_slot_mapping_array)
+
+
 def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                          seq_id: int, seq_len: int, context_len: int,
                          start_idx: int, block_size: int,
@@ -67,21 +95,22 @@ def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
     # sliding window is 8, and block size is 4, the first two
     # tokens are masked and the slot mapping will be
     # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
-    block_table = block_tables[seq_id]
+    padding_mask_len = max(0, start_idx - context_len)
+    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)
 
-    def add_slot(i):
-        block_number = block_table[i // block_size]
-        block_offset = i % block_size
-        slot = block_number * block_size + block_offset
-        slot_mapping.append(slot)
+    range_start = max(start_idx, context_len)
+    range_end = seq_len
+    numel = range_end - range_start
+    block_table = block_tables[seq_id]
 
-    if start_idx == 0 and (seq_len - context_len) == 1:
-        # Optimization for common-case of decoding next token
-        add_slot(seq_len - 1)
+    # numpy implementation will be faster than python if we have
+    # many elements, otherwise it will be slower.
+    if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
+        _compute_slot_mapping_python(slot_mapping, block_table, range_start,
+                                     range_end, block_size)
     else:
-        slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
-        for i in range(max(start_idx, context_len), seq_len):
-            add_slot(i)
+        _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
+                                    range_end, block_size)
 
 
 TAttentionMetadata = TypeVar("TAttentionMetadata", bound='AttentionMetadata')

```
