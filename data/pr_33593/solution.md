# PR #33593: [Perf] Optimize Python Slice for Structured Output using `islice` instead of [:]

- field: decoding_tpot  | area: vllm_reasoning
- author: yewentao256
- state: MERGED  created: 2026-02-02T21:19:02Z  merged: 2026-02-24T17:02:36Z
- url: https://github.com/vllm-project/vllm/pull/33593
- labels: structured-output, ready, v1, deepseek

# Issue

## Problem

The reasoning parsers used Python full-copy slicing (`list[:]`, `list[:-1]`, `list[delta_from:]`, `list[::-1]`) on the token-id list every streaming decode step. Each slice allocates a brand-new list and copies the elements, even though the result is only consumed by a membership test (`in`) or a forward iteration. For the common `end_token_id in input_ids[:-1]` and `all_token_ids[delta_from:]` paths, this is O(n) time and O(n) memory per call where only a short scan or lazy view is needed.

## Example

In the micro-benchmark, creating a 1M-element slice of a 2M list costs ~0.0034s and peaks at 7.63 MiB, versus ~1e-6s and 0.00 MiB for `islice`. Every generated token re-copies the growing `all_token_ids` list in `should_advance`.

## Use case

Structured/guided decoding with a reasoning parser (e.g. DeepSeek-R1 with `deepseek_r1` reasoning-parser) during long streaming generations, where `is_reasoning_end_streaming` / `extract_content_ids` run on each decode step.

## Resources

CPU time and transient host memory (allocation + copy churn, GC pressure) on the per-token decode hot path; adds to TPOT/ITL latency.

## Type of users

Users serving reasoning models with structured output enabled, especially with long outputs where the token list grows large (benchmark used `--output-len 1024`).

## Kind of inputs

Long and growing `input_ids` / `all_token_ids` sequences; the cost scales with total generated token count since the full list is re-sliced each step.

## Mechanism with the issue

`basic_parsers.extract_content_ids` did `end_token_id not in input_ids[:-1]`, `step3` similarly, `mistral.is_reasoning_end` iterated `input_ids[::-1]`, and `structured_output.should_advance` passed `all_token_ids[delta_from:]` as `delta_ids`. Each slice materializes a full copy before being scanned, so the repeated copying dominates when the list is large.

# Solution

## Fix in words

Replaces eager full-copy slices with lazy iterators. `input_ids[:-1]` membership tests become `islice(input_ids, 0, max(0, len(input_ids) - 1))`, avoiding a new list for the `in` scan. `input_ids[::-1]` in the mistral parser becomes `reversed(input_ids)`, dropping the reversed-copy allocation. In `should_advance`, `all_token_ids[delta_from:]` becomes `itertools.islice(all_token_ids, start, None)` with `start` computed to correctly handle negative `delta_from` (`max(len + delta_from, 0)`). The `delta_ids` parameter type is widened from `Sequence[int]` to `Iterable[int]` across all parsers so iterators are accepted. Net effect: O(1) view creation and near-zero transient memory instead of O(n) copies per decode step.

## Author's solution

Author replaced Python list splits/slices with iterators (`islice`), giving a perf improvement by avoiding copy and allocation. Backed with a micro-benchmark showing `islice` creation at ~1e-6s / 0 MiB vs slice at ~0.0034s / 7.63 MiB, plus an E2E structured-output serving benchmark on DeepSeek-R1-Distill-Qwen-7B showing lower mean TPOT (7.82ms vs main) and TTFT.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/reasoning/abs_reasoning_parsers.py`
- `vllm/reasoning/basic_parsers.py`
- `vllm/reasoning/deepseek_v3_reasoning_parser.py`
- `vllm/reasoning/identity_reasoning_parser.py`
- `vllm/reasoning/mistral_reasoning_parser.py`
- `vllm/reasoning/step3_reasoning_parser.py`
- `vllm/reasoning/step3p5_reasoning_parser.py`
- `vllm/v1/structured_output/__init__.py`

### Diff
```diff
diff --git a/vllm/reasoning/abs_reasoning_parsers.py b/vllm/reasoning/abs_reasoning_parsers.py
index 496eaaf3f366..83c3e6b90ac2 100644
--- a/vllm/reasoning/abs_reasoning_parsers.py
+++ b/vllm/reasoning/abs_reasoning_parsers.py
@@ -4,7 +4,7 @@
 import importlib
 import os
 from abc import abstractmethod
-from collections.abc import Callable, Sequence
+from collections.abc import Callable, Iterable, Sequence
 from functools import cached_property
 from typing import TYPE_CHECKING, Any
 
@@ -68,7 +68,7 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         """
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         """
         Check if the reasoning content ends in the input_ids on a
diff --git a/vllm/reasoning/basic_parsers.py b/vllm/reasoning/basic_parsers.py
index c066032fb325..5b1c0111cdd7 100644
--- a/vllm/reasoning/basic_parsers.py
+++ b/vllm/reasoning/basic_parsers.py
@@ -2,7 +2,8 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
 from abc import abstractmethod
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
+from itertools import islice
 from typing import TYPE_CHECKING, Any
 
 from vllm.entrypoints.openai.engine.protocol import DeltaMessage
@@ -77,7 +78,7 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         return False
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         end_token_id = self.end_token_id
         return end_token_id in delta_ids
@@ -86,7 +87,7 @@ def extract_content_ids(self, input_ids: list[int]) -> list[int]:
         """
         Extract the content after the end tokens
         """
-        if self.end_token_id not in input_ids[:-1]:
+        if self.end_token_id not in islice(input_ids, 0, max(0, len(input_ids) - 1)):
             return []
         else:
             return input_ids[input_ids.index(self.end_token_id) + 1 :]
diff --git a/vllm/reasoning/deepseek_v3_reasoning_parser.py b/vllm/reasoning/deepseek_v3_reasoning_parser.py
index e40f225907d5..c2efe650089d 100644
--- a/vllm/reasoning/deepseek_v3_reasoning_parser.py
+++ b/vllm/reasoning/deepseek_v3_reasoning_parser.py
@@ -1,7 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
 
 from transformers import PreTrainedTokenizerBase
 
@@ -41,7 +41,7 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         return self._parser.is_reasoning_end(input_ids)
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         return self._parser.is_reasoning_end_streaming(input_ids, delta_ids)
 
diff --git a/vllm/reasoning/identity_reasoning_parser.py b/vllm/reasoning/identity_reasoning_parser.py
index e1106362dfff..3c76901a31a3 100644
--- a/vllm/reasoning/identity_reasoning_parser.py
+++ b/vllm/reasoning/identity_reasoning_parser.py
@@ -1,7 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
 
 from transformers import PreTrainedTokenizerBase
 
@@ -36,7 +36,7 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         return True
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         return True
 
diff --git a/vllm/reasoning/mistral_reasoning_parser.py b/vllm/reasoning/mistral_reasoning_parser.py
index d7347462692a..c085ba4e4f21 100644
--- a/vllm/reasoning/mistral_reasoning_parser.py
+++ b/vllm/reasoning/mistral_reasoning_parser.py
@@ -69,7 +69,7 @@ def end_token(self) -> str:
     def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         has_eot_token = False
 
-        for id in input_ids[::-1]:
+        for id in reversed(input_ids):
             if id == self.start_token_id:
                 # Reasoning ends only if a BOT token is found before a EOT token.
                 return has_eot_token
diff --git a/vllm/reasoning/step3_reasoning_parser.py b/vllm/reasoning/step3_reasoning_parser.py
index 4758246acb3e..d932ba8b62dd 100644
--- a/vllm/reasoning/step3_reasoning_parser.py
+++ b/vllm/reasoning/step3_reasoning_parser.py
@@ -1,7 +1,8 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
+from itertools import islice
 
 import regex as re
 from transformers import PreTrainedTokenizerBase
@@ -104,13 +105,15 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         return self.think_end_token_id in input_ids
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         end_token_id = self.think_end_token_id
         return end_token_id in delta_ids
 
     def extract_content_ids(self, input_ids: list[int]) -> list[int]:
-        if self.think_end_token_id not in input_ids[:-1]:
+        if self.think_end_token_id not in islice(
+            input_ids, 0, max(0, len(input_ids) - 1)
+        ):
             return []
         else:
             return input_ids[input_ids.index(self.think_end_token_id) + 1 :]
diff --git a/vllm/reasoning/step3p5_reasoning_parser.py b/vllm/reasoning/step3p5_reasoning_parser.py
index b93f551426fb..af9aa4b4141b 100644
--- a/vllm/reasoning/step3p5_reasoning_parser.py
+++ b/vllm/reasoning/step3p5_reasoning_parser.py
@@ -1,7 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
-from collections.abc import Sequence
+from collections.abc import Iterable, Sequence
 
 from vllm.entrypoints.openai.chat_completion.protocol import (
     ChatCompletionRequest,
@@ -51,7 +51,7 @@ def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
         return self.end_offset < 1
 
     def is_reasoning_end_streaming(
-        self, input_ids: Sequence[int], delta_ids: Sequence[int]
+        self, input_ids: Sequence[int], delta_ids: Iterable[int]
     ) -> bool:
         if self.end_token_id in input_ids and self.end_offset > 0:
             self.end_offset -= 1
diff --git a/vllm/v1/structured_output/__init__.py b/vllm/v1/structured_output/__init__.py
index 921bee6a647a..cd17a21d9b3b 100644
--- a/vllm/v1/structured_output/__init__.py
+++ b/vllm/v1/structured_output/__init__.py
@@ -324,8 +324,11 @@ def should_advance(self, request: "Request") -> bool:
         # Check if reasoning ends in *this* step
         delta_from = request.num_computed_tokens - request.num_output_placeholders
         all_token_ids = request.all_token_ids
+        start = (
+            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
+        )
         if self.reasoner.is_reasoning_end_streaming(
-            all_token_ids, all_token_ids[delta_from:]
+            all_token_ids, itertools.islice(all_token_ids, start, None)
         ):
             # Reasoning just ended, so we shouldn't advance til
             # next pass

```
