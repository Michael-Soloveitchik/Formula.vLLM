# PR #32975: [Perf] Optimize detokenizer python logic

- field: scheduling  | area: v1_engine
- author: yewentao256
- state: MERGED  created: 2026-01-23T22:45:06Z  merged: 2026-02-10T07:54:41Z
- url: https://github.com/vllm-project/vllm/pull/32975
- labels: ready, v1

# Issue

## Problem

In `SlowIncrementalDetokenizer`, the `output_token_ids` property is not a plain list but computes a slice `self.token_ids[self.prompt_len:]` when a prompt is present. Code paths that only needed the count called `len(self.output_token_ids)`, which forced construction of a new sliced list just to measure its length. This wastes CPU and allocates a throwaway list on every detokenization step and every streaming-output decision.

## Example

A request with a long prompt (prompt_len large) that generates many output tokens: each `update()` call checking `min_tokens` and each `make_request_output()` streaming check slices the full `token_ids` list from `prompt_len` onward solely to call `len()` on the result.

## Use case

Standard v1 engine incremental detokenization during online serving, especially streaming responses where output length is polled repeatedly against `min_tokens`, `stop`, and `stream_interval`.

## Resources

CPU time and transient heap allocations (list copies) in the hot per-token Python detokenizer loop.

## Type of users

Users running the v1 engine with streaming output and non-empty prompts, particularly the slow (non-fast) detokenizer path.

## Kind of inputs

Requests with long prompts and/or many generated tokens, where `min_tokens`/`stop`/`stream_interval` checks run each decode step.

## Mechanism with the issue

`output_token_ids` in `SlowIncrementalDetokenizer` returns `self.token_ids[self.prompt_len:]`, a slice. Callers in `detokenizer.py` (`min_tokens` and `stop` checks) and `output_processor.py` (`make_request_output` streaming gate) wrapped this in `len(...)`, so each call allocated and copied a sliced list only to discard it after reading its length.

# Solution

## Fix in words

Diff adds a `num_output_tokens()` method on both detokenizer classes: base returns `len(self.token_ids)`, slow version returns `len(self.token_ids) - self.prompt_len` — computing the count arithmetically without slicing. All callers that previously did `len(self.detokenizer.output_token_ids)` now call `num_output_tokens()` instead (two sites in `detokenizer.py` for `min_tokens`/`stop`, two in `output_processor.py` for the stream-interval gate and `sent_tokens_offset`). This removes per-token list-copy allocations in the hot loop. Also drops the stale `TODO(woosuk)` comment about inefficiency.

## Author's solution

Author adds a `num_output_tokens` function to avoid `len(self.output_token_ids)` triggering a slice in `SlowIncrementalDetokenizer`. A second idea (accumulate string pieces and join once instead of repeated string concatenation) was proposed but struck out/dropped. Author notes e2e perf gain is <0.5% (within noise) and provided an AI-generated microbenchmark comparing old vs new `stop_check_offset` logic.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/engine/detokenizer.py`
- `vllm/v1/engine/output_processor.py`

### Diff
```diff
diff --git a/vllm/v1/engine/detokenizer.py b/vllm/v1/engine/detokenizer.py
index e77a316b2267..18e4c98f8829 100644
--- a/vllm/v1/engine/detokenizer.py
+++ b/vllm/v1/engine/detokenizer.py
@@ -35,6 +35,9 @@ def __init__(self):
     def output_token_ids(self) -> list[int]:
         return self.token_ids
 
+    def num_output_tokens(self) -> int:
+        return len(self.token_ids)
+
     def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
         self.token_ids.extend(new_token_ids)
         return None
@@ -112,14 +115,12 @@ def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
             skipped_stop_token_id = None
 
         # 1) Detokenize the new token ids incrementally.
-        # TODO(woosuk): This method becomes very inefficient when the number of
-        # new_token_ids is more than 1. We need to optimize this.
         stop_check_offset = len(self.output_text)
         for new_token_id in new_token_ids:
             self.token_ids.append(new_token_id)
             self.output_text += self.decode_next(new_token_id)
             # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
-            if self.min_tokens and len(self.output_token_ids) <= self.min_tokens:
+            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                 stop_check_offset = len(self.output_text)
 
         if skipped_stop_token_id is not None:
@@ -128,7 +129,7 @@ def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
 
         # 2) Evaluate stop strings.
         stop_string = None
-        if self.stop and len(self.output_token_ids) > self.min_tokens:
+        if self.stop and self.num_output_tokens() > self.min_tokens:
             stop = check_stop_strings(
                 output_text=self.output_text,
                 new_char_count=len(self.output_text) - stop_check_offset,
@@ -295,6 +296,9 @@ def output_token_ids(self) -> list[int]:
             else (self.token_ids[self.prompt_len :])
         )
 
+    def num_output_tokens(self) -> int:
+        return len(self.token_ids) - self.prompt_len
+
     def decode_next(self, next_token_id: int) -> str:
         new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
             tokenizer=self.tokenizer,
diff --git a/vllm/v1/engine/output_processor.py b/vllm/v1/engine/output_processor.py
index 00a5355e0221..58c73fbc6977 100644
--- a/vllm/v1/engine/output_processor.py
+++ b/vllm/v1/engine/output_processor.py
@@ -292,7 +292,7 @@ def make_request_output(
             if not (
                 finished
                 or self.sent_tokens_offset == 0
-                or len(self.detokenizer.output_token_ids) - self.sent_tokens_offset
+                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                 >= self.stream_interval
             ):
                 return None
@@ -303,7 +303,7 @@ def make_request_output(
                 new_token_ids = self.detokenizer.output_token_ids[
                     self.sent_tokens_offset :
                 ]
-                self.sent_tokens_offset = len(self.detokenizer.output_token_ids)
+                self.sent_tokens_offset = self.detokenizer.num_output_tokens()
 
         external_req_id = self.external_req_id
 

```
