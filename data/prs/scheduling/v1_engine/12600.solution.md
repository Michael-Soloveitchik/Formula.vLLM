# PR #12600: [V1] Bugfix: Validate Model Input Length (#12600)

# Issue

## Original PR description (captured source)

SUMMARY:
* avoid crashing the engine when we get an input longer than max_model_len

FIX #12567(*link existing issues this PR will resolve*)



## Full discussion

See [12600.md](./12600.md) and [12600.json](./12600.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/12600
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/12600.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `c2669ca0e45959f8fc807fee7bedf67b417af13ef4052d5730e27f2e377924df`

Existing semantic notes in [12600.sem.json](./12600.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/v1/engine/processor.py`

### Diff

```diff
diff --git a/vllm/v1/engine/processor.py b/vllm/v1/engine/processor.py
index 43419d2ff538..6196c1105207 100644
--- a/vllm/v1/engine/processor.py
+++ b/vllm/v1/engine/processor.py
@@ -206,6 +206,11 @@ def _validate_model_inputs(self, inputs: ProcessorInputs):
         if prompt_ids is None or len(prompt_ids) == 0:
             raise ValueError("Prompt cannot be empty")
 
+        if len(prompt_ids) >= self.model_config.max_model_len:
+            raise ValueError(
+                f"Prompt length of {len(prompt_ids)} is longer than the "
+                f"maximum model length of {self.model_config.max_model_len}.")
+
         if self.model_config.is_multimodal_model:
             max_prompt_len = self.model_config.max_model_len
 
```
