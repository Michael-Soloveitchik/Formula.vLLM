# PR #36557: [Bugfix] Fix `RuntimeError: Already borrowed` that degrades VLM serving throughput under concurrent load. (#36557)

# Issue

## Original PR description (captured source)

### Root cause

`BaseRenderer.__init__` passes the same HF tokenizer instance to both the `AsyncMicrobatchTokenizer` (API-side tokenization) and the multimodal processor (via `create_processor`). These run on different threads:

```
Request A (main thread, Step 3 — MM processing):
  process_for_engine → _process_multimodal → mm_processor.apply
    → call_hf_processor → hf_processor() → tokenizer.encode()
    → _encode_plus → set_truncation_and_padding
    → self._tokenizer.enable_truncation()          ← BORROWS Rust RefCell

Request B (executor thread, Step 2 — tokenization):
  tokenize_prompts_async → AsyncMicrobatchTokenizer.encode
    → run_in_executor → tokenizer.__call__()
    → _encode_plus → set_truncation_and_padding
    → self._tokenizer.enable_truncation()          ← ALREADY BORROWED → RuntimeError
```

The `call_hf_processor` retry loop (up to 5× with `time.sleep(0.5)`) masks the error from clients but causes severe latency spikes. Text-only models are unaffected because they never call `call_hf_processor`.

Ref: https://github.com/huggingface/tokenizers/issues/537

### Fix

`copy.deepcopy(tokenizer)` before passing it to the multimodal processor, so it gets its own Rust tokenizer backend. Called once at startup (~1-5 MB, sub-second). Zero per-request cost.

## Test Plan

Start a VLM server:
```bash
vllm serve Qwen/Qwen3-VL-4B-Instruct --max-model-len 4096 --dtype bfloat16
```

Send 512 concurrent multimodal chat completion requests (concurrency=128) using the OpenAI async client. Each request includes a 1x1 PNG image as a base64 data URL and `max_tokens=1` (or `max_tokens=100`). Measure per-request latency and check server logs for `"Failed to acquire tokenizer"` retry warnings.

## Test Result

### `max_tokens=1` (prefill-only, 512 requests, concurrency 128)

| Metric | Before | After |
|---|---|---|
| Throughput | 14.4 req/s | **253.8 req/s** |
| p50 latency | 8.563s | **0.471s** |
| p99 latency | 12.046s | **0.566s** |

### `max_tokens=100` (prefill + decode, 512 requests, concurrency 128)

| Metric | Before | After |
|---|---|---|
| Throughput | 20.3 req/s | **246.3 req/s** |
| p50 latency | 6.538s | **0.474s** |
| p99 latency | 10.087s | **0.584s** |

### Server-side retries

| | Before | After |
|---|---|---|
| "Failed to acquire tokenizer" warnings | 114 | **0** |

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [36557.md](./36557.md) and [36557.json](./36557.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/36557
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/36557
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `96bed732c75e20b021a44c7915aaac1ebae4d1b25b9ea5a3de9929606321e52d`

Existing semantic notes in [36557.sem.json](./36557.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/renderers/base.py`

### Diff

```diff
diff --git a/vllm/renderers/base.py b/vllm/renderers/base.py
index a82646688f45..853a48945eac 100644
--- a/vllm/renderers/base.py
+++ b/vllm/renderers/base.py
@@ -1,6 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 import asyncio
+import copy
 import time
 from abc import ABC, abstractmethod
 from collections.abc import Mapping, Sequence
@@ -90,10 +91,17 @@ def __init__(self, config: "VllmConfig", tokenizer: _T | None) -> None:
 
             mm_processor_cache = mm_registry.processor_cache_from_config(config)
 
+            # Deep-copy the tokenizer so the multimodal processor gets its
+            # own Rust tokenizer backend.  Without this, concurrent access
+            # from AsyncMicrobatchTokenizer and call_hf_processor causes
+            # "RuntimeError: Already borrowed" from the Rust RefCell.
+            # See: https://github.com/huggingface/tokenizers/issues/537
+            mm_tokenizer = copy.deepcopy(tokenizer)
+
             with set_default_torch_num_threads():
                 self.mm_processor = mm_registry.create_processor(
                     config.model_config,
-                    tokenizer=tokenizer,
+                    tokenizer=mm_tokenizer,
                     cache=mm_processor_cache,
                 )
 
```
