# PR #36557: [Bugfix] Fix `RuntimeError: Already borrowed` that degrades VLM serving throughput under concurrent load.

- field: scheduling  | area: vllm/renderers
- author: hallerite
- state: MERGED  created: 2026-03-09T23:05:00Z  merged: 2026-03-10T05:30:51Z
- url: https://github.com/vllm-project/vllm/pull/36557
- labels: bug, ready

## Body

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



## Comments

### github-actions @ 2026-03-09T23:05:10Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---