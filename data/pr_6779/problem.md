# Task PR #6779 — [Core] Use array to speedup padding (#6779)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> `SequenceData` stored prompt and output token IDs as Python `list[int]` objects. Every sampling step, the Sampler rebuilt per-sequence token lists by copying these into fresh Python lists (`list(seq_data.prompt_token_ids)` / `list(seq_data.output_token_ids)`) before feeding them to `make_tensor_with_pad`. Python lists box each integer as a full PyObject, so copying and iterating them is slow and memory-heavy, and the copies happen on the hot path for every request every decode step.

### Issue — detailed

> `SamplingTensors.from_sampling_metadata` in `sampling_metadata.py` collected per-seq `prompt_tokens`/`output_tokens` as `List[List[int]]`, copying each sequence's IDs via `list(...)`. The underlying `SequenceData` in `sequence.py` held `_prompt_token_ids`/`_output_token_ids` as Python lists, so both storage and the per-step copy used boxed-int lists that are slow to allocate and iterate for `make_tensor_with_pad`.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> `SequenceData` now stores `_prompt_token_ids` and `_output_token_ids` as `array('l', ...)` (C `long` arrays) instead of Python lists, and exposes new `prompt_token_ids_array`/`output_token_ids_array` properties returning the raw arrays. `SamplingTensors.from_sampling_metadata` now appends these arrays directly (`seq_data.prompt_token_ids_array`) instead of copying into fresh lists via `list(...)`, changing the collected type from `List[List[int]]` to `List[array]`. Prefill-logprob padding fills use `array('l')` instead of `[]`, and `_apply_min_tokens_penalty` reads `output_token_ids_array`. `_update_cached_all_tokens` wraps the array concatenation in `list(...)` to keep the cached-all-tokens contract. Result: no per-step Python-list copies and compact contiguous storage, cutting sampling-tensor build latency ~3.6x.

**Author's words:**

> Use `array.array` in `SequenceData` to store token IDs, which greatly speeds up `make_tensor_with_pad` in the Sampler. The author validated with a micro-benchmark (79ms to 22ms at 1024 len / 2048 batch) and an end-to-end qwen-1.5-0.5b throughput test showing higher it/s and toks/s.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **make_tensor_with_pad** (ms): reported 79 -> 22 ms. workload: `python3 /task/bench.py`
- **e2e_qwen05b** (it/s): reported 24.76 -> 29.44 it/s.
- note: Microbench pasted in PR body; make_tensor_with_pad over SequenceData, input_len=1024 batch=2048. v0 API.

## Reported numbers (PR author)
- gain: 18% (e2e 24.76->29.44 it/s) (pct)
- evidence: Nice to see an 18% speedup from this optimization.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
