# Task PR #6779 — [Core] Use array to speedup padding (#6779)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **make_tensor_with_pad** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **make_tensor_with_pad** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 79 -> 22 ms.
- **e2e_qwen05b** — `it/s` (iterations per second), ↑ higher is better. reported 24.76 -> 29.44 it/s.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Microbench pasted in PR body; make_tensor_with_pad over SequenceData, input_len=1024 batch=2048. v0 API.

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

## Reported numbers (PR author)
- gain: 18% (e2e 24.76->29.44 it/s) (pct)
- evidence: Nice to see an 18% speedup from this optimization.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
