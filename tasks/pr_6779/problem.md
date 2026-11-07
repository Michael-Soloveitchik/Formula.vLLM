# Task PR #6779 — [Core] Use array to speedup padding (#6779)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
Before sampling, the system rebuilds Python lists of each sequence's prompt and output token IDs for every decoding step, then pads them into a tensor. Building these lists is slow because Python boxes each integer separately. With large batch sizes this per-step CPU copying becomes significant and can slow token generation, especially when the model is small and fast.

### Issue — detailed
In `from_sampling_metadata` (vllm/model_executor/sampling_metadata.py), token IDs were gathered into `List[List[int]]` via `list(seq_data.prompt_token_ids)` and `list(seq_data.output_token_ids)` per sequence, copying each sequence's tokens into a new Python list each step. `SequenceData` (vllm/sequence.py) stored `_prompt_token_ids` and `_output_token_ids` as plain `List[int]`. These lists feed `make_tensor_with_pad` via `SamplingTensors.from_lists`, so per-step copy cost scales with batch size. This CPU work runs independent of logits and overlaps with GPU inference, but for small models it exceeds inference time and lands on the critical path.

## PR's solution (reference — what was merged)
**General:** Store token IDs in a compact typed array instead of a Python list. Expose the arrays directly so the sampler reuses them without copying into new lists. This cuts the per-step CPU cost of building the padded tensor.

**Detailed:** In `SequenceData` (vllm/sequence.py), `_prompt_token_ids` and `_output_token_ids` now use `array('l', ...)` instead of `List[int]`, and new properties `prompt_token_ids_array` / `output_token_ids_array` return the arrays directly. In `from_sampling_metadata` (vllm/model_executor/sampling_metadata.py), the per-sequence `list(seq_data.prompt_token_ids)` / `list(seq_data.output_token_ids)` calls are replaced with `seq_data.prompt_token_ids_array` / `output_token_ids_array`, avoiding the copy; the prefill-logprobs path appends empty `array('l')` instead of `[]`, and type hints change from `List[List[int]]` to `List[array]` (including `from_lists`). `_apply_min_tokens_penalty` in sampler.py switches its length check to `output_token_ids_array`, and `_update_cached_all_tokens` wraps the array concatenation in `list(...)`.

**Author's words:** Using `array.array` in SequenceData greatly improves performance of `make_tensor_with_pad` in Sampler. The `from_sampling_metadata` function optimized by this PR primarily runs on the CPU and is independent of logits, so it can overlap with the GPU work of model inference. It will only be on the critical path if its execution time exceeds that of model inference, which occurs with smaller models. This PR is for small models and large batch sizes.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **make_tensor_with_pad** (ms): reported 79.0 -> 22.0 ms. workload: latency ms
- **e2e_qwen05b** (it/s): reported 24.76 -> 29.44 it/s. workload: it/s

## Reported numbers (PR author)
- metric: micro-benchmark latency of make_tensor_with_pad; end-to-end throughput (input/output tokens/s and it/s) on qwen-1.5-0.5b
- value: latency 79 -> 22; input throughput 25352.26 -> 30150.97; output throughput 3165.44 -> 3764.60; 24.76 -> 29.44 it/s
- unit: ms (latency); toks/s and it/s (throughput)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
