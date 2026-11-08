# Task PR #32975 — [Perf] Optimize detokenizer python logic (#32975)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **detokenize** in `us/tok` (us/tok); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **detokenize** (headline) — `us/tok` (us/tok), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Real SlowIncrementalDetokenizer.update() one token at a time with a stop string set (per-update output length check), 8 seqs x 2048 tokens after a 256-token prompt, in-process byte-level BPE tokenizer (no download). us per generated token, median of 5 passes. Author: 1.628x micro ratio only.

## The problem (issue)
(from `solution.md`, quoted)

> In `SlowIncrementalDetokenizer`, the `output_token_ids` property is not a plain list but computes a slice `self.token_ids[self.prompt_len:]` when a prompt is present. Code paths that only needed the count called `len(self.output_token_ids)`, which forced construction of a new sliced list just to measure its length. This wastes CPU and allocates a throwaway list on every detokenization step and every streaming-output decision.

### Issue — detailed

> `output_token_ids` in `SlowIncrementalDetokenizer` returns `self.token_ids[self.prompt_len:]`, a slice. Callers in `detokenizer.py` (`min_tokens` and `stop` checks) and `output_processor.py` (`make_request_output` streaming gate) wrapped this in `len(...)`, so each call allocated and copied a sliced list only to discard it after reading its length.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Diff adds a `num_output_tokens()` method on both detokenizer classes: base returns `len(self.token_ids)`, slow version returns `len(self.token_ids) - self.prompt_len` — computing the count arithmetically without slicing. All callers that previously did `len(self.detokenizer.output_token_ids)` now call `num_output_tokens()` instead (two sites in `detokenizer.py` for `min_tokens`/`stop`, two in `output_processor.py` for the stream-interval gate and `sent_tokens_offset`). This removes per-token list-copy allocations in the hot loop. Also drops the stale `TODO(woosuk)` comment about inefficiency.

**Author's words:**

> Author adds a `num_output_tokens` function to avoid `len(self.output_token_ids)` triggering a slice in `SlowIncrementalDetokenizer`. A second idea (accumulate string pieces and join once instead of repeated string concatenation) was proposed but struck out/dropped. Author notes e2e perf gain is <0.5% (within noise) and provided an AI-generated microbenchmark comparing old vs new `stop_check_offset` logic.

## Reported numbers (PR author)
- gain: 1.628x, 38.57% (pct)
- evidence: ```bash yewentao256@dgx-b200-02:~/vllm-source$ python benchmarks/overheads/benchmark_detokenizer_min_tokens.py old: 787.270 ms (39.36 ns/token) new: 483.633 ms
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
