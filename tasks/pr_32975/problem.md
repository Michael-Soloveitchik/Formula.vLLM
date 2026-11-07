# Task PR #32975 — [Perf] Optimize detokenizer python logic (#32975)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (the PR's solution)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
A text-decoding component repeatedly counts how many output tokens have been produced so far. For one variant, answering that count built a copy of the token list just to measure its length. Copying a list only to take its length wastes time on a hot path that runs every output step.

### Issue — detailed
In vllm/v1/engine/detokenizer.py, callers computed the output-token count via len(self.output_token_ids). For SlowIncrementalDetokenizer, output_token_ids is a property returning self.token_ids[self.prompt_len:] when a prompt exists, so each call allocates a new sliced list only to take its length and discard the slice. This ran inside the per-token update() loop (min_tokens and stop-string checks) and in output_processor.py make_request_output() (stream_interval check and sent_tokens_offset update), so the slice recurred on the streaming output path. The slice is pure overhead because the count is derivable from stored fields.

## Expert solution (reference — the merged PR)
**General:** Add a dedicated method that returns the output-token count directly from stored values, without building any intermediate list. Switch the length-only call sites to use it. Same result, less work.

**Detailed:** Added num_output_tokens() to the detokenizer classes: the baseline class returns len(self.token_ids), and SlowIncrementalDetokenizer returns len(self.token_ids) - self.prompt_len, avoiding the self.token_ids[self.prompt_len:] slice. In detokenizer.py update(), both guards (len(self.output_token_ids) <= self.min_tokens and len(self.output_token_ids) > self.min_tokens) now call self.num_output_tokens(). In output_processor.py make_request_output(), the stream_interval check and the sent_tokens_offset assignment now call self.detokenizer.num_output_tokens() instead of len(...output_token_ids). A stale TODO about update() being inefficient when new_token_ids has more than one element was removed. The slicing output_token_ids property is kept for callers that need the actual list.

**Author's words:** Adding a num_output_tokens function, to avoid len(self.output_token_ids) might do a slice in SlowIncrementalDetokenizer. A second proposed change (accumulate pieces and join once), and a later spec-decoding fast/slow path split, were both reverted after review; only the num_output_tokens change was kept.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **detokenize** (us/tok): reported 1.628 -> 1.0 us/tok. workload: per tok

## Reported numbers (PR author)
- metric: End-to-end performance effect of the merged num_output_tokens change; author reports it below measurement noise.
- value: <0.5
- unit: %

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
