# Task PR #37347 — [Perf] Optimize token_embed for pooling models, 1.0% token throughput improvement (#37347)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> When pooling models build the PoolingCursor, build_pooling_cursor always recomputed the per-sequence token offset array from scratch: it allocated a new int64 tensor of length n_seq+1 on CPU, ran torch.cumsum over num_scheduled_tokens, then copied the result to the GPU. This cumulative sum is identical to query_start_loc, which the model runner has already computed for the same forward pass. Recomputing it duplicates CPU work, a tensor allocation, and a host-to-device copy on every pooling step.

### Issue — detailed

> In vllm/v1/pool/metadata.py, build_pooling_cursor unconditionally allocated cumsum = zeros(n_seq+1) on CPU and ran torch.cumsum(num_scheduled_tokens_cpu) then .to(device). The resulting first/last token index tensors equal slices of query_start_loc, which gpu_model_runner already holds (self.query_start_loc.gpu), so the whole recompute-and-copy was wasted duplicate work.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff adds an optional query_start_loc_gpu argument to build_pooling_cursor. Before, cumsum was always computed on CPU and copied to device; after, when query_start_loc_gpu is supplied the function reuses it directly as cumsum (with validation that its length == n_seq+1 and that it sits on the same device as the hidden states), skipping the CPU allocation, cumsum, and H2D copy. gpu_model_runner._pool now passes self.query_start_loc.gpu[:num_reqs+1] into the call. The CPU fallback path is retained for when no precomputed cursor is available, yielding ~1% token throughput improvement.

**Author's words:**

> Reuse the already-computed cumsum (query_start_loc_gpu) instead of recomputing the per-sequence offset prefix sum inside build_pooling_cursor. The author initially also proposed replacing the Python list-comprehension split with a single hidden_states.split(split_sizes), but after review kept only option 2 (cumsum reuse), which alone gave ~1% perf improvement.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **build_pooling_cursor_n128** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **build_pooling_cursor_n16** (us): no number reported in the PR.
- note: PoolingMetadata.build_pooling_cursor per pooling step: PR's solution reuses the runner's GPU query_start_loc (new kwarg), baseline tree rejects the kwarg so the bench calls it the pre-PR way (pinned CPU cumsum + H2D). 128 and 16 sequences x 256 tokens, wall-clock us per call. Author: ~1% token_embed throughput (no absolute numbers) -> reported null.

## Reported numbers (PR author)
- gain: 1% (pct)
- evidence: I benched again using only option 2, we can still get around 1% perf improvement, would you take a look again?
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
