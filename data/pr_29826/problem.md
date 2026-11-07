# Task PR #29826 — [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor (#29826)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> MinTokensLogitsProcessor.apply writes a Python float (-float("inf")) into a device tensor via indexed assignment. This creates a scalar on the host and triggers a pageable host-to-device (HtoD) memory transfer on every decode step. Because pageable HtoD copies synchronize with the device, they block the CPU until the GPU reaches that point, destroying scheduler/execution overlap.

### Issue — detailed

> In builtin.py MinTokensLogitsProcessor.apply, `logits[self.logits_slice] = -float("inf")` constructs a host scalar each step and copies it to the device via pageable memory. The pageable HtoD transfer is synchronous wrt the device, so under async scheduling the scheduler can no longer stay overlapped with model execution, inflating per-token decode latency.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Before: apply assigned a host-side `-float("inf")` into the device logits via `logits[self.logits_slice] = -float("inf")`, forcing a synchronizing pageable HtoD copy each step. After: the processor preallocates `self.neg_inf_tensor`, a scalar `-inf` tensor already on `self.device` in __init__, and applies it with `logits.index_put_(self.logits_slice, self.neg_inf_tensor)`. This removes the per-step host-to-device transfer and the implied device sync, restoring scheduler/execution overlap under async scheduling. A related MinPLogitsProcessor line was also switched to `logits.masked_fill_(invalid_token_mask, -float("inf"))` for the in-place masked write.

**Author's words:**

> Author identified the pageable HtoD transfer in MinTokensLogitsProcessor as a device-synchronizing operation that breaks scheduler overlap under --async-scheduling, causing up to 30% TPOT increase at low concurrency, backed by nsys traces. Fix avoids the transfer by using a preallocated device tensor for the -inf fill; per a review comment, the min-p path was converted to use `.masked_fill_(-float("inf"))`. Existing MinTokensLogitsProcessor tests deemed sufficient.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **min_tokens_apply** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **min_tokens_apply_after_kernel** (us): no number reported in the PR.
- note: MinTokensLogitsProcessor.apply on [64, 151936] fp32 logits with 64 pending min_tokens requests; baseline = pageable HtoD scalar copy (device sync), PR's solution = index_put_ with a device scalar. Wall-clock us, alone and right after an async filler kernel. PR gives nsys traces only (up to 30% TPOT at low concurrency), no numbers.

## Reported numbers (PR author)
- gain: 30% (pct)
- evidence: At low concurrencies, this leads to a significant (up to 30%) increase in TPOT when including min_tokens in the request.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
