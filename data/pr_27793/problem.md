# Task PR #27793 — [Core][Observability] Add KV cache residency metrics (#27793)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> (see solution.md)

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/27793
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/27793.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `84bd388c9a10b534c0821699417aaa4658d43dd72d81fdba1a6c70d4e7025641`
>
> Existing semantic notes in [27793.sem.json](./27793.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- the workload `pytest -v tests/v1/core/test_kv_cache_metrics.py` (unit); no per-metric number in the PR
- note: Correctness-only; deterministic fake clock. No timing number (overhead ~48 bytes/block, zero cost when off).

## Reported numbers (PR author)
- gain: N/A (observability feature; <0.1% overhead, not a speedup) (pct)
- evidence: Assuming typical request latency of 100-500ms, this represents **< 0.1% overhead per request**.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
