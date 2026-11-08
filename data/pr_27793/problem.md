# Task PR #27793 — [Core][Observability] Add KV cache residency metrics (#27793)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **pytest -v tests/v1/core/test_kv_cache_metrics.py** in `unit` (unit); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- `pytest -v tests/v1/core/test_kv_cache_metrics.py` — `unit`; no per-metric number reported in the PR.

Workload command (run per arm): `pytest -v tests/v1/core/test_kv_cache_metrics.py`
Mechanism / note: Correctness-only; deterministic fake clock. No timing number (overhead ~48 bytes/block, zero cost when off).

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

## Reported numbers (PR author)
- gain: N/A (observability feature; <0.1% overhead, not a speedup) (pct)
- evidence: Assuming typical request latency of 100-500ms, this represents **< 0.1% overhead per request**.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
