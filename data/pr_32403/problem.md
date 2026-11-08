# Task PR #32403 — [Performance] Improve Triton prefill attention kernel's performance  (#32403)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **prefill_attn_b8_s2048** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **prefill_attn_b8_s2048** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **prefill_attn_b16_s1024** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **offline_latency_b8_2048** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 113.1 -> 110 ms.
- **offline_throughput_b8_2048** — `req/s` (requests per second), ↑ higher is better. reported 34 -> 34.97 req/s.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: context_attention_fwd (Triton prefill attention) on batch 8 x 2048 and 16 x 1024, 32/8 heads, head 128, bf16, causal; PR = FA2-style exp2 rewrite (-1e8 mask, sm_scale/ln2). CUDA-event us per call. Author numbers are offline e2e (reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> (see solution.md)

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/32403
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/32403.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `673f0aebb1b7e7e9ecff7c034fad5563754369c3fd943ae66aa9be2cc7b1b2d1`
>
> Existing semantic notes in [32403.sem.json](./32403.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: — (audit)
- evidence: exp2 fusion speeds Triton prefill kernel
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
