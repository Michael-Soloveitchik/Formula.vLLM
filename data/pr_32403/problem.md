# Task PR #32403 — [Performance] Improve Triton prefill attention kernel's performance  (#32403)

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
> - Original PR: https://github.com/vllm-project/vllm/pull/32403
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/32403.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `673f0aebb1b7e7e9ecff7c034fad5563754369c3fd943ae66aa9be2cc7b1b2d1`
>
> Existing semantic notes in [32403.sem.json](./32403.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **prefill_attn_b8_s2048** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **prefill_attn_b16_s1024** (us): no number reported in the PR.
- **offline_latency_b8_2048** (ms): reported 113.1 -> 110 ms.
- **offline_throughput_b8_2048** (req/s): reported 34 -> 34.97 req/s.
- note: context_attention_fwd (Triton prefill attention) on batch 8 x 2048 and 16 x 1024, 32/8 heads, head 128, bf16, causal; PR = FA2-style exp2 rewrite (-1e8 mask, sm_scale/ln2). CUDA-event us per call. Author numbers are offline e2e (reported-only).

## Reported numbers (PR author)
- gain: — (audit)
- evidence: exp2 fusion speeds Triton prefill kernel
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
