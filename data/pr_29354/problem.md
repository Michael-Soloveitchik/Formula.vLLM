# Task PR #29354 — Add unpermute-aware fused MoE path and small-batch fallback (#29354)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **fused_moe_e128_top1_m1** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **fused_moe_e128_top1_m1** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 33.14 -> 28.19 us.
- **fused_moe_e128_top1_m4** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 49.42 -> 44.02 us.
- **fused_moe_e128_top1_m16** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 136.3 -> 130.5 us.
- **fused_moe_e128_top1_m32** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 234.6 -> 228.2 us.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: fused_experts small-batch naive-block-assignment path: E=128 top_k=1 N=1024 K=5120 bf16 (Maverick TP8 shape), M in 1/4/16/32 (all on the new path), torch softmax+topk routing, CUDA-event us per call. Reported = author's benchmark_moe.py table (H100).

## The problem (issue)
(from `solution.md`, quoted)

> (see solution.md)

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/29354
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/29354.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `9735f0e275eed21d7ce9d3d8b270f53892357b6fd1ba8e202dd3b3a1e594585f`
>
> Existing semantic notes in [29354.sem.json](./29354.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: up to 17.56% faster for small-batch MoE (M=1, Llama-4-Maverick TP8) (pct)
- evidence: faster unpermute MoE path, benchmark speedups
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
