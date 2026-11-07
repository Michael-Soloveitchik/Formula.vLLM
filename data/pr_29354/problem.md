# Task PR #29354 — Add unpermute-aware fused MoE path and small-batch fallback (#29354)

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
> - Original PR: https://github.com/vllm-project/vllm/pull/29354
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/29354.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `9735f0e275eed21d7ce9d3d8b270f53892357b6fd1ba8e202dd3b3a1e594585f`
>
> Existing semantic notes in [29354.sem.json](./29354.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **fused_moe_e128_top1_m1** (us): reported 33.14 -> 28.19 us. workload: `python3 /task/bench.py`
- **fused_moe_e128_top1_m4** (us): reported 49.42 -> 44.02 us.
- **fused_moe_e128_top1_m16** (us): reported 136.3 -> 130.5 us.
- **fused_moe_e128_top1_m32** (us): reported 234.6 -> 228.2 us.
- note: fused_experts small-batch naive-block-assignment path: E=128 top_k=1 N=1024 K=5120 bf16 (Maverick TP8 shape), M in 1/4/16/32 (all on the new path), torch softmax+topk routing, CUDA-event us per call. Reported = author's benchmark_moe.py table (H100).

## Reported numbers (PR author)
- gain: up to 17.56% faster for small-batch MoE (M=1, Llama-4-Maverick TP8) (pct)
- evidence: faster unpermute MoE path, benchmark speedups
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
