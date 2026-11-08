# Task PR #34846 — [Perf] Improve default triton fused moe configs (#34846)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **fused_moe_e64_top6_m1** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **fused_moe_e64_top6_m1** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **fused_moe_e64_top6_m16** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **fused_moe_e64_top6_m128** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **fused_moe_e64_top6_m1024** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **fused_moe_e64_top6_m4096** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: fused_experts with the arm's get_default_config (untuned GPU: no tuned JSON for RTX 5070 Ti): E=64 N=1408 K=2048 top_k=6 bf16, M in 1/16/128/1024/4096, CUDA-event us. Author gives only New/Old ratios per model (E=64 class: 0.93-1.01x bs1-8, 0.74-1.00x bs16-128, 0.54-0.71x bs256-4096 on H100) and an fp8 Mixtral table -> reported null. If a tuned config exists for the GPU both arms tie (bench prints the config).

## The problem (issue)
(from `solution.md`, quoted)

> (see solution.md)

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/34846
> - Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/34846
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `36493825f747624d223ed36c408067408b1eaad38238bacefbc50accdba8cc5d`
>
> Existing semantic notes in [34846.sem.json](./34846.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: up to 2x at large batch for many-expert MoE (H100) (unit)
- evidence: improved default triton fused-MoE configs
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
