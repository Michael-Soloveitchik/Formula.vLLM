# Task PR #34846 — [Perf] Improve default triton fused moe configs (#34846)

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
> - Original PR: https://github.com/vllm-project/vllm/pull/34846
> - Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/34846
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `36493825f747624d223ed36c408067408b1eaad38238bacefbc50accdba8cc5d`
>
> Existing semantic notes in [34846.sem.json](./34846.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **fused_moe_e64_top6_m1** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **fused_moe_e64_top6_m16** (us): no number reported in the PR.
- **fused_moe_e64_top6_m128** (us): no number reported in the PR.
- **fused_moe_e64_top6_m1024** (us): no number reported in the PR.
- **fused_moe_e64_top6_m4096** (us): no number reported in the PR.
- note: fused_experts with the arm's get_default_config (untuned GPU: no tuned JSON for RTX 5070 Ti): E=64 N=1408 K=2048 top_k=6 bf16, M in 1/16/128/1024/4096, CUDA-event us. Author gives only New/Old ratios per model (E=64 class: 0.93-1.01x bs1-8, 0.74-1.00x bs16-128, 0.54-0.71x bs256-4096 on H100) and an fp8 Mixtral table -> reported null. If a tuned config exists for the GPU both arms tie (bench prints the config).

## Reported numbers (PR author)
- gain: up to 2x at large batch for many-expert MoE (H100) (unit)
- evidence: improved default triton fused-MoE configs
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
