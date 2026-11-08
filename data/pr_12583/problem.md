# Task PR #12583 — Expert Parallelism (EP) Support for DeepSeek V2 (#12583)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **fused_moe_kernel** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **fused_moe_kernel** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 benchmarks/kernels/benchmark_moe.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 --tp-size 2 --dtype auto --seed 0`
Mechanism / note: FusedMoE Triton kernel microbench, GPU-only. PR gives no before/after numbers (design doc).

## The problem (issue)
(from `solution.md`, quoted)

> The current vLLM execution only supports TP when running MoE models.
>
> This PR adds support for Expert Parallelism (EP) for the FusedMoE Kernel and DeepSeek V2 model, which should be extendable to V3 and other MoE models as well.
>
> When VLLM_TEST_ENABLE_EP=1, EP is automatically applied on the FusedMoE layers using the user specified tensor_parallel_size as the expert parallel size.
>
> CUDAgraph works with EP.
>
> [Doc](https://docs.google.com/document/d/1XfJJU-3nO8LZwYhBnKMjy5jLShrjZ0NjbrPdR3GasRs/edit?usp=sharing)

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/12583
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/12583.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `fb5ee05d7363a076f4be62816fa3243ce6799ede1cef6e2cb0eaaf7cf048b90b`
>
> Existing semantic notes in [12583.sem.json](./12583.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: 30% (pct)
- evidence: I noticed that after enabling EP on a 16-card setup, the time taken for the all-reduce kernel has doubled, and the throughput has dropped by 30%.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
