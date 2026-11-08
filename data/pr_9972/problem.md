# Task PR #9972 — [V1] Prefix caching (take 2) (#9972)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **input_throughput** in `tok/s` (tokens per second); maximize it (higher is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **input_throughput** (headline) — `tok/s` (tokens per second), ↑ higher is better. reported 1.792e+04 -> 3.226e+04 tok/s.
- **output_throughput** — `tok/s` (tokens per second), ↑ higher is better. reported 485 -> 874 tok/s.
- **cost_time** — `s` (seconds), ↓ lower is better. reported 8.49 -> 4.83 s.

Workload command (run per arm): `VLLM_USE_V1=1 python3 benchmarks/benchmark_prefix_caching.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --num-prompts 200 --repeat-count 2 --input-length-range 256:512 --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --seed 0 --enable-prefix-caching`
Mechanism / note: Offline batching bench; needs model + ShareGPT dataset. GPU serving run. Numbers are cache-enabled (49% hit).

## The problem (issue)
(from `solution.md`, quoted)

> This PR adds prefix caching to V1 (take 2). Take 1 is in #9668.
> The main difference in take 2 is we adopt a custom doubly linked list to operate free blocks with eviction. This doubly linked list has the following features over the Python builtin `deque`:
> - It supports `.remove()` operator in O(1) time.
> - It does not allocate any new Python objects, but directly manipulates the given objects with pointers.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/9972
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/9972.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `73b3c6877f32ebcf554ec496b65500d5a39013756b36f9352e44fa9203733803`
>
> Existing semantic notes in [9972.sem.json](./9972.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: 3x (pct)
- evidence: It is ~3x faster than list matching.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
