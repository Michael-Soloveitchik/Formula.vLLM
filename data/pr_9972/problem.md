# Task PR #9972 — [V1] Prefix caching (take 2) (#9972)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **input_throughput** (tok/s): reported 1.792e+04 -> 3.226e+04 tok/s. workload: `VLLM_USE_V1=1 python3 benchmarks/benchmark_prefix_caching.py --model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --num-prompts 200 --repeat-count 2 --input-length-range 256:512 --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json --seed 0 --enable-prefix-caching`
- **output_throughput** (tok/s): reported 485 -> 874 tok/s.
- **cost_time** (s): reported 8.49 -> 4.83 s.
- note: Offline batching bench; needs model + ShareGPT dataset. GPU serving run. Numbers are cache-enabled (49% hit).

## Reported numbers (PR author)
- gain: 3x (pct)
- evidence: It is ~3x faster than list matching.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
