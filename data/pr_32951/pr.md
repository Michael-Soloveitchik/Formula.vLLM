# PR #32951: [Async][Spec Decoding] Zero-bubble async scheduling + spec decoding

- field: kv_cache  | area: v1/spec_decode
- author: MatthewBonanni
- state: MERGED  created: 2026-01-23T17:01:50Z  merged: 2026-03-23T19:37:22Z
- url: https://github.com/vllm-project/vllm/pull/32951
- labels: speculative-decoding, ready, v1

## Body

co-authored by @izhuhaoran 

## Purpose
This is a refactor of #29957. It improves the async-ness of spec decoding by optimistically assuming all draft tokens are accepted on the CPU and deferring the correction until after the forward pass. The GPU-side tensors are taken as the source of truth.

The `compute_slot_mappings_kernel` from model runner V2 (plus @yewentao256's changes from #34179) is adapted into V1 here.

## Test Plan
On H100:
```
vllm serve deepseek-ai/DeepSeek-V3.2 \
    -tp 8 -ep \
    --no-enable-prefix-caching \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```
with
```
vllm bench serve \
    --dataset-name spec_bench \
    --dataset-path question.jsonl \
    --spec-bench-output-len 2048 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --seed 42 --ignore-eos \
    --temperature 0 \
    --tokenizer deepseek-ai/DeepSeek-V3
```

## Test Result
TPOT 9.19 ms -> 8.90 ms (3% speedup)

Main

```
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  188.82
Total input tokens:                      801
Total generated tokens:                  20480
Request throughput (req/s):              0.05
Output token throughput (tok/s):         108.46
Peak output token throughput (tok/s):    57.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          112.70
---------------Time to First Token----------------
Mean TTFT (ms):                          62.86
Median TTFT (ms):                        57.16
P99 TTFT (ms):                           105.56
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.19
Median TPOT (ms):                        9.19
P99 TPOT (ms):                           9.51
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.79
Median ITL (ms):                         17.78
P99 ITL (ms):                            18.56
---------------Speculative Decoding---------------
Acceptance rate (%):                     93.56
Acceptance length:                       1.94
Drafts:                                  10578
Draft tokens:                            10578
Accepted tokens:                         9897
Per-position acceptance (%):
  Position 0:                            93.56
==================================================
```

PR:
```
============ Serving Benchmark Result ============
Successful requests:                     10
Failed requests:                         0
Maximum request concurrency:             1
Benchmark duration (s):                  182.79
Total input tokens:                      801
Total generated tokens:                  20480
Request throughput (req/s):              0.05
Output token throughput (tok/s):         112.04
Peak output token throughput (tok/s):    59.00
Peak concurrent requests:                2.00
Total token throughput (tok/s):          116.42
---------------Time to First Token----------------
Mean TTFT (ms):                          53.30
Median TTFT (ms):                        52.90
P99 TTFT (ms):                           80.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.90
Median TPOT (ms):                        8.89
P99 TPOT (ms):                           9.20
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.27
Median ITL (ms):                         17.27
P99 ITL (ms):                            17.90
---------------Speculative Decoding---------------
Acceptance rate (%):                     94.05
Acceptance length:                       1.94
Drafts:                                  10551
Draft tokens:                            10551
Accepted tokens:                         9923
Per-position acceptance (%):
  Position 0:                            94.05
==================================================
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mergify @ 2026-01-23T17:02:38Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### MatthewBonanni @ 2026-01-27T20:53:58Z

Thanks for the reviews! All comments have been addressed. Just trying to track down an apparent acceptance length regression (~10% reduction). When I hack the sampler to force an acceptance rate, I see ~1.7% throughput improvement on Qwen 3 235B. Once I fix the acceptance length issue I'll do more thorough debugging and take the PR out of draft
---
### mergify @ 2026-01-29T22:16:01Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-02-21T04:21:00Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### LucasWilkinson @ 2026-02-27T05:45:07Z

could we do something like: https://github.com/MatthewBonanni/vllm/pull/2 (claude assisted)
---
### mergify @ 2026-03-04T21:39:29Z

Hi @MatthewBonanni, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-03-04T22:03:21Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-05T18:02:21Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-06T05:21:39Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-11T17:06:20Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-20T18:03:53Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-22T08:26:37Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---