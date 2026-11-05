# PR #29627: [Attention] Cache attention metadata builds across hybrid KV-cache groups

- field: attention  | area: v1/attention
- author: LucasWilkinson
- state: MERGED  created: 2025-11-27T21:56:41Z  merged: 2025-12-16T22:10:16Z
- url: https://github.com/vllm-project/vllm/pull/29627
- labels: ready, v1

## Body

Scaled back version of: https://github.com/vllm-project/vllm/pull/22788 and generalization of: https://github.com/vllm-project/vllm/pull/29444

Depends on https://github.com/vllm-project/vllm/pull/29628 land that first

## Test Plan

Decode-heavy workload with APC enabled => **latency reduction through this PR: 13.7%**

```
vllm bench latency --model ibm-granite/granite-4.0-tiny-preview --input-len 128  --output-len 2048 \
    --batch-size 8 --enable-prefix-caching --num-iters 10 --num-iters-warmup 2
```

Decode-heavy workload with APC disabled => **latency reduction through this PR: 2.4%**

```
vllm bench latency --model ibm-granite/granite-4.0-tiny-preview --input-len 128  --output-len 2048 \
    --batch-size 8 --num-iters 10 --num-iters-warmup 2
```

## Test Result

### APC enabled:

Before:
```
Avg latency: 17.656723083648830 seconds
10% percentile latency: 17.629154206998646 seconds
25% percentile latency: 17.638833652948960 seconds
50% percentile latency: 17.659772212151438 seconds
75% percentile latency: 17.681711453711614 seconds
90% percentile latency: 17.693322991393508 seconds
99% percentile latency: 17.694964875970037 seconds
```

This PR:
```
Avg latency: 15.236061835102737 seconds
10% percentile latency: 15.214241081103683 seconds
25% percentile latency: 15.218232876388356 seconds
50% percentile latency: 15.237253282219172 seconds
75% percentile latency: 15.245128114242107 seconds
90% percentile latency: 15.250385893788188 seconds
99% percentile latency: 15.278700626296922 seconds
```

### APC disabled:

Before:
```
Avg latency: 14.836663954611868 seconds
10% percentile latency: 14.762344243377447 seconds
25% percentile latency: 14.770147853763774 seconds
50% percentile latency: 14.822804148774594 seconds
75% percentile latency: 14.853949831798673 seconds
90% percentile latency: 14.937904387619346 seconds
99% percentile latency: 15.010869848066940 seconds
```

This PR:
```
Avg latency: 14.485000116657465 seconds
10% percentile latency: 14.469038359075785 seconds
25% percentile latency: 14.475818350212649 seconds
50% percentile latency: 14.479432722553610 seconds
75% percentile latency: 14.504760635318235 seconds
90% percentile latency: 14.514666751492769 seconds
99% percentile latency: 14.517034560767934 seconds
```

Co-authored by: @s3woz (original PR: https://github.com/vllm-project/vllm/pull/29444)

## Comments

### mergify @ 2025-11-27T21:57:47Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @LucasWilkinson.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-12-10T19:57:39Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @LucasWilkinson.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-12-11T05:35:58Z

Hi @LucasWilkinson, the pre-commit checks have failed. Please run:

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
### mergify @ 2025-12-12T04:29:19Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @LucasWilkinson.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-12-12T04:48:28Z

Hi @LucasWilkinson, the pre-commit checks have failed. Please run:

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
### Josephasafg @ 2025-12-18T08:04:06Z

@LucasWilkinson is there a reason not to add this support for Mamba1 in `mamba1_attn.py`?

I have this [PR](https://github.com/vllm-project/vllm/pull/28133) which should consolidate most of the common logic between mamba1 and mamba2 so features like that won't be overlooked. I can add this logic to `mamba_attn.py` in my opened PR, so both models can enjoy this. @tdoublep what do you think?
---
### tdoublep @ 2025-12-18T11:29:14Z

@Josephasafg Yes, fully agreed on that. Will prioritize reviewing + merging your cleanup PR. 
---
### Josephasafg @ 2025-12-18T19:17:58Z

One interesting observation I made while adding `supports_update_block_table` to both Mamba models in my [PR](https://github.com/vllm-project/vllm/pull/28133)￼ is that the tests for Mamba1 hybrid models using `ai21labs/Jamba-tiny-dev` started failing.

After some debugging, I noticed that the Mamba2 model used in the hybrid APC tests is `hmellor/tiny-random-BambaForCausalLM`, and for this model the test passes. Looking deeper, it turns out this model has only a single Mamba layer and a single attention layer, so `builder.update_block_table` is never invoked.

When I replaced the model in the APC test with `ibm-granite/granite-4.0-tiny-preview`, the test failed with the same issue observed in Jamba, specifically a logprobs mismatch.

@LucasWilkinson  do you have any insight into why this might be happening?

cc: @tdoublep 
---
### LucasWilkinson @ 2025-12-18T19:22:57Z

> One interesting observation I made while adding `supports_update_block_table` to both Mamba models in my [PR](https://github.com/vllm-project/vllm/pull/28133)￼ is that the tests for Mamba1 hybrid models using `ai21labs/Jamba-tiny-dev` started failing.
> 
> After some debugging, I noticed that the Mamba2 model used in the hybrid APC tests is `hmellor/tiny-random-BambaForCausalLM`, and for this model the test passes. Looking deeper, it turns out this model has only a single Mamba layer and a single attention layer, so `builder.update_block_table` is never invoked.
> 
> When I replaced the model in the APC test with `ibm-granite/granite-4.0-tiny-preview`, the test failed with the same issue observed in Jamba, specifically a logprobs mismatch.
> 
> @LucasWilkinson do you have any insight into why this might be happening?
> 
> cc: @tdoublep

Can you give me a repro command? happy to help out
---
### Josephasafg @ 2025-12-18T19:27:46Z

@LucasWilkinson Thanks.
You can run this test 
`pytest tests/models/language/generation/test_hybrid.py::test_apc_single_prompt[1-5-2-64-ibm-granite/granite-4.0-tiny-preview]` but you would need to add `HYBRID_MODELS[4]` to list of allowed models in the test's [parameterize](https://github.com/vllm-project/vllm/blob/main/tests/models/language/generation/test_hybrid.py#L390)
---
### Josephasafg @ 2025-12-20T19:38:22Z

@LucasWilkinson have you managed to reproduce it?
---