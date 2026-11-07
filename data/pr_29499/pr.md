# PR #29499: [EPLB] Optimize EPLB with numpy

- field: parallelism  | area: distributed/eplb
- author: ilmarkov
- state: MERGED  created: 2025-11-26T15:20:48Z  merged: 2026-01-07T20:21:35Z
- url: https://github.com/vllm-project/vllm/pull/29499
- labels: ready

## Body

This PR adds following optimizations in EPLB algorithm which are applicable to sync and async mode:
- Removes multiple gpu-cpu transfers in `move_from_buffer`.
- Implements `get_ep_ranks_with_expert`, `move_to_buffer`, `move_from_buffer` in numpy which gives 20-25% on average for each primitive
- Adds `preserve_intragpu_slots` primitive that does post processing rebalance algo results. It ensures that the experts that are assigned to be moved within the same gpu, stay in their slots. It helps avoid unnecessary gpu memcopies.

Adds `log_balancedness_interval` config parameter to tune the frequency of balancedness logs and possible reduce number of collectives required for this logging.


## Purpose
Improves performance of sync and async eplb. Fixes bug in EPLB config post processing.

## Test Plan
Add tests for grouped layer weight exchange, tests for `preserve_intragpu_slots`. 

## Validation Result
Client 
`lm_eval --model local-completions --tasks gsm8k --model_args model=$model,base_url=http://0.0.0.0:$port/v1/completions,num_concurrent=50,max_retries=3,tokenized_requests=False`

Main
Qwen-30B-FP8 DP=8
```
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 8 -enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 32 --eplb-config.use_async true
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8939|±  |0.0085|
|     |       |strict-match    |     5|exact_match|↑  |0.8870|±  |0.0087|
```


Qwen-30B-FP8 DP=8 AsyncEPLB=True
```
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 8 --max-num-seqs 256 --enable-expert-parallel --port $port --gpu-memory-utilization 0.8 --enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 32 --eplb-config.use_async true
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8848|±  |0.0088|
|     |       |strict-match    |     5|exact_match|↑  |0.8764|±  |0.0091|
```

Qwen-30B-FP8 DP=8 AsyncEPLB=False
```
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 8 --max-num-seqs 256 --enable-expert-parallel  --gpu-memory-utilization 0.8 --enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 32 --eplb-config.use_async false
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8855|±  |0.0088|
|     |       |strict-match    |     5|exact_match|↑  |0.8719|±  |0.0092|
```

---

Main
DeepSeek-V2-Lite DP=4
`vllm serve deepseek-ai/DeepSeek-V2-Lite --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 4 --enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 16 --eplb-config.use_async true`

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3813|±  |0.0134|
|     |       |strict-match    |     5|exact_match|↑  |0.3776|±  |0.0134|
```

DeepSeek-V2-Lite DP=4 AsyncEPLB=True
```
vllm serve deepseek-ai/DeepSeek-V2-Lite --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 4 --max-num-seqs 256 --enable-expert-parallel --gpu-memory-utilization 0.9 --enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 16 --eplb-config.use_async true
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3813|±  |0.0134|
|     |       |strict-match    |     5|exact_match|↑  |0.3768|±  |0.0133|
```


DeepSeek-V2-Lite DP=4 AsyncEPLB=False
```
vllm serve deepseek-ai/DeepSeek-V2-Lite --disable-log-requests --no-enable-prefix-caching -tp 1 -dp 4 --max-num-seqs 256 --enable-expert-parallel --gpu-memory-utilization 0.9 --enable-eplb --eplb-config.window_size 128 --eplb-config.step_interval 512 --eplb-config.num_redundant_experts 16 --eplb-config.use_async false
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3783|±  |0.0134|
|     |       |strict-match    |     5|exact_match|↑  |0.3745|±  |0.0133|
```


## Benchmark results

Server:
```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct --disable-log-requests --no-enable-prefix-caching -dp 8 -tp 1 --max-num-seqs 256 --enable-expert-parallel --gpu-memory-utilization 0.9 --enable-eplb --eplb-config.window_size 32 --eplb-config.step_interval 128 --eplb-config.num_redundant_experts 128 --eplb-config.use_async false --max-model-len 4096
```

We compare sync versions of EPLB. Profile the third EPLB call, to get more stable version.
In profile logs
Main: eplb_state.py: rearrange takes 1s370ms
PR: eplb_state.py: rearrange takes 1s77ms. ~27% speedup

Profile logs of move_to_buffer and move_from_buffer

Main:

<img width="638" height="594" alt="Screenshot 2025-11-26 at 15 31 44" src="https://github.com/user-attachments/assets/7a00c362-533b-48c3-bcb7-8d586f11000a" />



PR:
<img width="597" height="739" alt="Screenshot 2025-11-26 at 15 32 12" src="https://github.com/user-attachments/assets/7d8ca4dd-5a0a-4495-a4af-47c0b1932f3d" />

 
---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### chatgpt-codex-connector @ 2025-11-26T15:20:54Z

Codex usage limits have been reached for code reviews. Please check with the admins of this repo to increase the limits by adding credits.
---
### david6666666 @ 2025-11-27T02:02:31Z

LGTM.
EPLB config post processing were re-written by default values also occur in #29385, thx for fix and optimization.
---
### mergify @ 2025-12-09T11:43:43Z

Hi @ilmarkov, the pre-commit checks have failed. Please run:

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
### mergify @ 2025-12-12T14:49:23Z

Hi @ilmarkov, the pre-commit checks have failed. Please run:

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