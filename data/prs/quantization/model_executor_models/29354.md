# PR #29354: Add unpermute-aware fused MoE path and small-batch fallback

- field: quantization  | area: model_executor/models
- author: RunkaiTao
- state: CLOSED  created: 2025-11-24T22:36:03Z  merged: 2026-01-09T19:58:39Z
- url: https://github.com/vllm-project/vllm/pull/29354
- labels: performance, ready

## Body


## Purpose
This PR enhances the fused MoE implementation by adding support for the unpermute execution path. It introduces a small-batch fallback path designed for models with large numbers of experts, where the regular kernel becomes inefficient. For these regimes, the new path provides a speed-up by reducing unnecessary data movement and improving kernel efficiency for low-token workloads.

## Test Result

### Benchmark: Llama-4-Maverick-17B-128E (TP = 8)

Command:
```bash
python3 benchmarks/kernels/benchmark_moe.py \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --tp-size 8
```

Results:
| M  | with permute (µs) | without permute (µs) | speed-up |
|----|-------------------|-----------------------|----------|
| 1  | 33.14             | 28.19                 | **-17.56%** |
| 2  | 36.05             | 30.70                 | **-17.43%** |
| 4  | 49.42             | 44.02                 | **-12.27%** |
| 8  | 76.77             | 71.41                 | **-7.51%**  |
| 16 | 136.29            | 130.47                | **-4.46%**  |
| 24 | 183.81            | 175.97                | **-4.46%**  |
| 32 | 234.63            | 228.19                | **-2.82%**  |
| 48 | 309.59            | 309.73                | 0.05%    |
| 64 | 382.59            | 382.68                | 0.02%    |
| 96 | 513.39            | 512.00                | -0.27%   |

### Benchmark: DeepSeek-R1 (TP = 8)

Command:
```bash
python3 benchmarks/kernels/benchmark_moe.py \
  --model deepseek-ai/DeepSeek-R1 \
  --tp-size 8
```

Results:
| M  | with permute (µs) | without permute (µs) | speed-up |
|----|-------------------|-----------------------|----------|
| 1  | 70.69             | 65.55                 | **-7.84%**  |
| 2  | 83.58            | 78.20                 | **-6.87%** |
| 4  | 112.80            | 108.65                | **-3.8%** |
| 8  | 171.68            | 172.13                | 0.2%          |
| 16 | 338.09            | 338.50                | 0.12%       |

### Benchmark: Qwen3-Coder-480B-A35B-Instruct-FP8 (TP = 8)

Command:
```bash
python3 benchmarks/kernels/benchmark_moe.py \
  --model Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
  --tp-size 8
```

Results
| M  | with permute (µs) | without permute (µs) | speed-up |
|----|-------------------|-----------------------|----------|
| 1  | 69.76            | 64.48                 | **-8.18%**  |
| 2  | 85.90            | 80.07                 | **-7.28%** |
| 4  | 116.18            | 112.25                | **-3.5%** |
| 8  | 176.28            |176.36                 | 0.004%  |
| 16 | 340.01            | 343.42                | -0.01%     |

### Benchmark: GPT-OSS-120B (TP = 4)

Command:
```bash
python3 benchmarks/kernels/benchmark_moe.py \
  --model openai/gpt-oss-120b \
  --tp-size 4
```

Results:
| M  | with permute (µs) | without permute (µs) | speed-up |
|----|-------------------|-----------------------|----------|
| 1  | 49.55             | 44.96                 | **-10.21%** |
| 2  | 68.18             | 63.54                 | **-7.30%**  |
| 4  | 120.50            | 116.96                | **-3.03%**  |
| 8  | 227.08            | 227.45                | 0.16%       |
| 16 | 322.53            | 322.60                | 0.02%       |



Accuracy:
Command:
```bash
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mixtral-8x7B-Instruct-v0.1 --quantization fp8 --tensor-parallel-size 8 --max-model-len 32768 --host 0.0.0.0 --port 8000 --max-num-batched-tokens 512 --max-num-seqs 1
```
```bash
python3 tests/evals/gsm8k/gsm8k_eval.py --port 8000 --num-questions 100 --num-shots 5 --max-tokens 1000 --temperature 0.0 --seed 42
```
before:
```
Accuracy: 0.620
Invalid responses: 0.000
Total latency: 51.251 s
Questions per second: 1.951
Total output tokens: 11623
Output tokens per second: 226.787
```
after:
```
Accuracy: 0.620
Invalid responses: 0.000
Total latency: 50.132 s
Questions per second: 1.995
Total output tokens: 11623
Output tokens per second: 231.849
```
---

<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>

---

> [!NOTE]
> Introduces an unpermute-aware fast path for small-token, many-expert regimes.
> 
> - Kernel: add `naive_block_assignment` mode to `fused_moe_kernel`; permit `sorted_token_ids` to be `None`, adjust token offsets and EM computation, and plumb through `invoke_fused_moe_triton_kernel`/`dispatch_fused_moe_kernel` (asserts and grids updated)
> - Execution: heuristic selects naive path when `tokens_in_chunk * top_k` is sparse vs `global_num_experts` (and not block-quantized), bypassing `moe_align_block_size`
> - Tests: add `test_naive_block_assignment_moe` plus small-M/top-k and large-expert configs (`NUM_EXPERTS_LARGE`, `TOP_KS_SMALL`, `FUSED_MOE_MNK_FACTORS_SMALL_M`)
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit cd7ae7c2e9c25764515a3df407d023e6172c8425. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

### chatgpt-codex-connector[bot] @ 2025-12-01T18:44:24Z

Codex usage limits have been reached for code reviews. Please check with the admins of this repo to increase the limits by adding credits.

### varun-sundar-rabindranath @ 2025-12-03T17:17:53Z

@RunkaiTao  can you add some tests for the new code-path.  Thanks 🙌 

edit: Can you also test the expert_map code-path when `use_unpermute` is enabled. 

### heheda12345 @ 2025-12-03T18:33:43Z

CC @tlrmchlsmth 

### mergify[bot] @ 2025-12-05T13:05:40Z

Hi @RunkaiTao, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

### mergify[bot] @ 2025-12-08T03:27:52Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @RunkaiTao.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

### varun-sundar-rabindranath @ 2025-12-11T06:20:13Z

Thanks @RunkaiTao . LGTM ! It'd be great if we could get inputs from an additional reviewer. Thanks! 
cc @bnellnm @mgoin 

### varun-sundar-rabindranath @ 2025-12-17T13:08:46Z

Thank you @RunkaiTao . Nice performance boost. Can you add a unit test that exercises `naive_block_assignment` code-path. 

### dcmaddix @ 2025-12-18T21:35:04Z

> LGTM! Thanks @RunkaiTao

Thanks a lot @varun-sundar-rabindranath and great work @RunkaiTao!

### varun-sundar-rabindranath @ 2026-01-06T18:50:33Z

@mgoin @bnellnm can you take a look please. Thanks 🙌 

