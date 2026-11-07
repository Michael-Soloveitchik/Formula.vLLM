# PR #32019: [LoRA][Perf] Improve FusedMoE LoRA performance for small rank

- field: gpu_kernels  | area: lora/layers
- author: xyang16
- state: MERGED  created: 2026-01-09T09:52:17Z  merged: 2026-01-10T19:04:19Z
- url: https://github.com/vllm-project/vllm/pull/32019
- labels: ready

## Body

## Purpose

This PR is to improve performance of FusedMoE LoRA for smaller rank
* Currently the default BLOCK_SIZE_N is 64, see [here](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/model_executor/layers/fused_moe/fused_moe.py#L1079)
* For shrink, rank is N. `offs_bn` is computed using mod [here](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/lora/ops/triton_ops/fused_moe_lora_op.py#L136). If N is smaller than BLOCK_SIZE_N, `offs_bn` will be repeated. This will make lora weights being loaded repeated [here](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/lora/ops/triton_ops/fused_moe_lora_op.py#L151-L157), creating much more data loading and computation for `tl.dot`. So setting BLOCK_SIZE_N to N if N is smaller.
* Similarly for expand, rank is K, setting BLOCK_SIZE_K to K to save some data loading and computation.
* e2e output token throughput improves 9% for openai/gpt-oss-20b.

## Test Plan

```
pytest -s -v tests/lora/test_gptoss_tp.py 
```

## Test Result

Tests passed.

## Profiling

Main:

<img width="1275" height="377" alt="Screenshot 2026-01-09 at 1 44 58 AM" src="https://github.com/user-attachments/assets/38196b76-07c4-44f6-8f4d-25573ecd22dc" />

PR:

<img width="1271" height="376" alt="Screenshot 2026-01-09 at 1 45 15 AM" src="https://github.com/user-attachments/assets/4ebc9905-1d13-4ef1-9ea2-e84255e4a541" />

Before the change, for prefill shrink `_fused_moe_lora_kernel` takes around 63µs, expand `_fused_moe_lora_kernel` takes around 49µs. After the change, shrink takes 42µs, expand takes around 37µs.

## Benchmark

```
vllm serve openai/gpt-oss-20b \
  --tensor-parallel-size 1 \
  --max-num-seqs 16 \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8, 16]}' \
  --enable-lora \
  --max-loras 1 \
  --lora-modules lora1=jeeejeee/gpt-oss-20b-lora-adapter-text2sql \
  --max-lora-rank 8 \
  --no-enable-prefix-caching
```

```
vllm bench serve \
  --model openai/gpt-oss-20b \
  --lora-modules lora1 \
  --dataset-name sharegpt \
  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
  --sharegpt-output-len 300 \
  --max-concurrency 16 \
  --num-prompts 1000 \
  --num-warmups 20 \
  --ignore-eos
```

Main:
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  195.62    
Total input tokens:                      226792    
Total generated tokens:                  300000    
Request throughput (req/s):              5.11      
Output token throughput (tok/s):         1533.55   
Peak output token throughput (tok/s):    1680.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2692.88   
---------------Time to First Token----------------
Mean TTFT (ms):                          81.64     
Median TTFT (ms):                        71.15     
P99 TTFT (ms):                           197.38    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.13     
Median TPOT (ms):                        10.15     
P99 TPOT (ms):                           10.64     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.13     
Median ITL (ms):                         9.79      
P99 ITL (ms):                            18.42     
==================================================
```
PR:

```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  179.00    
Total input tokens:                      226792    
Total generated tokens:                  300000    
Request throughput (req/s):              5.59      
Output token throughput (tok/s):         1675.97   
Peak output token throughput (tok/s):    1872.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2942.97   
---------------Time to First Token----------------
Mean TTFT (ms):                          86.89     
Median TTFT (ms):                        86.15     
P99 TTFT (ms):                           183.36    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.22      
Median TPOT (ms):                        9.23      
P99 TPOT (ms):                           9.67      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.22      
Median ITL (ms):                         8.88      
P99 ITL (ms):                            15.48     
==================================================
```

e2e output token throughput improves 9%.

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
> Improves FusedMoE LoRA efficiency at small ranks by making kernel tiling depend on `rank`.
> 
> - Adds `try_get_optimal_moe_lora_config` to wrap `try_get_optimal_moe_config`, overriding `BLOCK_SIZE_N` for shrink and `BLOCK_SIZE_K` for expand using `next_power_of_2(rank)` with bounds
> - Updates `FusedMoEWithLoRA` to use the new helper and pass op-specific `op_type` for `w13`/`w2` shrink/expand
> - Tunes Triton defaults in `get_lora_op_configs` for `fused_moe_lora_*`: shrink `block_n=min(64, next_power_of_2(rank))`; expand `block_k=max(16, min(32, next_power_of_2(rank)))`
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit ae1486be06f965ac70c06419f04dbfa216a20eb8. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Improves FusedMoE LoRA efficiency at small ranks by making kernel tiling depend on `rank` and wiring it through config selection.
> 
> - Adds `try_get_optimal_moe_lora_config` to wrap `try_get_optimal_moe_config`, overriding `BLOCK_SIZE_N` (shrink) and `BLOCK_SIZE_K` (expand) using `next_power_of_2(rank)` with bounds
> - Updates `FusedMoEWithLoRA` to use the new helper and pass op-specific `op_type` for `w13`/`w2` shrink/expand when tuned configs are absent
> - Tunes Triton defaults in `get_lora_op_configs` for `fused_moe_lora_*`: shrink `block_n=min(64, next_power_of_2(rank))`; expand `block_k=max(16, min(32, next_power_of_2(rank)))`
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 01e6e26176336b1bc23296cb62cd3f69066c19a5. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

### dcmaddix @ 2026-01-09T22:54:18Z

thanks a lot @xyang16 ! Nice deep dive and improvement.
---
### mergify @ 2026-01-10T02:39:54Z

Hi @xyang16, the pre-commit checks have failed. Please run:

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
### mergify @ 2026-01-10T04:34:24Z

Hi @xyang16, the pre-commit checks have failed. Please run:

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
### mergify @ 2026-01-10T04:40:59Z

Hi @xyang16, the pre-commit checks have failed. Please run:

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