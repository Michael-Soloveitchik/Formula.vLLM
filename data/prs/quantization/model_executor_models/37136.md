# PR #37136: [Performance][Model Loader] Skip non-local expert weights during EP model loading

- field: quantization  | area: model_executor/models
- author: esmeetu
- state: CLOSED  created: 2026-03-16T03:08:24Z  merged: 2026-03-16T08:33:36Z
- url: https://github.com/vllm-project/vllm/pull/37136
- labels: ready

## Body


## Purpose
In DP+EP deployments, every rank currently reads **all** expert weights from disk via `safe_open().get_tensor()`, only for `FusedMoE.weight_loader` to discard non-local experts afterward. For large MoE models (e.g. Kimi-K2.5 with 384 experts at 591GB), each rank reads the full 591GB but only keeps ~144GB (dense + 1/8 experts).

This PR moves the filtering **before** the disk read by checking the tensor name against the local expert set in the weight iterator, so `f.get_tensor()` is never called for non-local experts.

- Skip reading non-local expert weights from disk during model loading when expert parallelism (EP) is enabled
- Each rank only reads its own expert shard + shared dense weights, avoiding ~87% of storage I/O for typical MoE models
- No change to non-EP or non-MoE loading paths

## Test Plan

- [x] Verify on MoE model with EP enabled: only local expert weights loaded, model output unchanged
- [x] Verify on 3D MoE model: gpt-oss-120b
- [x] Verify non-MoE model loading is unaffected (`local_expert_ids` remains `None`)
- [x] Verify EP=1 (no filtering) produces identical results to baseline
- [x] Test with different EP sizes (8, 16, 24) to confirm correct expert distribution


 ## Benchmark Results

  Kimi-K2.5-NVFP4 (591GB, 384 experts):

  ### DP/EP=4 (1 node × 4 GPUs)

  | Metric | main | PR | Speedup |
  |---|---:|---:|---:|
  | Loading Weights | 96.58s | **67.59s** | **1.4x** |
  | Model Loading Total | 103.53s | **72.89s** | **1.4x** |

  ### DP/EP=8 (2 nodes × 4 GPUs)

  | Metric | main | PR | Speedup |
  |---|---:|---:|---:|
  | Loading Weights | 58.45s | **25.35s** | **2.3x** |
  | Model Loading Total | 63.42s | **32.04s** | **2.0x** |

  ### DP/EP=16 (4 nodes × 4 GPUs)

  | Metric | main | PR | Speedup |
  |---|---:|---:|---:|
  | Loading Weights | 54.84s | **21.71s** | **2.5x** |
  | Model Loading Total | 83.46s | **39.71s** | **2.2x** |

> **Note on warm vs cold cache:** The numbers above were measured with OS page cache already populated (repeated runs). On cold start (first load after boot, deployment, or scale-up), the speedup is expected to be
   **larger** because the dominant cost shifts from CPU-side mmap fault handling to network filesystem I/O, where reducing read volume from 591GB to ~144GB per rank (EP=8) has a proportionally greater effect. Cold
  cache testing was not performed but can be reproduced by running `sync && echo 3 > /proc/sys/vm/drop_caches` (requires root) before the first load.

  Speedup scales with EP size: higher EP → more experts skipped → greater I/O reduction.

| EP size | Experts/rank | Per-rank I/O | Expert I/O skipped |
  |---|---|---|---|
  | EP=4 | 96 | ~208GB | 75% |
  | EP=8 | 48 | ~144GB | 87.5% |
  | EP=16 | 24 | ~112GB | 93.75% |
  | EP=24 | 16 | ~101GB | 95.8% |

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

### mergify[bot] @ 2026-03-16T03:12:05Z

Hi @esmeetu, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>

### mergify[bot] @ 2026-03-16T03:54:13Z

Hi @esmeetu, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
