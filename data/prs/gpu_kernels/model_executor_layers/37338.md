# PR #37338: [Perf] [Bugfix] Fix Triton autotuning in inference for Qwen3.5

- field: gpu_kernels  | area: model_executor/layers
- author: arpera
- state: MERGED  created: 2026-03-17T19:03:23Z  merged: 2026-03-23T07:37:58Z
- url: https://github.com/vllm-project/vllm/pull/37338
- labels: bug, ready, qwen

## Body

## Purpose

Bugfix for PR #36599 + type annotation cleanup: the GDN Triton warmup introduced in that PR uses dummy tensors whose dtypes don't match real inference, so Triton autotuner cache entries created during warmup are never reused — autotuning re-runs on the first inference batch. Additionally, `cu_seqlens` type annotations across FLA ops are corrected from `torch.LongTensor` to `torch.Tensor` to reflect the actual `int32` dtype used at runtime.

### Problem

When serving `Qwen/Qwen3.5-397B-A17B-FP8` with `-dp 8 --enable-expert-parallel`, the first inference batch triggers **746 Triton autotuning events** across 6 FLA kernels (`chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_local_cumsum_scalar_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`, `merge_16x16_to_64x64_inverse_kernel`, `recompute_w_u_fwd_kernel`). This happens because `_warmup_prefill_kernels()` (added in #36599) produces tensors with different dtypes than the real inference path, causing Triton cache key mismatches at serve time.

### Root cause

Three dtype/shape mismatches in `qwen3_next.py` warmup (lines 707–750):

| Tensor | Warmup (before) | Real inference | Fix |
|---|---|---|---|
| `g` (gate) | `torch.randn(..., dtype=bfloat16)` | `fused_gdn_gating()` output → `float32` | Call `fused_gdn_gating()` during warmup |
| `cu_seqlens` | `dtype=torch.long` (int64) | `dtype=torch.int32` | Use `torch.int32` |
| `output_final_state` | `False` | `True` | Set to `True` |

Each mismatch produces a distinct Triton kernel specialization key, so autotuning reruns at inference.

### Changes

1. **`vllm/model_executor/models/qwen3_next.py`** — produce warmup tensors matching inference dtypes:
   - Generate `g, beta` via `fused_gdn_gating()` instead of `torch.randn` (float32 vs bfloat16)
   - Use `dtype=torch.int32` for `cu_seqlens` (matching inference path)
   - Set `output_final_state=True` (matching inference path)

2. **`vllm/model_executor/layers/fla/ops/*.py`** (10 files) — cleanup: change type annotations from `torch.LongTensor` to `torch.Tensor` for `cu_seqlens` parameters. These annotations were inaccurate (`cu_seqlens` is actually `int32` at runtime), and while fixing them has no performance impact, it keeps the code consistent with the dtype change above.

## Test Plan

A/B experiment on 8× NVIDIA B200 (183 GiB each) with `TRITON_PRINT_AUTOTUNING=1`, clean Triton cache (`rm -rf ~/.triton/cache/*`) before each run:

```bash
# Server command (both experiments)
TRITON_PRINT_AUTOTUNING=1 vllm serve "Qwen/Qwen3.5-397B-A17B-FP8" \
    --port 8000 -tp 1 -pp 1 -dp 8 --enable-expert-parallel \
    --language-model-only --reasoning-parser qwen3 \
    --stream-interval 100 --safetensors-load-strategy prefetch

# Benchmark command (both experiments)
vllm bench serve --backend vllm --model Qwen/Qwen3.5-397B-A17B-FP8 \
    --port 8000 --endpoint /v1/completions --dataset-name random \
    --random-input 8000 --random-output 1 --max-concurrency 8 \
    --num-prompt 128 --ignore-eos --temperature 0.0
```

- **Experiment A**: baseline — clear Triton cache, start server, run benchmark
- **Experiment B**: baseline + this patch — clear Triton cache, start server, run benchmark
- Metric: count `Autotuning` lines in server logs before and after `Application startup complete`

## Test Result

### Autotuning distribution

| Phase | **A** (without patch) | **B** (with patch) |
|---|---|---|
| During server startup | 1,162 | 1,162 |
| During benchmark | **746** | **0** |
| **Total** | **1,908** | **1,162** |

With the patch, **all** Triton autotuning completes during server warmup. Zero autotuning events leak into inference.

Example: full autotuning cycle for one kernel from Experiment A, happening **during live inference**:
```
Autotuning kernel chunk_local_cumsum_scalar_kernel with config num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None
Autotuning kernel chunk_local_cumsum_scalar_kernel with config num_warps: 2, num_ctas: 1, num_stages: 3, maxnreg: None
Autotuning kernel chunk_local_cumsum_scalar_kernel with config num_warps: 4, num_ctas: 1, num_stages: 3, maxnreg: None
Autotuning kernel chunk_local_cumsum_scalar_kernel with config num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
Triton autotuning for function chunk_local_cumsum_scalar_kernel,
with key as (1, 64, 64, True, False, 'torch.float32', 'torch.float32', 'torch.int32', 'torch.int32'),
finished after 1.87s,
best config selected: num_warps: 1, num_ctas: 1, num_stages: 3, maxnreg: None;
...
```

### Benchmark performance (128 prompts, 8K input tokens each)

| Metric | **A** (without patch) | **B** (with patch) | Improvement |
|---|---|---|---|
| Duration | 109.73 s | 27.34 s | **4.0× faster** |
| Total token throughput | 9,333 tok/s | 37,459 tok/s | **4.0× higher** |
| Mean TTFT | 6,821 ms | 1,670 ms | **4.1× lower** |
| Median TTFT | 929 ms | 999 ms | ~parity |
| Failed requests | 0 | 0 | — |

The dramatic improvement in mean TTFT and throughput is due to eliminating autotuning stalls during inference. Median TTFT is at parity because median captures the typical (non-stalled) request latency.

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

### vadiklyutiy @ 2026-03-17T21:10:50Z

cc @AuYang261 @ZJY0516 
---
### AuYang261 @ 2026-03-18T09:44:52Z

LGTM.
---
### arpera @ 2026-03-23T07:36:45Z

@sighingnow @ywang96 review this PR please
---