# PR #34580: Flashinfer cuDNN backend for Qwen3 VL ViT attention

- field: gpu_kernels  | area: model_executor/models
- author: maxyanghu
- state: MERGED  created: 2026-02-15T11:36:07Z  merged: 2026-02-27T12:20:23Z
- url: https://github.com/vllm-project/vllm/pull/34580
- labels: performance, ready, ci/build, v1, multi-modality, qwen, nvidia

## Body

## Purpose
Enable by `--mm-encoder-attn-backend=FLASHINFER`

### Details
- Add `FLASHINFER` as a backend for `MMEncoderAttention`
- Add computation for `cu_seqlens` and `sequence_lengths` before calling vision blocks. This is due to cuDNN backend's requirement to pass batch offsets and actual sequence lengths
- Pad `cu_seqlens`, `sequence_lengths` and `max_seqlen` to avoid cuDNN frontend graph recompilation
- Only support Qwen3 VL ViT, Qwen2.5 VL ViT is not supported

## Credit

NVIDIA's MLPerf Qwen3 VL MoE Submission Team @Anerudhan @b-mu @zhandaz @vedaanta @wangshangsam

## Test Plan
```
vllm bench mm-processor --mm-encoder-attn-backend=FLASHINFER --model Qwen/Qwen3-VL-8B-Instruct

tests/kernels/attention/test_mha_attn.py:test_mha_attn_varlen_forward_flashinfer
```

## Test Result
```
vllm bench mm-processor --mm-encoder-attn-backend=FLASHINFER/FLASH_ATTN --num-prompt 1000 --model Qwen/Qwen3-VL-30B-A3B-Instruct
```

#### mm request distribution:
1024 text tokens + 1 image (50% 256×256, 50% 720×1280), then generate 128 tokens.

#### Performance improvement (GB200):
***FLASH_ATTN***
```                                                                                                                                
MM Processor Metrics:
                     Stage  Mean Median   Std P99.0
          get_mm_hashes_ms  0.52   0.88  0.39  0.93
get_cache_missing_items_ms  0.01   0.01  0.00  0.02
     apply_hf_processor_ms 14.56  25.10 12.13 28.46
        merge_mm_kwargs_ms  0.05   0.05  0.02  0.10
   apply_prompt_updates_ms  1.71   2.67  1.07  2.93
     preprocessor_total_ms 16.85  28.79 13.60 32.21
        encoder_forward_ms  7.16   6.47  3.18 16.61
         num_encoder_calls  1.00   1.00  0.00  1.00

Summary: 1000 total encoder calls across 1000 requests.

End-to-End Latency (ms):
Metric Value (ms)
  Mean   25937.46
Median   27018.73
   Std   10737.60
 P99.0   40904.64
```
***FLASHINFER***
```
MM Processor Metrics:                                                              
                     Stage  Mean Median   Std P99.0                                
          get_mm_hashes_ms  0.53   0.88  0.39  0.93                                
get_cache_missing_items_ms  0.01   0.01  0.00  0.02                                
     apply_hf_processor_ms 14.68  25.78 12.24 28.17                                
        merge_mm_kwargs_ms  0.05   0.04  0.02  0.09                                
   apply_prompt_updates_ms  1.72   2.68  1.08  2.96                                
     preprocessor_total_ms 16.99  29.45 13.72 31.97                                
        encoder_forward_ms  5.78   5.16  3.32 13.00                                
         num_encoder_calls  1.00   1.00  0.00  1.00                                

Summary: 1000 total encoder calls across 1000 requests.                                                                                                                

End-to-End Latency (ms):                                                           
Metric Value (ms)                                                                  
  Mean   24701.19                                                                  
Median   25512.37                                                                  
   Std   10516.90                                                                  
 P99.0   40645.69   
```
***mm encoder performance gain = 19.3%***

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

### mergify @ 2026-02-15T14:37:54Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @maxyanghu.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Isotr0py @ 2026-02-21T10:09:05Z

PTAL the failing attention kernels tests.
---
### mergify @ 2026-02-21T16:58:37Z

Hi @maxyanghu, the pre-commit checks have failed. Please run:

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