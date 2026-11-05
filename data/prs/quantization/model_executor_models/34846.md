# PR #34846: [Perf] Improve default triton fused moe configs

- field: quantization  | area: model_executor/models
- author: mgoin
- state: CLOSED  created: 2026-02-18T22:27:27Z  merged: 2026-02-24T00:01:29Z
- url: https://github.com/vllm-project/vllm/pull/34846
- labels: performance, ready

## Body

<!-- markdownlint-disable -->

## Purpose

Improve the default Triton fused MoE kernel configs when no tuned config file is available. The old defaults were chosen without much consideration: `BLOCK_SIZE_K=32` matched <1% of tuned configs, `num_warps`/`num_stages` were never set, and there was no batch-size awareness. 

New defaults were derived by analyzing patterns across all 298 tuned config files in the repo with Claude Code and discussing. Key changes:
- Batch-size-aware `BLOCK_SIZE_M` (16→32→64→128 as batch grows)
- `BLOCK_SIZE_K=64/128` instead of 32
- Always set `num_warps` and `num_stages` explicitly
- `GROUP_SIZE_M` based on tokens-per-expert ratio
- FP8 block quant path: smaller M tile for decode, batch-aware GROUP_SIZE_M

Many-expert models see up to 2x speedup at large batch size with no regressions.

## Benchmarking

All numbers done on H100 with the new `benchmarks/kernels/benchmark_moe_defaults.py` script that include configs for common models to compare old, new, and tuned config.

Example output:
```
python benchmarks/kernels/benchmark_moe_defaults.py

==========================================================================================
  Mixtral fp8  (E=8, N=7168, K=4096, topk=2)
==========================================================================================
INFO 02-18 18:00:10 [fused_moe.py:1080] Using configuration from /home/mgoin/code/vllm/vllm/model_executor/layers/fused_moe/configs/E=8,N=7168,device_name=NVIDIA_H100_80GB_HBM3,dtype=fp8_w8a8.json for MoE layer.
  Tuned config available: True
   Batch |  Tuned (us) |    Old (us) |    New (us) |  New/Old |  New/Tuned
  ------------------------------------------------------------------------
       1 |      258.77 |      267.13 |      251.48 |    0.94x |       0.97x
       4 |      260.61 |      282.70 |      259.65 |    0.92x |       1.00x
       8 |      267.05 |      283.94 |      263.18 |    0.93x |       0.99x
      16 |      266.13 |      417.76 |      263.93 |    0.63x |       0.99x
      32 |      269.63 |      420.88 |      271.31 |    0.64x |       1.01x
      64 |      274.47 |      429.02 |      280.77 |    0.65x |       1.02x
     128 |      282.78 |      436.25 |      279.23 |    0.64x |       0.99x
     256 |      296.60 |      583.05 |      306.81 |    0.53x |       1.03x
     512 |      350.87 |      892.34 |      464.26 |    0.52x |       1.32x
    1024 |      471.42 |     1799.26 |      528.32 |    0.29x |       1.12x
    2048 |      853.16 |     3226.97 |      899.67 |    0.28x |       1.05x
    4096 |     1572.37 |     6240.76 |     1740.57 |    0.28x |       1.11x
```

### Summary

Every many-expert model is faster with new defaults compare to old at bs>16 (a lower ratio is better):                                                     
                
```
┌─────────────────────┬────────────┬──────────────┬────────────────┐                                                                            
│        Model        │ Batch 1-8  │ Batch 16-128 │ Batch 256-4096 │                                                                            
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ Qwen2-MoE (E=60)    │ ~1.00x     │ 0.82-0.97x   │ 0.59-0.80x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ DeepSeek-V2 (E=64)  │ 0.93-1.01x │ 0.74-1.00x   │ 0.54-0.71x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ OLMoE (E=64)        │ 0.89-1.01x │ 0.79-1.01x   │ 0.54-0.71x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ GLM-4-MoE (E=128)   │ 0.97-1.01x │ 0.91-1.00x   │ 0.56-0.70x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ Qwen3-MoE (E=128)   │ 1.00-1.01x │ 0.97-1.02x   │ 0.59-0.80x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ DeepSeek-V3 (E=256) │ 0.96-1.00x │ 0.95-1.08x   │ 0.49-0.98x     │
├─────────────────────┼────────────┼──────────────┼────────────────┤
│ Qwen3-Next (E=512)  │ 1.01-1.02x │ 0.97-1.01x   │ 0.67-1.00x     │
└─────────────────────┴────────────┴──────────────┴────────────────┘
```

Against tuned configs, new defaults are very close:

```
┌───────────────────────┬─────────────────┬────────────┐
│     Model (tuned)     │ Worst New/Tuned │  Typical   │
├───────────────────────┼─────────────────┼────────────┤
│ DS-V3 tp4 bf16        │ 1.04x           │ 0.95-1.01x │
├───────────────────────┼─────────────────┼────────────┤
│ Qwen3-Next bf16       │ 1.03x           │ 0.96-1.01x │
├───────────────────────┼─────────────────┼────────────┤
│ Qwen3-Next tp2 bf16   │ 1.06x           │ 0.97-1.01x │
├───────────────────────┼─────────────────┼────────────┤
│ E128 N1856 bf16       │ 1.12x           │ 0.93-1.04x │
├───────────────────────┼─────────────────┼────────────┤
│ Qwen3-Next tp2 fp8blk │ 1.11x           │ 0.96-1.03x │
└───────────────────────┴─────────────────┴────────────┘
```

FP8 block quant: neutral to slightly better (no regressions after the GROUP_SIZE_M fix).

## Test Plan

Run existing CI

## Test Result

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
