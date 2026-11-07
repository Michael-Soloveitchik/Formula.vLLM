# PR #31722: [PERF] Speed-up of GDN attention decode part (Qwen3-Next)

- field: attention  | area: model_executor/layers
- author: vadiklyutiy
- state: MERGED  created: 2026-01-05T12:37:58Z  merged: 2026-01-06T17:32:47Z
- url: https://github.com/vllm-project/vllm/pull/31722
- labels: performance, ready, qwen

## Body

Speed Up GDN Attention (Decode) - `fused_recurrent_gated_delta_rule_fwd`

# Benchmarks

## H200 — `fused_recurrent_gated_delta_rule_fwd` with shapes from Qwen3-Next

### Before

| Batch Size | TP=4 (μs) | TP=2 (μs) |
|-----------:|----------:|----------:|
| 32   | 19.87 ± 0.08 | 30.14 ± 0.02 |
| 128  | 52.06 ± 0.02 | 93.38 ± 0.02 |
| 256  | 94.24 ± 0.02 | 176.10 ± 0.02 |
| 512  | 177.73 ± 0.03 | 341.63 ± 0.03 |
| 1024 | 345.12 ± 0.20 | 672.48 ± 0.05 |

### After

| Batch Size | TP=4 (μs) | TP=2 (μs) |
|-----------:|----------:|----------:|
| 32   | 13.60 ± 0.07 | 20.29 ± 0.02 |
| 128  | 33.76 ± 0.02 | 59.17 ± 0.02 |
| 256  | 60.29 ± 0.02 | 111.14 ± 0.36 |
| 512  | 113.28 ± 0.05 | 212.70 ± 0.11 |
| 1024 | 215.12 ± 0.13 | 420.22 ± 0.20 |

## B200 — `fused_recurrent_gated_delta_rule_fwd` with shapes from Qwen3-Next

### Before

| Batch Size | TP=4 (μs) | TP=2 (μs) |
|-----------:|----------:|----------:|
| 32   | 12.26 ± 0.02 | 19.23 ± 0.01 |
| 128  | 34.91 ± 0.01 | 63.58 ± 0.01 |
| 256  | 64.80 ± 0.01 | 122.30 ± 0.01 |
| 512  | 124.64 ± 0.02 | 239.23 ± 0.02 |
| 1024 | 244.60 ± 0.02 | 473.53 ± 0.05 |

### After

| Batch Size | TP=4 (μs) | TP=2 (μs) |
|-----------:|----------:|----------:|
| 32   | 8.22 ± 0.02 | 14.11 ± 0.01 |
| 128  | 24.54 ± 0.01 | 42.21 ± 0.01 |
| 256  | 42.11 ± 0.02 | 80.29 ± 0.02 |
| 512  | 80.80 ± 0.03 | 156.64 ± 0.06 |
| 1024 | 154.94 ± 0.07 | 305.21 ± 0.21 |

## End-to-End Decode

Server Launch

```bash
VLLM_USE_FLASHINFER_MOE_FP8=1 \
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  -tp 2 \
  --enable-expert-parallel \
  --async-scheduling \
  --no-enable-prefix-caching \
  --compilation_config.max_cudagraph_capture_size 2048
```

Benchmark Command

```bash
vllm bench serve \
  --backend vllm \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input 32 \
  --random-output 1000 \
  --max-concurrency 1024 \
  --num-prompt 1024 \
  --ignore-eos
```

### Results

**Before**

- Output token throughput: 23891 tok/s

**After**

- Output token throughput: 27904 tok/s

**Speedup: 16.8%**


## Comments

### vadiklyutiy @ 2026-01-05T12:43:23Z

cc @ZJY0516, @sighingnow
---
### vadiklyutiy @ 2026-01-06T15:14:54Z

Seems `multi-modal-processor-test-cpu` fails on top of the tree

<img width="2450" height="472" alt="image" src="https://github.com/user-attachments/assets/19b8b058-52d6-4ac6-875a-dc005e275fb3" />

---