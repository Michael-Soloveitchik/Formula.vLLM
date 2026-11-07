# PR #24783: [Perf] Fix DeepGEMM Contiguous Layout Issue, 5.5% Throughput Improvement

- field: quantization  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-09-12T21:41:33Z  merged: 2025-09-14T15:20:17Z
- url: https://github.com/vllm-project/vllm/pull/24783
- labels: performance, ready, deepseek

## Body

## Purpose

We call `get_col_major_tma_aligned_tensor` to get an aligned col-major tensor, but unexpectedly call `contiguous()` and convert it to row major, which causes a lot of perf overhead during runtime.

This PR fixes the issue

## Test

### Perf

`VLLM_USE_DEEP_GEMM=1 vllm bench throughput --model Qwen/Qwen3-30B-A3B-FP8 --load-format dummy --input-len 1000 --output-len 100 --trust_remote_code --enable-expert-parallel`

```bash
# now
Throughput: 44.26 requests/s, 48567.10 total tokens/s, 4426.17 output tokens/s
# main
Throughput: 41.92 requests/s, 45996.43 total tokens/s, 4191.89 output tokens/s
```

### Acc

`VLLM_USE_DEEP_GEMM=1 lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto`

```bash
# now
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8522|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8863|±  |0.0087|

# main
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8522|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8863|±  |0.0087|
```


## Comments

### tlrmchlsmth @ 2025-09-12T23:05:18Z

🕵️
---