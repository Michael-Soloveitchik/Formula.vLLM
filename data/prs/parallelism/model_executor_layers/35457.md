# PR #35457: [Model Performance] Add Qwen3MoE tuned MoE configs for H200

- field: quantization  | area: model_executor/layers
- author: chengyinie
- state: MERGED  created: 2026-02-27T00:41:17Z  merged: 2026-02-27T05:51:15Z
- url: https://github.com/vllm-project/vllm/pull/35457
- labels: ready, qwen

## Body

Add tuned Triton kernel configurations for Qwen3MoE (E=128, N=96) on NVIDIA H200 GPU with TP=8.

Configs added:
- E=128,N=96,device_name=NVIDIA_H200.json (BF16)
- E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json (FP8)

Benchmark results show significant improvements:

BF16 (tuned vs default):
- Batch 16-64: 20-24% faster
- Batch 1-8: 2-12% faster

FP8 (tuned vs default):
- Batch 48-128: 8-11% faster
- Batch 1024-2048: 5-6% faster

FP8 vs BF16 (both tuned):
- 10-17% faster at batch sizes 4+

Addresses issue #34256 (Improve MoE Oracle for Qwen3MoE on Hopper)





## Comments

(no discussion comments)