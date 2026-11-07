# PR #34674: Adding Nemotron fp8 Triton MoE Config

- field: quantization  | area: model_executor_layers
- author: yugong333
- state: MERGED  created: 2026-02-17T07:03:00Z  merged: 2026-02-24T23:56:38Z
- url: https://github.com/vllm-project/vllm/pull/34674
- labels: 

# Issue

## Problem

vLLM's fused MoE Triton kernel selects tiling parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) from per-(expert-count, intermediate-size, device, dtype) JSON config files. When no tuned config matches a given shape, the kernel falls back to generic default heuristics that are not optimal for that specific GEMM shape, leaving GPU throughput on the table. No tuned config existed for the Nemotron-3-Nano-30B-A3B-FP8 MoE shape (E=128, N=1856) on NVIDIA H200 with fp8_w8a8.

## Example

Running Nemotron-3-Nano-30B-A3B-FP8 (128 experts, intermediate N=1856) on an H200 falls through to default Triton tiling, yielding ~5-7% lower output tokens per second than achievable with shape-tuned block sizes.

## Use case

Serving the FP8-quantized Nemotron MoE model (with a LoRA adapter) on H200 GPUs for inference throughput.

## Resources

GPU compute (Tensor Core utilization) and kernel efficiency of the fused MoE GEMM; poor tiling wastes SM occupancy and memory-pipeline overlap.

## Type of users

Users serving FP8-quantized MoE models (specifically Nemotron-3-Nano-30B-A3B-FP8) on NVIDIA H200 hardware.

## Kind of inputs

fp8_w8a8 MoE layers with 128 experts and intermediate size N=1856, across batch/token-count buckets from 1 up to 4096 tokens per forward.

## Mechanism with the issue

The fused_moe layer looks up a config JSON keyed by E, N, device_name, and dtype; the matching entry maps each token-count bucket (M dimension) to tuned Triton launch params. Absent a file for E=128,N=1856,H200,fp8_w8a8, the kernel uses suboptimal default tiling for every bucket, under-utilizing the H200 for this GEMM shape.

# Solution

## Fix in words

The diff adds a new config file E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json containing Triton (3.6.0) tuned launch parameters for 19 token-count buckets (1 through 4096). Each bucket maps to BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages chosen by offline autotuning: small-M buckets use BLOCK_SIZE_M=16 with GROUP_SIZE_M=1 and 5 stages, while large-M buckets scale up to BLOCK_SIZE_M=128, BLOCK_SIZE_N=256, 8 warps. At runtime the fused MoE kernel now loads these tuned tilings instead of generic defaults, improving GEMM efficiency. Effect: ~5-7% higher output tokens per second for the Nemotron FP8 MoE model on H200.

## Author's solution

Author uploaded a tuned FP8 MoE config for Nemotron-3-Nano-30B-A3B-FP8 (used with a LoRA adapter), validated performance via llmperf, and observed roughly 5-7% improvement in output tokens per second (OTPS).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
new file mode 100644
index 000000000000..f2d518434947
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=1856,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.6.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    }
+}

```
