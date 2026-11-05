# PR #29217: [Model][Qwen3VL] Tune Triton w8a8 block fp8 kernel for L40s

- field: quantization  | area: model_executor_layers
- author: lgeiger
- state: MERGED  created: 2025-11-22T01:24:02Z  merged: 2025-11-22T01:30:55Z
- url: https://github.com/vllm-project/vllm/pull/29217
- labels: qwen

# Issue

## Problem

The Triton w8a8 block-fp8 GEMM kernel selects tile/scheduling parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) from per-shape tuning config files keyed by (N, K, device, dtype, block_shape). For the NVIDIA L40S GPU there were no tuned configs matching the weight shapes of Qwen3-VL-32B-Instruct-FP8, so the kernel fell back to generic default parameters that are suboptimal for the L40S's compute/memory characteristics. This left throughput on the table in the block-fp8 quantized GEMM path.

## Example

Serving Qwen/Qwen3-VL-32B-Instruct-FP8 on a single L40S over ShareGPT text-only: untuned kernels gave 1333.62 tok/s total throughput and 305.45s benchmark duration; tuned configs for shapes like N=10240,K=5120 / N=5120,K=25600 / N=5120,K=8192 / N=51200,K=5120 raised it to 1486.02 tok/s / 274.13s, ~11.4% gain.

## Use case

Online LLM serving of an FP8-quantized vision-language model running text-only inference on a single mid-range datacenter GPU (L40S).

## Resources

GPU tensor/compute throughput and HBM bandwidth on the L40S, exercised by the block-fp8 GEMM kernel; no extra memory cost since only JSON tuning tables are added.

## Type of users

Users serving FP8 w8a8 block-quantized models (e.g. Qwen3-VL-32B-Instruct-FP8) on L40S GPUs.

## Kind of inputs

fp8_w8a8 dtype with block_shape [128,128] at the specific weight matrix dimensions of Qwen3-VL-32B (N/K pairs 10240/5120, 5120/25600, 5120/8192, 51200/5120, etc.) across batch/token M-sizes from 1 up to 4096.

## Mechanism with the issue

vLLM's block-fp8 matmul looks up a per-shape JSON under quantization/utils/configs/ named by N,K,device_name,dtype,block_shape; the dict maps M bucket -> Triton launch params. With no L40S file for these shapes, the lookup missed and the kernel used fallback defaults, so tile sizes and pipeline depth were mismatched to L40S occupancy and latency, slowing the GEMM.

# Solution

## Fix in words

The diff adds new tuning config JSON files for device_name=NVIDIA_L40S, dtype=fp8_w8a8, block_shape=[128,128] at the Qwen3-VL-32B weight shapes (N=10240,K=5120; N=5120,K=25600; N=5120,K=8192; N=51200,K=5120; and more). Each file maps M buckets (1,2,4,...,4096) to L40S-tuned BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages chosen by autotuning on the hardware. At runtime the kernel's config lookup now hits a tuned entry instead of falling back to defaults, selecting better tile shapes and pipeline stages per M-size. Effect: ~11.4% higher total token throughput (1333.62 -> 1486.02 tok/s) on ShareGPT text-only. No kernel logic changes, only data tables.

## Author's solution

Author tuned the Triton w8a8 block-fp8 kernels on a single L40s GPU using the matrix shapes from Qwen3-VL-32B-Instruct-FP8, producing per-shape config tables. This improved ShareGPT text-only throughput by ~11.4%, verified with vllm serve + vllm bench serve before/after benchmarks.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/quantization/utils/configs/N=10240,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/quantization/utils/configs/N=5120,K=25600,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/quantization/utils/configs/N=5120,K=8192,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/quantization/utils/configs/N=51200,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/quantization/utils/configs/N=10240,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/quantization/utils/configs/N=10240,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..6b2c1dc1312b
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/N=10240,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "24": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "32": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "48": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=25600,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=25600,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..b0eaf02a541a
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=25600,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
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
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
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
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    }
+}
diff --git a/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=8192,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=8192,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..4cd357d5086c
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/N=5120,K=8192,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "24": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "512": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    }
+}
diff --git a/vllm/model_executor/layers/quantization/utils/configs/N=51200,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/quantization/utils/configs/N=51200,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..ca2179ddf3d2
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/N=51200,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "256": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 3
+    }
+}

```
