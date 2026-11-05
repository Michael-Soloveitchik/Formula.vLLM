# PR #24266: QWEN3 Coder Fused MoE kernels Optimization configs

- field: gpu_kernels  | area: model_executor_layers
- author: samanamp
- state: MERGED  created: 2025-09-04T17:10:29Z  merged: 2025-09-04T20:33:44Z
- url: https://github.com/vllm-project/vllm/pull/24266
- labels: performance, ready, qwen

# Issue

## Problem

The Triton fused MoE grouped-GEMM kernel selects tile/scheduling parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) from a per-shape tuning config keyed by expert count E, intermediate size N, device, dtype, and block_shape. When no config file matches the running model/GPU, vLLM falls back to generic default heuristics that are not tuned for the actual problem shape, leaving the kernel under-occupied and poorly pipelined. For Qwen3-Coder MoE on FP8 w8a8 with block-wise 128x128 quantization, no matching config existed, so the kernel ran suboptimally across the full batch/token range.

## Example

Qwen3-Coder-480B-A35B-Instruct-FP8 on 4xGPU TP with expert-parallel: MoE shapes E=160,N=640 and E=20,N=2560 (fp8_w8a8, block_shape [128,128]) had no tuned entry, so decode-phase GEMMs for M buckets from 1 up to 4096 tokens used generic tiling. E2E P99 TPOT was 490 ms vs 83 ms after tuning (5.9x).

## Use case

High-concurrency online serving of a large FP8 MoE coder model with long input/output (7500/7500 tokens, 128 concurrent requests), where per-token decode latency dominates throughput.

## Resources

GPU compute (tensor-core/SM occupancy and software-pipeline depth in the Triton GEMM) plus HBM bandwidth for weight/activation tiles; bad tiling wastes both via low occupancy and poor memory reuse.

## Type of users

Users serving large MoE models quantized to FP8 w8a8 with block-wise scaling on NVIDIA B200, GB200, and H100, typically multi-GPU TP with expert parallelism.

## Kind of inputs

FP8 w8a8 MoE workloads with 128x128 block quantization at expert configs E=160,N=640 and E=20,N=2560, across a wide token-count (M) range from single-token decode (M=1) to large prefill batches (M=4096).

## Mechanism with the issue



# Solution

## Fix in words

Diff adds new per-shape MoE tuning config JSON files for the Qwen3-Coder expert shapes (E=160,N=640 and E=20,N=2560, fp8_w8a8, block_shape [128,128]) covering NVIDIA B200, GB200, and H100. Each file maps token-count buckets (1,2,...,4096) to tuned BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages values. Before: these shapes hit untuned generic defaults; after: the kernel loader finds an exact match and launches with shapes-specific tiling and deeper pipelining (e.g. small-M uses BLOCK_SIZE_M=16 with num_stages 3-5; large-M scales to BLOCK_SIZE_M=32/64 with BLOCK_SIZE_K 128/256). Effect: 6.5% higher output token throughput, 17% lower median TPOT, and 5.9x lower P99 TPOT.

## Author's solution

Author generated and committed optimized Triton fused-MoE kernel tuning configs for the Qwen3-Coder expert shapes on B200/GB200/H100, verified via the MoE kernel microbenchmark and an E2E serving benchmark on Qwen3-Coder-480B-A35B-Instruct-FP8, reporting 6.5% output throughput gain, 17% median TPOT reduction, and 5.9x P99 TPOT improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..bdbaf3811c93
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
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
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
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
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..6e17bcd21474
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..aa7610cd75e7
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=160,N=640,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..df920e8b39ba
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "48": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
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
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..e8fe8ea67f24
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "48": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "96": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
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
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..0baf13cb6a5c
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=20,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..4fc4868eaa85
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
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
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..d70adca05e77
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
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
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..0f5867fea5f8
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=40,N=2560,device_name=NVIDIA_H100,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 5
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 4
+    }
+}

```
