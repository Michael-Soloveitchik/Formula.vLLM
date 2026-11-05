# PR #23504: [Perf] Add Triton config for DeepSeek V3 FP8 EP32 H200

- field: quantization  | area: model_executor_layers
- author: minosfuture
- state: MERGED  created: 2025-08-24T18:31:47Z  merged: 2025-08-25T01:06:35Z
- url: https://github.com/vllm-project/vllm/pull/23504
- labels: performance, ready, deepseek

# Issue

## Problem

vLLM ships per-shape Triton kernel autotuning configs for FP8 W8A8 block-quantized GEMMs and fused MoE kernels, keyed by matrix dims, device name, dtype, and block_shape. When a config file for a given (N,K)/(E,N) shape on a specific GPU is missing, the kernel falls back to a generic default tiling. That default is not tuned for the actual problem geometry, so the block-FP8 matmul and fused MoE kernels run with suboptimal BLOCK_SIZE/GROUP_SIZE/num_warps/num_stages, leaving GPU throughput on the table.

## Example

Serving DeepSeek V3-0324 in FP8 on H200 under DP8TP4EP32 emits warnings for missing configs at N=2112,K=7168 (block-FP8 GEMM) and E=8,N=2048 (fused MoE), both block_shape=[128,128], forcing the default-config code path.

## Use case

High-concurrency online serving of a large FP8 MoE model (DeepSeek V3) across many GPUs with expert parallelism.

## Resources

GPU compute (Tensor Core / SM utilization in the FP8 GEMM and MoE kernels); poor tiling wastes compute and memory-bandwidth efficiency.

## Type of users

Users serving FP8 block-quantized MoE models on H200 with multi-GPU TP + expert-parallel (EP32) deployments.

## Kind of inputs

FP8 w8a8 block-quantized weights with block_shape [128,128], GEMM shapes N=2112,K=7168 and MoE E=8,N=2048, across a wide range of token/batch counts (M from 1 to 8192).

## Mechanism with the issue

fp8_utils.py (block-FP8 matmul config loader, warning at line 581) and fused_moe.py (MoE config loader, warning at line 727) look up a JSON config file by shape+device+dtype+block_shape; a missing file triggers the 'Using default ... kernel config. Performance might be sub-optimal!' path, applying untuned Triton launch params instead of hardware-specific ones.

# Solution

## Fix in words

Adds two new tuned Triton config JSON files for H200: a fused-MoE config at E=8,N=2048,dtype=fp8_w8a8,block_shape=[128,128] with per-M-bucket tilings (M=1..8192), and a block-FP8 GEMM config at N=2112,K=7168,dtype=fp8_w8a8,block_shape=[128,128] with entries for M=2048/3072/4096. With these files present, the config loaders in fp8_utils.py and fused_moe.py find a match and apply the tuned BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages instead of the generic default, silencing the sub-optimal-config warnings. Also adds a README documenting that configs are generated via benchmarks/kernels/ scripts, and fixes the tqdm import in benchmark_w8a8_block_fp8.py (from `import tqdm` to `from tqdm import tqdm`).

## Author's solution

Add the optimal tuned Triton config for DeepSeek V3 under the DP8TP4EP32 deployment on H200, which resolves the two missing-config warnings (block-FP8 N=2112,K=7168 and fused MoE E=8,N=2048). Validated by benchmarking DeepSeek V3-0324 serving at max-concurrency 2048, 4096 prompts, 2048/1024 input/output len; tuned vs untuned showed comparable-to-slightly-better throughput (16872 vs 16776 tok/s total) and TPOT.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `benchmarks/kernels/benchmark_w8a8_block_fp8.py`
- `vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/quantization/utils/configs/README.md`

### Diff
```diff
diff --git a/benchmarks/kernels/benchmark_w8a8_block_fp8.py b/benchmarks/kernels/benchmark_w8a8_block_fp8.py
index 4fcdbadd65ec..e648a91077fd 100644
--- a/benchmarks/kernels/benchmark_w8a8_block_fp8.py
+++ b/benchmarks/kernels/benchmark_w8a8_block_fp8.py
@@ -11,8 +11,8 @@
 from typing import Any
 
 import torch
-import tqdm
 import triton
+from tqdm import tqdm
 
 from vllm.model_executor.layers.quantization.utils.fp8_utils import (
     _w8a8_block_fp8_matmul,
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..d677d69c57a2
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=8,N=2048,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,154 @@
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
+        "num_stages": 4
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
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "32": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "48": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "64": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "96": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "128": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
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
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "8192": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    }
+}
diff --git a/vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..fbca5ce05d01
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/N=2112,K=7168,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -0,0 +1,26 @@
+{
+    "2048": {
+        "BLOCK_SIZE_M": 256,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/quantization/utils/configs/README.md b/vllm/model_executor/layers/quantization/utils/configs/README.md
new file mode 100644
index 000000000000..1110ced4fa06
--- /dev/null
+++ b/vllm/model_executor/layers/quantization/utils/configs/README.md
@@ -0,0 +1,3 @@
+# Quantization Kernel Config
+
+Use scripts under `benchmarks/kernels/` to generate these config files.

```
