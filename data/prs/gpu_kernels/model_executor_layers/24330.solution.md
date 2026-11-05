# PR #24330: QWEN3 Thinking Fused MoE kernels Optimization configs

- field: gpu_kernels  | area: model_executor_layers
- author: samanamp
- state: MERGED  created: 2025-09-05T16:13:22Z  merged: 2025-09-07T03:18:54Z
- url: https://github.com/vllm-project/vllm/pull/24330
- labels: performance, ready, qwen

# Issue

## Problem

Triton fused MoE kernels in vLLM dispatch tile configs (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) per (num_experts, intermediate_size, device, dtype, block_shape) key, read from JSON files. For Qwen3 Thinking MoE (E=128, N=384 and N=768 shards, fp8_w8a8, block_shape [128,128]) on NVIDIA B200/GB200, no tuned configs existed or existing ones were suboptimal. Kernel fell back to default/generic tile params, producing poor occupancy and pipelining for these expert/shape sizes.

## Example

Qwen3-235B-A22B-Thinking-2507-FP8 at TP=4 EP on B200: N=768 shard, batch M=2 previously used BLOCK_SIZE_N=256, num_stages=5; M=512 used BLOCK_SIZE_K=256, GROUP_SIZE_M=32. Suboptimal tiles cost throughput and tail latency.

## Use case

Serving Qwen3 Thinking MoE in FP8 with expert parallelism across multiple Blackwell GPUs under concurrent long-context decode.

## Resources

GPU compute (Tensor Core utilization) and HBM bandwidth via SM occupancy and software-pipeline depth in the grouped GEMM kernel.

## Type of users

Users serving large FP8-quantized MoE models (Qwen3-235B class) with tensor/expert parallelism on NVIDIA B200 and GB200 hardware.

## Kind of inputs

fp8_w8a8 MoE GEMMs over 128 experts, intermediate shards N=384 and N=768, block_shape [128,128], across batch/token counts M from 1 to 4096 — high-concurrency decode (64 concurrent, 7500 in/out tokens).

## Mechanism with the issue



# Solution

## Fix in words

Diff adds new per-shape tuning JSON files for E=128,N=384 and E=128,N=768 on both NVIDIA_B200 and NVIDIA_GB200 (fp8_w8a8, block_shape [128,128]), and rewrites existing E=128,N=768 B200 entries. Per batch-size bucket it sets better tile params: e.g. N=768 B200 M=2 drops BLOCK_SIZE_N 256→128 and num_stages 5→4; M=512 drops BLOCK_SIZE_K 256→128 and GROUP_SIZE_M 32→1; large M buckets retune GROUP_SIZE_M 32→16. New N=384 configs use BLOCK_SIZE_N=128 with K=128/256 and num_stages 3-5 scaling by M. No kernel logic changes — only lookup-table tile selections. Effect: autotuner picks higher-occupancy, better-pipelined tiles for these Qwen3 MoE shapes.

## Author's solution

Author added optimized Fused MoE kernel tuning configs for Qwen3 Thinking, validated via benchmark_moe.py single-kernel bench and E2E serving of Qwen3-235B-A22B-Thinking-2507-FP8 at TP=4 with expert parallelism. Reports 13.7% higher output token throughput, 12% lower median TPOT, and 17% lower P99 TPOT.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json`
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..e5059358c91e
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
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
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
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
+        "num_stages": 5
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
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
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 4
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
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..db1b6e98df46
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=384,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
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
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
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
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
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
+        "num_stages": 5
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
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
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
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
+        "BLOCK_SIZE_K": 128,
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
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
index b9dc2d71f6dc..1bbb8aa61399 100644
--- a/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_B200,dtype=fp8_w8a8,block_shape=[128,128].json
@@ -9,16 +9,16 @@
     },
     "2": {
         "BLOCK_SIZE_M": 16,
-        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
-        "num_stages": 5
+        "num_stages": 4
     },
     "4": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 256,
+        "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
@@ -26,15 +26,15 @@
     "8": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 128,
+        "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
-        "num_stages": 4
+        "num_stages": 3
     },
     "16": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 256,
+        "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
@@ -42,7 +42,7 @@
     "24": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 256,
+        "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
@@ -53,12 +53,12 @@
         "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
-        "num_stages": 4
+        "num_stages": 5
     },
     "48": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 128,
+        "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
@@ -82,10 +82,10 @@
     "128": {
         "BLOCK_SIZE_M": 16,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 256,
+        "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1,
         "num_warps": 4,
-        "num_stages": 4
+        "num_stages": 3
     },
     "256": {
         "BLOCK_SIZE_M": 16,
@@ -98,8 +98,8 @@
     "512": {
         "BLOCK_SIZE_M": 64,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 256,
-        "GROUP_SIZE_M": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
     },
@@ -107,7 +107,7 @@
         "BLOCK_SIZE_M": 64,
         "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 128,
-        "GROUP_SIZE_M": 32,
+        "GROUP_SIZE_M": 16,
         "num_warps": 4,
         "num_stages": 4
     },
@@ -115,7 +115,7 @@
         "BLOCK_SIZE_M": 64,
         "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 128,
-        "GROUP_SIZE_M": 1,
+        "GROUP_SIZE_M": 16,
         "num_warps": 4,
         "num_stages": 4
     },
@@ -123,15 +123,15 @@
         "BLOCK_SIZE_M": 64,
         "BLOCK_SIZE_N": 128,
         "BLOCK_SIZE_K": 256,
-        "GROUP_SIZE_M": 32,
+        "GROUP_SIZE_M": 16,
         "num_warps": 4,
         "num_stages": 4
     },
     "3072": {
         "BLOCK_SIZE_M": 64,
         "BLOCK_SIZE_N": 128,
-        "BLOCK_SIZE_K": 128,
-        "GROUP_SIZE_M": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
         "num_warps": 4,
         "num_stages": 4
     },
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
new file mode 100644
index 000000000000..8fb4947d62ab
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_GB200,dtype=fp8_w8a8,block_shape=[128,128].json
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
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
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
+        "num_stages": 4
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
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
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
+        "GROUP_SIZE_M": 64,
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
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
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

```
