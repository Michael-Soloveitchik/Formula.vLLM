# PR #35457: [Model Performance] Add Qwen3MoE tuned MoE configs for H200

- field: parallelism  | area: model_executor_layers
- author: chengyinie
- state: MERGED  created: 2026-02-27T00:41:17Z  merged: 2026-02-27T05:51:15Z
- url: https://github.com/vllm-project/vllm/pull/35457
- labels: ready, qwen

# Issue

## Problem

Qwen3MoE (E=128 experts, intermediate size N=96 per shard) running fused Triton MoE kernels on H200 had no device-specific tuned configuration. Without a matching config file, vLLM falls back to generic default Triton kernel parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages), which are suboptimal for this shape on H200, leaving GPU throughput on the table.

## Example

At batch sizes 16-64 in BF16, the default kernel params were 20-24% slower than tuned; FP8 at batch 48-128 was 8-11% slower. Shape is E=128, N=96, TP=8 on NVIDIA H200.

## Use case

Serving Qwen3MoE on Hopper H200 nodes with tensor parallelism TP=8, both BF16 and FP8 (w8a8) quantized inference.

## Resources

GPU compute (Tensor Core / SM occupancy) and HBM bandwidth in the fused MoE grouped-GEMM; wrong tile sizes waste compute and memory throughput.

## Type of users

Users serving MoE models (Qwen3MoE) on H200 GPUs with multi-GPU tensor parallelism, in both BF16 and FP8 w8a8 quantized configs.

## Kind of inputs

Wide range of batch / token counts (M = 1 to 4096), BF16 and fp8_w8a8 dtypes, E=128 experts, N=96 shard intermediate dim.

## Mechanism with the issue

Fused MoE layer (vllm/model_executor/layers/fused_moe) selects Triton kernel launch params by looking up a JSON config keyed on E, N, device_name, and dtype. No file matched E=128,N=96,NVIDIA_H200, so the lookup missed and fell back to heuristic defaults that don't match H200's SM/warp characteristics for this problem size.

# Solution

## Fix in words

Diff adds two new config files under fused_moe/configs: E=128,N=96,device_name=NVIDIA_H200.json (BF16) and the fp8_w8a8 variant. Each maps batch-size buckets (1..4096) to tuned BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages. Before: lookup missed and used generic defaults; after: the MoE dispatcher finds exact tuned params per batch size and dtype. Effect: small-M buckets use narrow BLOCK_SIZE_N (32/64) with low GROUP_SIZE_M, large-M buckets scale up to 128x128 tiles and num_warps=8, yielding the measured 2-24% speedups. No kernel logic changes — pure autotuned parameter tables.

## Author's solution

Author ran Triton MoE autotuning for Qwen3MoE (E=128, N=96) on H200 at TP=8 and committed the resulting best configs as two JSON files, one BF16 and one FP8 w8a8. Benchmarks confirm 20-24% BF16 gains at batch 16-64 and 8-11% FP8 gains at batch 48-128, addressing issue #34256 on improving the MoE Oracle for Qwen3MoE on Hopper.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json`
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200.json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
new file mode 100644
index 000000000000..620fe9365aa7
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200,dtype=fp8_w8a8.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.6.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
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
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
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
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
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
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 2
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200.json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200.json
new file mode 100644
index 000000000000..fc7dda8a7844
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=96,device_name=NVIDIA_H200.json
@@ -0,0 +1,147 @@
+{
+    "triton_version": "3.6.0",
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "256": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    }
+}

```
