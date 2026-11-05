# PR #31407: Add Fused MoE Triton kernels for GLM-4.5-Air, GLM-4.5v, GLM-4.6v on 2x RTX Pro 6000

- field: gpu_kernels  | area: model_executor_layers
- author: mratsim
- state: MERGED  created: 2025-12-27T10:00:14Z  merged: 2025-12-28T16:38:34Z
- url: https://github.com/vllm-project/vllm/pull/31407
- labels: performance, ready

# Issue

## Problem

vLLM's Fused MoE Triton kernel selects tiling parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) from per-(expert-count, intermediate-size, device, dtype) JSON config files. When no config exists for a given shape+GPU combination, vLLM falls back to generic default heuristics that are not tuned for the specific hardware, producing suboptimal GEMM tiling and lower MoE throughput. No config files existed for GLM-4.5-Air / GLM-4.5V / GLM-4.6V expert shapes on the RTX Pro 6000 Blackwell GPU with fp8_w8a8.

## Example

GLM-4.5-Air uses E=129 experts (and E=128 for the no-MTP finetunes) with intermediate size N=704 in fp8_w8a8. On 2x RTX Pro 6000 Blackwell, these shapes had no matching config, so the MoE kernel used untuned fallback block sizes across batch sizes 1-4096.

## Use case

Serving GLM-4.5-Air, GLM-4.5V, and GLM-4.6V (and MTP-free finetunes) on dual RTX Pro 6000 Blackwell Workstation cards.

## Resources

GPU compute (Tensor Core utilization) and HBM bandwidth in the fused MoE GEMM; poor tiling wastes both and raises kernel runtime.

## Type of users

Users running GLM-4.5/4.6 MoE models in fp8 quantization on RTX Pro 6000 Blackwell workstation GPUs, typically 2-GPU setups.

## Kind of inputs

fp8_w8a8 MoE forward passes across the full batch/token range (M = 1 to 4096), with expert counts E=128/129 and intermediate N=704.

## Mechanism with the issue



# Solution

## Fix in words

The diff adds two new tuning config JSON files under fused_moe/configs: E=128,N=704 and E=129,N=704 for device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition, dtype=fp8_w8a8. Each file maps 18 batch-size buckets (1, 2, 4, ... 4096) to tuned Triton GEMM parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages). Before, these shape+device combos had no entry and vLLM used generic fallback heuristics; after, the kernel autotuner loads these device-specific configs. Effect: better tiling per batch size, improving MoE kernel throughput on the target GPU. No code logic changed — only data configs added.

## Author's solution

Add optimized fused MoE configs for 2x RTX Pro 6000 covering GLM-4.5-Air (N=129) plus GLM-4.5V/4.6V and Air finetunes with no MTP layers. Configs produced via the SGLang tuner. Tested locally on vLLM and in SGLang; author reports it works.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`
- `vllm/model_executor/layers/fused_moe/configs/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json b/vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
new file mode 100644
index 000000000000..75dfc52cb46d
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 2
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 2
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
+        "BLOCK_SIZE_M": 256,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 2
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json b/vllm/model_executor/layers/fused_moe/configs/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
new file mode 100644
index 000000000000..75dfc52cb46d
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=129,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Workstation_Edition,dtype=fp8_w8a8.json
@@ -0,0 +1,146 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "2": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 4
+    },
+    "4": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "24": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 16,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 4,
+        "num_stages": 5
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 32,
+        "num_warps": 8,
+        "num_stages": 3
+    },
+    "96": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 2
+    },
+    "128": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 4
+    },
+    "256": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 3
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 4,
+        "num_stages": 2
+    },
+    "1536": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 2
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
+        "BLOCK_SIZE_M": 256,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 8,
+        "num_stages": 2
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_stages": 4
+    }
+}

```
