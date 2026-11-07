# PR #9245: [Kernel] adding fused moe kernel config for L40S TP4

- field: gpu_kernels  | area: model_executor_layers
- author: bringlein
- state: MERGED  created: 2024-10-10T12:03:18Z  merged: 2024-10-11T15:54:23Z
- url: https://github.com/vllm-project/vllm/pull/9245
- labels: 

# Issue

## Problem

The fused MoE Triton kernel in vLLM selects tiling and scheduling parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) from per-device JSON config files keyed by expert count and intermediate dim. When no config exists for a given (num_experts, N, device) combination, the kernel falls back to generic default heuristics that are not tuned for the specific GPU's SM count, register file, and shared memory. On the NVIDIA L40S this leaves fused MoE throughput and latency on the table.

## Example

Mixtral-8x7B with TP4 has 8 experts and intermediate N=3584 per shard, matching config file `E=8,N=3584,device_name=NVIDIA_L40S.json`. Without it, batch sizes 8-64 ran ~5-6% slower (e.g. bs=64: 9.53s avg latency vs 8.98s tuned).

## Use case

Serving Mixtral-8x7B (sparse MoE) across 4 L40S GPUs with tensor parallelism for throughput-oriented inference workloads.

## Resources

GPU compute and on-chip resources (SM occupancy, shared memory for tiling, register pressure via num_warps/num_stages pipelining) on the L40S.

## Type of users

Users serving MoE models (Mixtral) with multi-GPU tensor parallelism on NVIDIA L40S hardware specifically.

## Kind of inputs

Full range of token/batch sizes from 1 to 8192 tokens per forward, bf16/fp16, MoE routing to 8 experts with N=3584 shard intermediate size.

## Mechanism with the issue

`fused_moe` kernel dispatch (`get_moe_configs` / `try_get_optimal_moe_config`) looks up a JSON file named by `E=<experts>,N=<intermediate>,device_name=<gpu>`. Missing file means the code path returns untuned fallback block/warp/stage parameters, producing suboptimal Triton grouped-GEMM schedules for L40S.

# Solution

## Fix in words

Adds a new config file `E=8,N=3584,device_name=NVIDIA_L40S.json` with hand/auto-tuned Triton launch params for 20 batch-size buckets (1 through 8192). Before: L40S fell back to generic defaults; after: the lookup finds this file and loads per-bucket BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_ctas, num_stages tuned for L40S's architecture. No kernel logic changes—purely a data file that the existing config-loading path picks up, yielding up to 10% more throughput and ~6% better latency.

## Author's solution

Author added Triton configurations for `fused_moe` on L40S (Mixtral-8x7B, TP4), tuned via latency benchmarks across batch sizes 1-64 and validated with fmperf on a larger workload. Reports up to 10% higher throughput and ~6% better latency versus untuned upstream.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_L40S.json`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_L40S.json b/vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_L40S.json
new file mode 100644
index 000000000000..d720deb4bdd7
--- /dev/null
+++ b/vllm/model_executor/layers/fused_moe/configs/E=8,N=3584,device_name=NVIDIA_L40S.json
@@ -0,0 +1,173 @@
+{
+    "1": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 4,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "2": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 64,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 2,
+        "num_warps": 4,
+        "num_ctas": 1,
+        "num_stages": 7
+    },
+    "4": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 128,
+        "num_warps": 2,
+        "num_ctas": 1,
+        "num_stages": 4
+    },
+    "8": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 2,
+        "num_ctas": 1,
+        "num_stages": 1
+    },
+    "16": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 256,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 2,
+        "num_ctas": 1,
+        "num_stages": 1
+    },
+    "32": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 2,
+        "num_warps": 4,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "48": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 2,
+        "num_warps": 4,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "64": {
+        "BLOCK_SIZE_M": 16,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 2,
+        "num_warps": 4,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "96": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 8,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "128": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 8,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "192": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 8,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "256": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 16,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "512": {
+        "BLOCK_SIZE_M": 32,
+        "BLOCK_SIZE_N": 32,
+        "BLOCK_SIZE_K": 64,
+        "GROUP_SIZE_M": 128,
+        "num_warps": 2,
+        "num_ctas": 1,
+        "num_stages": 8
+    },
+    "1024": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 128,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 4,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 3
+    },
+    "2048": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 16,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "3072": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 16,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "4096": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "6144": {
+        "BLOCK_SIZE_M": 128,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 32,
+        "GROUP_SIZE_M": 64,
+        "num_warps": 8,
+        "num_ctas": 1,
+        "num_stages": 2
+    },
+    "8192": {
+        "BLOCK_SIZE_M": 64,
+        "BLOCK_SIZE_N": 256,
+        "BLOCK_SIZE_K": 128,
+        "GROUP_SIZE_M": 1,
+        "num_warps": 16,
+        "num_ctas": 1,
+        "num_stages": 2
+    }
+}
\ No newline at end of file

```
