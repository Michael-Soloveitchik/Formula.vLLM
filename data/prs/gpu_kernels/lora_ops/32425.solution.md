# PR #32425: [LoRA] Update LoRA expand kernel heuristic

- field: gpu_kernels  | area: lora_ops
- author: xyang16
- state: MERGED  created: 2026-01-15T18:53:38Z  merged: 2026-01-16T18:38:07Z
- url: https://github.com/vllm-project/vllm/pull/32425
- labels: ready

# Issue

## Problem

The LoRA expand Triton kernel `_lora_expand_kernel` used a fixed `block_n=128` tiling heuristic regardless of the layer structure. For fused/sliced layers like `QKVParallelLinearWithLoRA`, the output is split into multiple slices (num_slices > 1), so each slice's output dimension N is small. A large block_n of 128 produces too few blocks per slice, leaving streaming multiprocessors (SMs) underutilized and serializing work the GPU could run in parallel.

## Example

For qkv_proj with 3 fused slices (Q, K, V), N per slice is ~1/3 the total output size, so block_n=128 launches far fewer blocks than SMs available. Profiling showed `_lora_expand_kernel` at ~14µs on main vs ~10µs after the fix (~30% faster; up to 2x with a tuned config).

## Use case

Online serving of LoRA-adapted models where every decode step runs expand kernels across fused QKV/gate-up projection layers.

## Resources

GPU compute — specifically SM occupancy/concurrency. Too-coarse tiling leaves SMs idle.

## Type of users

Users serving LoRA adapters (--enable-lora) on models with fused multi-slice linear layers (QKV, gate-up), e.g. gpt-oss-20b with LoRA rank 16.

## Kind of inputs

Decode-phase workloads with small output tiles per slice; many concurrent requests (max-concurrency 16) hitting fused QKVParallelLinearWithLoRA / MergedColumnParallel layers each step.

## Mechanism with the issue

`get_lora_op_configs` in vllm/lora/ops/triton_ops/utils.py returns the default tiling config. The hardcoded `block_n=128` ignored `num_slices`, so expand kernels over sliced layers (N already divided by num_slices) tiled too coarsely, yielding few CTAs and low SM concurrency — the expand kernel ran slower than the shrink kernel.

# Solution

## Fix in words

The diff changes the default `block_n` in `get_lora_op_configs` from a constant `128` to `max(64, next_power_of_2(128 // num_slices))`. Before: every layer got block_n=128 regardless of slice count. After: block_n scales down with the number of slices (e.g. 128→64 for a 2-way split, clamped to a floor of 64 and rounded to a power of 2). Smaller block_n over sliced layers launches more blocks, raising SM concurrency and cutting `_lora_expand_kernel` time ~30% (~14µs→~10µs).

## Author's solution

Author found via profiling that qkv_proj expand kernel was slower than shrink because multi-slice layers (num_slices > 1) have smaller per-slice output size N. Setting a smaller block_n launches more blocks and increases SM concurrency, improving `_lora_expand_kernel` ~30% (up to 2x with tuned config) and e2e output token throughput ~1%.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/ops/triton_ops/utils.py`

### Diff
```diff
diff --git a/vllm/lora/ops/triton_ops/utils.py b/vllm/lora/ops/triton_ops/utils.py
index 51535f32c136..f0cb9a5c0499 100644
--- a/vllm/lora/ops/triton_ops/utils.py
+++ b/vllm/lora/ops/triton_ops/utils.py
@@ -251,7 +251,7 @@ def get_lora_op_configs(
     else:
         default = {
             "block_m": 64,
-            "block_n": 128,
+            "block_n": max(64, next_power_of_2(128 // num_slices)),
             "block_k": 16,
             "num_warps": 4,
             "num_ctas": 1,

```
