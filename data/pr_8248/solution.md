# PR #8248: [Kernel] [Triton] Memory optimization for awq_gemm and awq_dequantize, 2x throughput

- field: gpu_kernels  | area: model_executor_layers
- author: rasmith
- state: MERGED  created: 2024-09-06T18:19:01Z  merged: 2024-09-06T22:26:09Z
- url: https://github.com/vllm-project/vllm/pull/8248
- labels: rocm, ready

# Issue

## Problem

The Triton awq_dequantize and awq_gemm kernels loaded the packed int4 weights (and the zeros) redundantly from global memory. AWQ packs 8 four-bit values into each int32 element, but the kernels computed column offsets as `tl.arange(0, BLOCK_SIZE_X * 8) // 8`, which maps 8 consecutive output lanes to the SAME packed int32 address. Each packed word was therefore fetched from HBM 8 times instead of once. Zeros and scales were loaded the same wasteful way via `// group_size` index expansion.

## Example

Running `benchmark_throughput.py --input-len 2048 --output-len 128 --quantization awq --model Qwen2-7B-Instruct-AWQ` only reached 6.14 req/s (13351 tok/s) with the unoptimized Triton kernel — slower than the AWQ CUDA kernel at 11.51 req/s.

## Use case

Serving AWQ 4-bit quantized LLMs with the Triton backend, where every decode/prefill step dequantizes weight tiles inside the GEMM inner loop.

## Resources

HBM / memory bandwidth — the kernel is memory-bound on redundant weight loads, not compute-bound.

## Type of users

Users serving INT4 AWQ-quantized models on the Triton path (e.g. non-CUDA or portability-focused deployments) who want throughput matching or beating the handwritten CUDA kernel.

## Kind of inputs

Any AWQ-quantized weight tensors (packed 8x int4 per int32) under normal prefill/decode batch sizes; larger batch/throughput workloads make the bandwidth waste dominate.

## Mechanism with the issue



# Solution

## Fix in words

Before: offsets used `tl.arange(0, BLOCK_SIZE_X * 8) // 8`, forcing 8 duplicate HBM loads per packed int32; zeros/scales similarly reloaded per-group-row. After: offsets use `tl.arange(0, BLOCK_SIZE_X)` (and `BLOCK_SIZE_N // 8`), loading each packed word exactly once, then replicating it in-register with three `tl.interleave(x, x)` calls (2^3 = 8 copies) to recover the 8 lanes. Zeros get the same interleave plus `tl.broadcast_to`, and scales/zeros collapse their y/k index to `tl.arange(0, 1)` + `tl.broadcast_to` instead of redundant per-row loads. Net effect: memory loads cut ~8x, dequant math unchanged, yielding 2x throughput (12.77 req/s vs 6.14) and beating the CUDA kernel.

## Author's solution

The author reduced the number of memory loads performed by awq_gemm and awq_dequantize, stating this yields 2x throughput over the previous Triton implementation. Benchmarked on Qwen2-7B-Instruct-AWQ: optimized Triton hit 27783 tok/s vs 13351 tok/s unoptimized, surpassing AWQ CUDA at 25049 tok/s.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/quantization/awq_triton.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/quantization/awq_triton.py b/vllm/model_executor/layers/quantization/awq_triton.py
index ad706f28a742..d0b210c3a274 100644
--- a/vllm/model_executor/layers/quantization/awq_triton.py
+++ b/vllm/model_executor/layers/quantization/awq_triton.py
@@ -22,7 +22,7 @@ def awq_dequantize_kernel(
 
     # Compute offsets and masks for qweight_ptr.
     offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
-    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
+    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
     offsets = num_cols * offsets_y[:, None] + offsets_x[None, :]
 
     masks_y = offsets_y < num_rows
@@ -43,6 +43,9 @@ def awq_dequantize_kernel(
 
     # Load the weights.
     iweights = tl.load(qweight_ptr + offsets, masks)
+    iweights = tl.interleave(iweights, iweights)
+    iweights = tl.interleave(iweights, iweights)
+    iweights = tl.interleave(iweights, iweights)
 
     # Create reverse AWQ order as tensor: [0, 4, 1, 5, 2, 6, 3, 7]
     # that will map given indices to the correct order.
@@ -59,9 +62,8 @@ def awq_dequantize_kernel(
     iweights = (iweights >> shifts) & 0xF
 
     # Compute zero offsets and masks.
-    zero_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size +
-                      tl.arange(0, BLOCK_SIZE_Y) // group_size)
-    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X * 8) // 8
+    zero_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
+    zero_offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
     zero_offsets = num_cols * zero_offsets_y[:, None] + zero_offsets_x[None, :]
 
     zero_masks_y = zero_offsets_y < num_rows // group_size
@@ -70,13 +72,16 @@ def awq_dequantize_kernel(
 
     # Load the zeros.
     zeros = tl.load(zeros_ptr + zero_offsets, zero_masks)
+    zeros = tl.interleave(zeros, zeros)
+    zeros = tl.interleave(zeros, zeros)
+    zeros = tl.interleave(zeros, zeros)
+    zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
 
     # Unpack and reorder: shift out the correct 4-bit value and mask.
     zeros = (zeros >> shifts) & 0xF
 
     # Compute scale offsets and masks.
-    scale_offsets_y = (pid_y * BLOCK_SIZE_Y // group_size +
-                       tl.arange(0, BLOCK_SIZE_Y) // group_size)
+    scale_offsets_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
     scale_offsets_x = (pid_x * BLOCK_SIZE_X * 8 +
                        tl.arange(0, BLOCK_SIZE_X * 8))
     scale_offsets = (num_cols * 8 * scale_offsets_y[:, None] +
@@ -87,6 +92,7 @@ def awq_dequantize_kernel(
 
     # Load the scales.
     scales = tl.load(scales_ptr + scale_offsets, scale_masks)
+    scales = tl.broadcast_to(scales, (BLOCK_SIZE_Y, BLOCK_SIZE_X * 8))
 
     # Dequantize.
     iweights = (iweights - zeros) * scales
@@ -137,12 +143,10 @@ def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
     offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
     masks_am = offsets_am < M
 
-    offsets_bn = (pid_n * (BLOCK_SIZE_N // 8) +
-                  tl.arange(0, BLOCK_SIZE_N) // 8)
+    offsets_bn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
     masks_bn = offsets_bn < N // 8
 
-    offsets_zn = (pid_n * (BLOCK_SIZE_N // 8) +
-                  tl.arange(0, BLOCK_SIZE_N) // 8)
+    offsets_zn = pid_n * (BLOCK_SIZE_N // 8) + tl.arange(0, BLOCK_SIZE_N // 8)
     masks_zn = offsets_zn < N // 8
 
     offsets_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
@@ -165,22 +169,30 @@ def awq_gemm_kernel(a_ptr, b_ptr, c_ptr, zeros_ptr, scales_ptr, M, N, K,
 
         masks_b = masks_k[:, None] & masks_bn[None, :]
         b = tl.load(b_ptrs, mask=masks_b)
+        b = tl.interleave(b, b)
+        b = tl.interleave(b, b)
+        b = tl.interleave(b, b)
 
         # Dequantize b.
         offsets_szk = (
             (BLOCK_SIZE_K * SPLIT_K * k + pid_z * BLOCK_SIZE_K) // group_size +
-            tl.arange(0, BLOCK_SIZE_K) // group_size)
+            tl.arange(0, 1))
         offsets_z = (N // 8) * offsets_szk[:, None] + offsets_zn[None, :]
         masks_zk = offsets_szk < K // group_size
         masks_z = masks_zk[:, None] & masks_zn[None, :]
         zeros_ptrs = zeros_ptr + offsets_z
         zeros = tl.load(zeros_ptrs, mask=masks_z)
+        zeros = tl.interleave(zeros, zeros)
+        zeros = tl.interleave(zeros, zeros)
+        zeros = tl.interleave(zeros, zeros)
+        zeros = tl.broadcast_to(zeros, (BLOCK_SIZE_K, BLOCK_SIZE_N))
 
         offsets_s = N * offsets_szk[:, None] + offsets_sn[None, :]
         masks_sk = offsets_szk < K // group_size
         masks_s = masks_sk[:, None] & masks_sn[None, :]
         scales_ptrs = scales_ptr + offsets_s
         scales = tl.load(scales_ptrs, mask=masks_s)
+        scales = tl.broadcast_to(scales, (BLOCK_SIZE_K, BLOCK_SIZE_N))
 
         b = (b >> shifts) & 0xF
         zeros = (zeros >> shifts) & 0xF

```
