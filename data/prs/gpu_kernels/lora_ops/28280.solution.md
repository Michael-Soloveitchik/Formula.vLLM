# PR #28280: [Kernel] Optimization of the mm_k operator.

- field: gpu_kernels  | area: lora_ops
- author: caozuoba
- state: MERGED  created: 2025-11-07T08:43:39Z  merged: 2025-11-10T16:03:47Z
- url: https://github.com/vllm-project/vllm/pull/28280
- labels: ready

# Issue

## Problem

The `mm_k` Triton kernel helper (used by LoRA shrink/expand ops) handles the non-`EVEN_K` case inefficiently. On every K-loop iteration it applies element-wise masking on both `tl.load` calls and always issues a `tl.dot()`, even when the K-block is entirely in bounds or entirely out of bounds. Masking is only truly needed for the single partial block at the K boundary, so the per-iteration mask computation and the wasted `tl.dot()` on fully-out-of-range blocks are redundant overhead.

## Example

A LoRA matmul where K is not a multiple of `BLOCK_K` (or `BLOCK_K * SPLIT_K`): all but the last K iteration are fully in-bounds yet still pay mask overhead, and under SPLIT_K some groups iterate past K and still execute masked loads plus a `tl.dot()` that contributes nothing.

## Use case

Serving LoRA-adapted models, where the shrink and expand GEMMs run per decode/prefill step on every request using an adapter.

## Resources

GPU compute — redundant mask ops and no-op `tl.dot()` matrix-multiply instructions burn SM cycles, hurting TTFT/TPOT throughput.

## Type of users

Users serving models with LoRA adapters, especially on less advanced / non-Hopper GPU architectures (validated on Metax C500 and NVIDIA H800) where wasted tensor-core work is more costly.

## Kind of inputs

Matmuls whose K dimension is not evenly divisible by the K tile (`EVEN_K` false), and the shrink kernel's SPLIT_K decomposition where K groups may exceed K.

## Mechanism with the issue

In `vllm/lora/ops/triton_ops/kernel_utils.py`, the `else` (not `EVEN_K`) branch of the `for k in range(tl.cdiv(K, BLOCK_K*SPLIT_K))` loop computes a mask `offset_k < K - k*(BLOCK_K*SPLIT_K)` on each `tl.load` and unconditionally accumulates `tl.dot(tiled_a, tiled_b)`. This treats every iteration as if it could be partial, so fully-valid blocks pay mask cost and fully-invalid blocks still run a zeroed `tl.dot()`.

# Solution

## Fix in words

The diff adds a `base_k` parameter and precomputes each iteration's global K offset `iter_k = k*STEP_K + base_k` (with `STEP_K = BLOCK_K*SPLIT_K`) and `block_end = iter_k + BLOCK_K`. For the non-`EVEN_K` case it now branches three ways: if `iter_k >= K` the block is fully out of range and is skipped entirely (no load, no `tl.dot()`); if `block_end <= K` it takes a fast path with unmasked `tl.load`/`tl.dot`; only the remaining partial boundary block builds a `mask = iter_k + arange(0,BLOCK_K) < K` and loads with masking. Callers pass `base_k=0` in `do_expand_kernel` and `base_k=pid_sk*BLOCK_K` in `do_shrink_kernel` to account for SPLIT_K grouping. This removes per-iteration masking from in-bounds blocks and eliminates wasted matmuls on out-of-bounds blocks, speeding up the kernel.

## Author's solution

The author observed that in the non-`EVEN_K` case the main branch adds masking and a `tl.dot()` on every loop iteration, but masking is only needed for partially out-of-bound accesses. They split each iteration into three cases: completely in bounds (skip masking, just `tl.dot()`), completely out of bounds (skip entirely — no masking or `tl.dot()`), and partial (mask as before). Reducing redundant masking and `tl.dot()` calls improves speed, particularly on less advanced GPU architectures, with gains shown on Metax C500 and H800.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/ops/triton_ops/kernel_utils.py`

### Diff
```diff
diff --git a/vllm/lora/ops/triton_ops/kernel_utils.py b/vllm/lora/ops/triton_ops/kernel_utils.py
index ebfffc17ae87..c6c2a02fdeb5 100644
--- a/vllm/lora/ops/triton_ops/kernel_utils.py
+++ b/vllm/lora/ops/triton_ops/kernel_utils.py
@@ -23,6 +23,7 @@ def mm_k(
     CAST_TYPE: tl.constexpr,
     b_dtype: tl.constexpr,
     USE_GDC: tl.constexpr,
+    base_k,
 ):
     """
     Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
@@ -47,32 +48,62 @@ def mm_k(
           matrix dtype.
         b_dtype: datatype of the B matrix
         USE_GDC: Whether to use PDL. True indicates use.
+        base_k: Base offset along K dimension for current SPLIT_K group
     """
     accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-    for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
+
+    # Step size along K for each iteration
+    STEP_K = BLOCK_K * SPLIT_K
+
+    # Total number of iterations (compile-time constant)
+    num_iters = tl.cdiv(K, STEP_K)
+
+    for k in range(num_iters):
+        # Current iteration's global K offset
+        iter_k = k * STEP_K + base_k
+
+        # Check if this iteration is completely valid (no masking needed)
+        block_end = iter_k + BLOCK_K
+
         if EVEN_K:
-            # pre-fetech lora weight
+            # K is divisible by BLOCK_K, no masking ever needed
+            # pre-fetch lora weight
             tiled_b = tl.load(b_ptr)
             if USE_GDC:
                 tl.extra.cuda.gdc_wait()
             tiled_a = tl.load(a_ptr)
+            if CAST_TYPE:
+                tiled_a = tiled_a.to(b_dtype)
+            accumulator += tl.dot(tiled_a, tiled_b)
         else:
-            tiled_b = tl.load(
-                b_ptr, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K), other=0
-            )
-            if USE_GDC:
-                tl.extra.cuda.gdc_wait()
-            tiled_a = tl.load(
-                a_ptr, mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K), other=0
-            )
-        if CAST_TYPE:
-            tiled_a = tiled_a.to(b_dtype)
-        accumulator += tl.dot(
-            tiled_a,
-            tiled_b,
-        )
-        a_ptr += BLOCK_K * SPLIT_K * ak_stride
-        b_ptr += BLOCK_K * SPLIT_K * bk_stride
+            # Check if we need element-wise masking
+            if iter_k >= K:
+                # Entire block out of range, skip
+                pass
+            elif block_end <= K:
+                # Entire block in range, no masking needed (fast path)
+                tiled_b = tl.load(b_ptr)
+                if USE_GDC:
+                    tl.extra.cuda.gdc_wait()
+                tiled_a = tl.load(a_ptr)
+                if CAST_TYPE:
+                    tiled_a = tiled_a.to(b_dtype)
+                accumulator += tl.dot(tiled_a, tiled_b)
+            else:
+                # Partial block, need masking (only last iteration)
+                k_offsets = tl.arange(0, BLOCK_K)
+                mask = iter_k + k_offsets < K
+                tiled_b = tl.load(b_ptr, mask=mask[:, None], other=0.0)
+                if USE_GDC:
+                    tl.extra.cuda.gdc_wait()
+                tiled_a = tl.load(a_ptr, mask=mask[None, :], other=0.0)
+                if CAST_TYPE:
+                    tiled_a = tiled_a.to(b_dtype)
+                accumulator += tl.dot(tiled_a, tiled_b)
+
+        a_ptr += STEP_K * ak_stride
+        b_ptr += STEP_K * bk_stride
+
     return accumulator
 
 
@@ -178,6 +209,7 @@ def do_expand_kernel(
         CAST_TYPE,
         cur_lora_ptr.dtype.element_ty,
         USE_GDC,
+        base_k=0,
     )
 
     tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
@@ -284,6 +316,7 @@ def do_shrink_kernel(
         False,
         cur_lora_ptr.dtype.element_ty,
         False,  # USE_GDC is always False in shrink kernel
+        base_k=pid_sk * BLOCK_K,
     )
     # GDC launch dependents hints the runtime system to launch dependent kernels.
     if USE_GDC:

```
