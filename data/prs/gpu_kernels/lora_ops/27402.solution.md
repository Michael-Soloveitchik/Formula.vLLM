# PR #27402: [Kernel] LoRA triton kernels support PDL

- field: gpu_kernels  | area: lora_ops
- author: jeejeelee
- state: MERGED  created: 2025-10-23T08:07:34Z  merged: 2025-11-07T08:05:48Z
- url: https://github.com/vllm-project/vllm/pull/27402
- labels: ready

# Issue

## Problem

The LoRA triton kernels run as serialized, back-to-back GPU launches (shrink followed by expand, and the fused-MoE shrink/expand pair) with no overlap between a kernel finishing and the next one starting. Each dependent kernel can only begin after the prior kernel's grid fully completes and the runtime schedules the next launch, leaving the GPU idle during that launch/drain gap. This gap is pure overhead on top of the actual matmul work and is proportionally largest when the matmuls themselves are small.

## Example

On Qwen3-32B shapes at TP=4, the small-M LoRA matmuls spend a notable fraction of time in inter-kernel launch gaps; eliminating the gap yields ~5-20% kernel runtime speedup for smaller Ms. End-to-end H20 serving of Qwen3-30B-A3B (max-lora-rank=64, 100 prompts, 2048/1024 in/out) drops benchmark duration from 74.05s to 69.74s.

## Use case

Online LoRA serving where every decode/prefill step fires the shrink and expand LoRA kernels in sequence, so the per-launch overhead recurs on every forward pass.

## Resources

GPU compute utilization and kernel-launch/scheduling latency — the GPU stalls in the idle window between dependent kernel launches rather than doing useful matmul work.

## Type of users

Users serving LoRA (and fused-MoE LoRA) adapters on Hopper-class hardware (SM90+, e.g. H20/H100) with Triton 3.5.0; the gain is largest for small batch / small-M decode traffic.

## Kind of inputs

Small-M workloads — low effective token counts per step (decode-heavy or small batch), where matmul time is small relative to fixed launch overhead; moderate LoRA ranks.

## Mechanism with the issue

The shrink output feeds the expand input, forcing strict ordering between the two triton launches (`_lora_shrink`/`_lora_expand` and the `_fused_moe_lora_shrink`/`_fused_moe_lora_expand` pair). The dependent kernel cannot start until the producer kernel's entire grid drains, so the scheduler inserts an idle gap on every launch; `mm_k` and the fused-MoE main loop have no mechanism to begin prefetching the next stage's inputs during that gap.

# Solution

## Fix in words

The diff adds CUDA Programmatic Dependent Launch (PDL) via Triton GDC primitives, gated by a new cached `supports_pdl()` helper that returns True only on CUDA SM90+. A `USE_GDC` constexpr is threaded through the shrink/expand kernels and `mm_k`/`do_shrink_kernel`/`do_expand_kernel`, and each launch passes `launch_pdl=use_gdc` as kernel metadata. In the K-loop the LoRA weight `b`/`tiled_b` load is hoisted above a `tl.extra.cuda.gdc_wait()` so the weight prefetch overlaps the wait on the prior kernel; the producer kernel (shrink, `IS_PRIMARY=True`) calls `tl.extra.cuda.gdc_launch_dependents()` to let the runtime launch the dependent expand kernel early. The fused-MoE expand runs as the waiter (`IS_PRIMARY=False`), and the split-K write-back is relaxed to `tl.atomic_add(..., sem="relaxed")`, together overlapping the two launches and removing the inter-kernel idle gap.

## Author's solution

The author adds PDL support to the LoRA triton shrink/expand kernels so dependent kernels overlap, reporting a slight end-to-end serving gain on H20 (74.05s -> 69.74s, TPOT 63.67ms -> 59.51ms) and 5-20% kernel-runtime speedup for smaller Ms on Qwen3-32B TP=4 shapes. This is part 2 of issue #27371 and requires Triton 3.5.0 (Triton 3.4.0 errored). Adding the same logic to fused_moe_lora shrink/expand was suggested as follow-up.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/ops/triton_ops/fused_moe_lora_op.py`
- `vllm/lora/ops/triton_ops/kernel_utils.py`
- `vllm/lora/ops/triton_ops/lora_expand_op.py`
- `vllm/lora/ops/triton_ops/lora_shrink_op.py`
- `vllm/lora/ops/triton_ops/utils.py`

### Diff
```diff
diff --git a/vllm/lora/ops/triton_ops/fused_moe_lora_op.py b/vllm/lora/ops/triton_ops/fused_moe_lora_op.py
index 8f85f926aa4f..6d6de2529de3 100644
--- a/vllm/lora/ops/triton_ops/fused_moe_lora_op.py
+++ b/vllm/lora/ops/triton_ops/fused_moe_lora_op.py
@@ -6,6 +6,8 @@
 from vllm.triton_utils import tl, triton
 from vllm.utils.torch_utils import direct_register_custom_op
 
+from .utils import supports_pdl
+
 _LORA_PTR_DICT: dict[tuple[int, ...], torch.tensor] = {}
 
 
@@ -82,6 +84,8 @@ def _fused_moe_lora_kernel(
     BLOCK_SIZE_K: tl.constexpr,
     GROUP_SIZE_M: tl.constexpr,
     SPLIT_K: tl.constexpr,
+    USE_GDC: tl.constexpr,
+    IS_PRIMARY: tl.constexpr,
 ):
     pid = tl.program_id(axis=0)
     slice_id = tl.program_id(axis=1)
@@ -110,13 +114,11 @@ def _fused_moe_lora_kernel(
     num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr + lora_id)
     if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
         return
-
     # get the expert_id to process curr shard
     ind = lora_id * stride_el + pid_m
     expert_id = tl.load(expert_ids_ptr + ind, ind < max_loras * stride_el, -1)
     if expert_id == -1:
         return
-
     # get a_ptr,b_ptr,c_ptr
     cur_a_ptr = a_ptr + (slice_id % num_slice_a) * slice_a_size
     cur_b_ptr = tl.load(b_ptr + slice_id).to(tl.pointer_type(c_ptr.dtype.element_ty))
@@ -149,12 +151,17 @@ def _fused_moe_lora_kernel(
     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
     for k in range(0, grid_k):
         k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
+        # pre-fetch lora weight
+        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
+        # GDC wait waits for ALL programs in the the prior kernel to complete
+        # before continuing.
+        if USE_GDC and not IS_PRIMARY:
+            tl.extra.cuda.gdc_wait()
         a = tl.load(
             a_ptrs,
             mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
             other=0.0,
         )
-        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
         accumulator += tl.dot(a, b)
         # Advance the ptrs to the next K block.
         a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
@@ -163,12 +170,15 @@ def _fused_moe_lora_kernel(
     if MUL_ROUTED_WEIGHT:
         moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
         accumulator = accumulator * moe_weight[:, None]
-
+    if USE_GDC and IS_PRIMARY:
+        # GDC launch dependents hints the runtime system to launch dependent kernels.
+        tl.extra.cuda.gdc_launch_dependents()
     accumulator = accumulator.to(c_ptr.dtype.element_ty)
     # Write back the block of the output
     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
     c_ptrs = cur_c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
     c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
+
     if SPLIT_K == 1:
         tl.store(c_ptrs, accumulator, mask=c_mask)
     else:
@@ -209,7 +219,7 @@ def _fused_moe_lora_shrink(
     mul_routed_weight: bool = False,
 ) -> None:
     w1_lora_a_stacked = lora_a_stacked[0]
-
+    use_gdc = supports_pdl(qcurr_hidden_states.device)
     shrink_config = {
         "BLOCK_SIZE_M": block_size_m,
         "BLOCK_SIZE_N": block_size_n,
@@ -218,6 +228,8 @@ def _fused_moe_lora_shrink(
         "num_warps": num_warps,
         "num_stages": num_stages,
         "SPLIT_K": split_k,
+        "USE_GDC": use_gdc,
+        "launch_pdl": use_gdc,  # triton kernel metadata
     }
 
     b_ptr = _get_ptr(lora_a_stacked, device)
@@ -229,7 +241,6 @@ def _fused_moe_lora_shrink(
         len(lora_a_stacked),
         lora_a_stacked[0].shape[0],
     )
-
     _fused_moe_lora_kernel[grid](
         qcurr_hidden_states,
         b_ptr,
@@ -261,6 +272,7 @@ def _fused_moe_lora_shrink(
         num_slice_c=num_slices,
         top_k=1 if mul_routed_weight else top_k_num,
         MUL_ROUTED_WEIGHT=False,
+        IS_PRIMARY=True,
         **shrink_config,
     )
 
@@ -314,7 +326,7 @@ def _fused_moe_lora_expand(
         dtype=output.dtype,
         device=device,
     )
-
+    use_gdc = supports_pdl(a_intermediate_cache1.device)
     expand_config = {
         "BLOCK_SIZE_M": block_size_m,
         "BLOCK_SIZE_N": block_size_n,
@@ -323,6 +335,8 @@ def _fused_moe_lora_expand(
         "num_warps": num_warps,
         "num_stages": num_stages,
         "SPLIT_K": split_k,  # Set split_k = 1 for expand calls
+        "USE_GDC": use_gdc,
+        "launch_pdl": use_gdc,  # triton kernel metadata
     }
 
     grid = lambda META: (
@@ -361,6 +375,7 @@ def _fused_moe_lora_expand(
         num_slice_c=num_slices,
         top_k=1,
         MUL_ROUTED_WEIGHT=mul_routed_weight,
+        IS_PRIMARY=False,
         **expand_config,
     )
     for i in range(num_slices):
diff --git a/vllm/lora/ops/triton_ops/kernel_utils.py b/vllm/lora/ops/triton_ops/kernel_utils.py
index f6397a68ddb8..ebfffc17ae87 100644
--- a/vllm/lora/ops/triton_ops/kernel_utils.py
+++ b/vllm/lora/ops/triton_ops/kernel_utils.py
@@ -22,6 +22,7 @@ def mm_k(
     SPLIT_K: tl.constexpr,
     CAST_TYPE: tl.constexpr,
     b_dtype: tl.constexpr,
+    USE_GDC: tl.constexpr,
 ):
     """
     Given a_ptr and b_ptr, that identify the rows of A (m x k) and columns of
@@ -45,19 +46,25 @@ def mm_k(
         CAST_TYPE: if True, cast the values from the A matrix to the B
           matrix dtype.
         b_dtype: datatype of the B matrix
+        USE_GDC: Whether to use PDL. True indicates use.
     """
     accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     for k in range(tl.cdiv(K, BLOCK_K * SPLIT_K)):
         if EVEN_K:
-            tiled_a = tl.load(a_ptr)
+            # pre-fetech lora weight
             tiled_b = tl.load(b_ptr)
+            if USE_GDC:
+                tl.extra.cuda.gdc_wait()
+            tiled_a = tl.load(a_ptr)
         else:
-            tiled_a = tl.load(
-                a_ptr, mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K), other=0
-            )
             tiled_b = tl.load(
                 b_ptr, mask=offset_k[:, None] < K - k * (BLOCK_K * SPLIT_K), other=0
             )
+            if USE_GDC:
+                tl.extra.cuda.gdc_wait()
+            tiled_a = tl.load(
+                a_ptr, mask=offset_k[None, :] < K - k * (BLOCK_K * SPLIT_K), other=0
+            )
         if CAST_TYPE:
             tiled_a = tiled_a.to(b_dtype)
         accumulator += tl.dot(
@@ -102,6 +109,7 @@ def do_expand_kernel(
     EVEN_K: tl.constexpr,
     CAST_TYPE: tl.constexpr,
     ADD_INPUTS: tl.constexpr,
+    USE_GDC: tl.constexpr,
 ):
     """
     Given an array of integers that identifies the rows of A, ram,
@@ -154,6 +162,7 @@ def do_expand_kernel(
 
     # Compute the block matrix product.
     SPLIT_K = 1
+
     accumulator = mm_k(
         a_ptr,
         b_ptr,
@@ -168,6 +177,7 @@ def do_expand_kernel(
         SPLIT_K,
         CAST_TYPE,
         cur_lora_ptr.dtype.element_ty,
+        USE_GDC,
     )
 
     tiled_c = accumulator.to(cur_lora_ptr.dtype.element_ty)
@@ -223,6 +233,7 @@ def do_shrink_kernel(
     EVEN_K: tl.constexpr,
     SPLIT_K: tl.constexpr,
     SLICE_NUM: tl.constexpr,
+    USE_GDC: tl.constexpr,
 ):
     """
     Given an array of integers that identifies the rows of A, ram,
@@ -272,8 +283,11 @@ def do_shrink_kernel(
         SPLIT_K,
         False,
         cur_lora_ptr.dtype.element_ty,
+        False,  # USE_GDC is always False in shrink kernel
     )
-
+    # GDC launch dependents hints the runtime system to launch dependent kernels.
+    if USE_GDC:
+        tl.extra.cuda.gdc_launch_dependents()
     # Identify the C output pointers to store the results of the accumulator.
     offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
     offset_cm = tl.arange(0, BLOCK_M)
@@ -284,10 +298,10 @@ def do_shrink_kernel(
         + offset_cn[None, :] * output_d2_stride
     )
     c_mask = (offset_cm[:, None] < M_LEN) & (offset_cn[None, :] < N)
-
     accumulator *= scaling
+
     # handles write-back with reduction-splitting
     if SPLIT_K == 1:
         tl.store(c_ptr, accumulator, mask=c_mask)
     else:
-        tl.atomic_add(c_ptr, accumulator, mask=c_mask)
+        tl.atomic_add(c_ptr, accumulator, mask=c_mask, sem="relaxed")
diff --git a/vllm/lora/ops/triton_ops/lora_expand_op.py b/vllm/lora/ops/triton_ops/lora_expand_op.py
index fd4c1364de7e..7f7d70cdc3a4 100644
--- a/vllm/lora/ops/triton_ops/lora_expand_op.py
+++ b/vllm/lora/ops/triton_ops/lora_expand_op.py
@@ -14,6 +14,8 @@
 from vllm.triton_utils import tl, triton
 from vllm.utils.torch_utils import direct_register_custom_op
 
+from .utils import supports_pdl
+
 
 @triton.jit
 def _lora_expand_kernel(
@@ -45,6 +47,7 @@ def _lora_expand_kernel(
     CAST_TYPE: tl.constexpr,
     SLICE_NUM: tl.constexpr,
     SAME_STRIDE: tl.constexpr,
+    USE_GDC: tl.constexpr,
 ):
     cta_n_num = tl.cdiv(N, BLOCK_N)
     cta_m_num = tl.cdiv(M, BLOCK_M)
@@ -121,6 +124,7 @@ def _lora_expand_kernel(
         EVEN_K,
         CAST_TYPE,
         ADD_INPUTS,
+        USE_GDC,
     )
 
 
@@ -236,7 +240,7 @@ def _lora_expand(
         # thread blocks simply exit.
         MAX_LORAS,
     )
-
+    use_gdc = supports_pdl(inputs.device)
     _lora_expand_kernel[grid](
         inputs,
         lora_ptr_tensor,
@@ -266,9 +270,11 @@ def _lora_expand(
         CAST_TYPE,
         NUM_SLICES,
         same_stride,
+        use_gdc,
         num_warps=NUM_WARPS,
         num_ctas=NUM_CTAS,
         num_stages=NUM_STAGES,
+        launch_pdl=use_gdc,
     )
 
     return
diff --git a/vllm/lora/ops/triton_ops/lora_shrink_op.py b/vllm/lora/ops/triton_ops/lora_shrink_op.py
index adc5c9dce5e8..e78379cf684a 100644
--- a/vllm/lora/ops/triton_ops/lora_shrink_op.py
+++ b/vllm/lora/ops/triton_ops/lora_shrink_op.py
@@ -14,6 +14,8 @@
 from vllm.triton_utils import tl, triton
 from vllm.utils.torch_utils import direct_register_custom_op
 
+from .utils import supports_pdl
+
 
 @triton.jit
 def _lora_shrink_kernel(
@@ -43,6 +45,7 @@ def _lora_shrink_kernel(
     SPLIT_K: tl.constexpr,
     GROUP_SIZE_M: tl.constexpr,
     SLICE_NUM: tl.constexpr,
+    USE_GDC: tl.constexpr,
 ):
     cta_n_num = tl.cdiv(N, BLOCK_N)
     cta_m_num = tl.cdiv(M, BLOCK_M)
@@ -83,7 +86,6 @@ def _lora_shrink_kernel(
     cta_lora_seq_indices = (
         token_indices_sorted_by_lora_ids + lora_m_indices_start + cta_m_offset
     )
-
     # Load all relevant row indices.
     offset_m = tl.arange(0, BLOCK_M) % cta_m_len
     ram = tl.load(cta_lora_seq_indices + offset_m)
@@ -118,6 +120,7 @@ def _lora_shrink_kernel(
         EVEN_K,
         SPLIT_K,
         SLICE_NUM,
+        USE_GDC,
     )
 
 
@@ -217,7 +220,7 @@ def _lora_shrink(
         # thread blocks exit early.
         MAX_LORAS,
     )
-
+    use_gdc = supports_pdl(inputs.device)
     _lora_shrink_kernel[grid](
         inputs,
         lora_ptr_tensor,
@@ -245,9 +248,11 @@ def _lora_shrink(
         SPLIT_K,
         GROUP_SIZE_M,
         NUM_SLICES,
+        use_gdc,
         num_warps=NUM_WARPS,
         num_ctas=NUM_CTAS,
         num_stages=NUM_STAGES,
+        launch_pdl=use_gdc,
     )
 
     return
diff --git a/vllm/lora/ops/triton_ops/utils.py b/vllm/lora/ops/triton_ops/utils.py
index bd413a6db26b..8ed42382e3a8 100644
--- a/vllm/lora/ops/triton_ops/utils.py
+++ b/vllm/lora/ops/triton_ops/utils.py
@@ -3,6 +3,7 @@
 
 import functools
 import json
+from functools import lru_cache
 from pathlib import Path
 from typing import Any
 
@@ -10,6 +11,7 @@
 
 from vllm import envs
 from vllm.logger import init_logger
+from vllm.platforms import current_platform
 
 logger = init_logger(__name__)
 
@@ -282,3 +284,12 @@ def get_lora_op_configs(
 
     assert config_data is not None
     return config_data
+
+
+@lru_cache
+def supports_pdl(device: torch.device | None = None) -> bool:
+    """
+    Refer to: https://github.com/triton-lang/triton/blob/v3.5.0/python/tutorials/11-programmatic-dependent-launch.py
+    """
+    # PDL requires compute capability SM90 or above
+    return current_platform.is_cuda() and current_platform.has_device_capability(90)

```
