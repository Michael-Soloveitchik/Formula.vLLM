# PR #30494: [Perf] Optimize deepgemm experts initialization, 3.9% TTFT improvement

- field: gpu_kernels  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2025-12-11T16:15:17Z  merged: 2025-12-11T22:28:34Z
- url: https://github.com/vllm-project/vllm/pull/30494
- labels: performance, ready, deepseek

# Issue

## Problem

DeepGEMM uses negative values in m_indices (expert_ids) to mark invalid/padded blocks its scheduler should skip. The old code chose between torch.zeros and torch.empty to initialize expert_ids based on whether empty blocks might exist. torch.empty left garbage that could be misread as valid expert 0, and torch.zeros forced an extra full-buffer zero-fill plus still marked padded rows as valid expert 0 instead of invalid. Separately, the scatter kernel's inner store had no bounds mask, writing cur_expert past the real token count in the last block.

## Example

Serving Qwen3-VL-235B-A22B-Thinking-FP8 with tp 8 and expert parallel, prefill with random-input-len 4 over 256 prompts: mean TTFT drops from 765ms to 736ms (~3.9%).

## Use case

Prefill/TTFT path of MoE inference where tokens get permuted and grouped per-expert for DeepGEMM grouped GEMM.

## Resources

GPU compute + kernel time: wasted DeepGEMM work on padded blocks that should be skipped, plus an unnecessary full-buffer zero-fill.

## Type of users

Users serving large MoE FP8 models on multi-GPU TP with expert parallel using the DeepGEMM backend.

## Kind of inputs

Batches where per-expert token counts don't align to block size, leaving padded/aligned regions, and configs where some experts get zero tokens.

## Mechanism with the issue

deepgemm_moe_permute in deep_gemm_utils.py allocated expert_ids via zeros-or-empty heuristic, and _fwd_kernel_ep_scatter_1 stored cur_expert for a full BLOCK_E range without masking, overwriting aligned-region rows past cur_expert_token_num. Padded rows thus got valid expert ids (0 or garbage) instead of -1, so DeepGEMM's scheduler processed blocks it should skip.

# Solution

## Fix in words

Replaces the torch.zeros/torch.empty heuristic with an unconditional torch.full(..., fill_value=-1) so expert_ids starts all -1, marking every row invalid by default. The scatter kernel now computes offs = start_m + off_expert and a mask offs < cur_expert_token_num, passing mask= to tl.store so only real-token rows get cur_expert written. Rows in the aligned/padded region stay at -1. Net effect: padded blocks are reliably skipped by DeepGEMM's scheduler with no extra zero-fill pass, cutting wasted prefill compute and improving TTFT ~3.9% with no accuracy change.

## Author's solution

DeepGEMM uses negative m_indices values to flag completely invalid/padded blocks for skipping. Author always initializes expert_ids to -1 so any row not explicitly written by the scatter kernel is treated as invalid and skipped by DeepGEMM's scheduler. Validated with Qwen3-VL-235B FP8: ~3.9% TTFT improvement and unchanged gsm8k accuracy.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py b/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
index 6cca95412327..57d303cd53fe 100644
--- a/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
+++ b/vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
@@ -84,10 +84,16 @@ def _fwd_kernel_ep_scatter_1(
     m_indices_start_ptr = m_indices + cur_expert_start
     off_expert = tl.arange(0, BLOCK_E)
 
+    # any rows in the per-expert aligned region that do not correspond to
+    # real tokens are left untouched here and should remain initialized to
+    # -1 so DeepGEMM can skip them
     for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
+        offs = start_m + off_expert
+        mask = offs < cur_expert_token_num
         tl.store(
-            m_indices_start_ptr + start_m + off_expert,
+            m_indices_start_ptr + offs,
             cur_expert,
+            mask=mask,
         )
 
 
@@ -366,12 +372,17 @@ def deepgemm_moe_permute(
         (M_sum, H // block_k), device=device, dtype=torch.float32
     )
 
-    maybe_has_empty_blocks = (expert_tokens_meta is None) or (
-        expert_tokens_meta.expert_num_tokens_cpu is None
+    # DeepGEMM uses negative values in m_indices (here expert_ids) to mark
+    # completely invalid / padded blocks that should be skipped. We always
+    # initialize expert_ids to -1 so any row that is not explicitly written
+    # by the scatter kernel will be treated as invalid and skipped by
+    # DeepGEMM's scheduler.
+    expert_ids = torch.full(
+        (M_sum,),
+        fill_value=-1,
+        device=device,
+        dtype=torch.int32,
     )
-    expert_ids_init = torch.zeros if maybe_has_empty_blocks else torch.empty
-
-    expert_ids = expert_ids_init((M_sum), device=device, dtype=torch.int32)
     inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)
 
     expert_num_tokens = None

```
