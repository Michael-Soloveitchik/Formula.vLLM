# PR #20308: [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel

- field: gpu_kernels  | area: attention_ops
- author: jvlunteren
- state: MERGED  created: 2025-07-01T08:44:06Z  merged: 2025-07-07T19:08:12Z
- url: https://github.com/vllm-project/vllm/pull/20308
- labels: ready

# Issue

## Problem

The unified Triton attention kernel computed attention over every KV block in the sequence during prefill, iterating `num_blocks = cdiv(seq_len, BLOCK_SIZE)` tiles for each query block. Under causal masking, query tokens early in the prompt can only attend to KV positions at or before themselves, so KV blocks beyond the current query block's maximum reachable position contribute nothing but are still loaded and processed. This wastes compute and memory bandwidth on tiles that are fully masked out.

## Example

For meta-llama/Llama-3.1-8B-Instruct on an H100 at input length 16000 and batch size 1, the wasted tiles dominate: skipping them yields ~1.75x speedup over the previous kernel.

## Use case

Prefill of long prompts in online serving and latency benchmarks, where time-to-first-token is dominated by the prefill attention pass.

## Resources

GPU compute (SM occupancy on matmul tiles) and HBM bandwidth spent loading KV tiles that are entirely masked by the causal constraint.

## Type of users

Users serving or benchmarking long-context models with the Triton unified attention backend (VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1), especially on long prompts.

## Kind of inputs

Long prefill sequences (prompt lengths of thousands to tens of thousands of tokens) under causal attention; benefit grows with sequence length.

## Mechanism with the issue

In `kernel_unified_attention_2d` in vllm/attention/ops/triton_unified_attention.py, the per-query-block tile loop `for j in range(0, num_blocks)` used `num_blocks` derived from the full `seq_len`. Because the kernel iterated to the end of the sequence regardless of which query positions the current q_block covers, it processed KV tiles that lie entirely in the causally-masked future region, doing useless loads and flops.

# Solution

## Fix in words

Before, `num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)` iterated KV tiles over the entire sequence length. After, the kernel computes `max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1`, the longest sequence prefix any query token in the current q_block can attend to under causal masking, clamps it to `seq_len` via `tl.minimum` to account for padding in the last q_block, and sets `num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)`. This bounds the tile loop to only the KV blocks that can contribute, skipping fully-masked future tiles. The effect is fewer loaded/computed tiles per query block during prefill, lowering TTFT and overall prefill latency without changing results.

## Author's solution

The author optimizes prefill attention in the unified Triton kernel by reducing the number of tiles processed during the prefill phase. They leverage the causal mask to skip unnecessary computations—blocks beyond the longest prefix reachable by the current query block—yielding more efficient execution that is especially beneficial for long prompts (~1.75x at 16k tokens, batch size 1).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/attention/ops/triton_unified_attention.py`

### Diff
```diff
diff --git a/vllm/attention/ops/triton_unified_attention.py b/vllm/attention/ops/triton_unified_attention.py
index c65f09523a3c..f9645f651351 100644
--- a/vllm/attention/ops/triton_unified_attention.py
+++ b/vllm/attention/ops/triton_unified_attention.py
@@ -145,7 +145,19 @@ def kernel_unified_attention_2d(
                               mask=query_mask_1,
                               other=0.0)
 
-    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)
+    # compute the length of the longest sequence prefix spanned by any
+    # query token in the current q_block (q_block_local_idx)
+    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (
+        BLOCK_M - 1) // num_queries_per_kv + 1
+
+    # adjust for potential padding in the last q_block by considering the
+    # actual sequence length
+    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
+
+    # calculate the number of tiles (blocks) that need to be processed to
+    # cover the longest sequence prefix (due to causal masking, blocks beyond
+    # this prefix can be skipped)
+    num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)
 
     # iterate through tiles
     for j in range(0, num_blocks):

```
