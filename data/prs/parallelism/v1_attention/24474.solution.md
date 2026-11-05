# PR #24474: [Perf] Convert np array to torch tensor to index into block table for attn chunking

- field: parallelism  | area: v1_attention
- author: shinoss
- state: MERGED  created: 2025-09-09T02:37:49Z  merged: 2025-09-10T03:01:06Z
- url: https://github.com/vllm-project/vllm/pull/24474
- labels: ready, v1

# Issue

## Problem

`make_local_attention_virtual_batches` builds a block table for chunked local attention by advanced-indexing a torch tensor (`block_table`) with two numpy int arrays (`batch_indices`, `block_indices`). An upstream PyTorch change (pytorch/pytorch#160256) altered how a torch tensor is expanded/broadcast against numpy array indices, introducing a large overhead on this indexing operation. As a result the single indexing line became ~10x+ slower than before.

## Example

On local benchmarks the line `block_table_local = block_table[batch_indices, block_indices].view(virtual_batches, -1)` took 10+ ms, up from <1 ms before the upstream torch change, measured on Llama-4-Maverick-17B-128E-Instruct-FP8 at tp=8, max-model-len 4096.

## Use case

Serving Llama 4 models that use chunked local attention, where virtual batches must be constructed every forward pass to map queries to KV blocks within local attention windows.

## Resources

CPU time and per-step kernel/op dispatch latency on the attention metadata build path; the regression inflates host-side prep time added to every batch.

## Type of users

Users serving Llama 4 (Scout/Maverick) with chunked local attention on multi-GPU tensor-parallel setups, especially on torch 2.8.0 where the upstream indexing regression is present.

## Kind of inputs

Any batch requiring local-attention virtual batching; the cost scales with `actual_batch_size`, `local_blocks`, and `pages_per_local_batch` since `batch_indices`/`block_indices` sizes grow with these.

## Mechanism with the issue

In `vllm/v1/attention/backends/utils.py`, `make_local_attention_virtual_batches` indexes `block_table` (torch tensor) with numpy arrays `batch_indices` and `block_indices`. After pytorch/pytorch#160256, mixing numpy-array indices with a torch tensor triggers a slow expansion path inside torch's advanced indexing, making this one line the dominant cost of the function.

# Solution

## Fix in words

The diff converts the two numpy index arrays to torch tensors with `torch.from_numpy(batch_indices)` and `torch.from_numpy(block_indices)` before indexing into `block_table`. The indexing line then uses `block_table[batch_indices_torch, block_indices_torch]` instead of the raw numpy arrays. This avoids the slow torch-tensor-expansion path triggered by numpy-array indices after pytorch/pytorch#160256, restoring the operation from 10+ ms back to <1 ms. A comment documents the upstream PyTorch regression as the reason for the workaround.

## Author's solution

Author found the regression stems from upstream pytorch/pytorch#160256 which slows expanding a torch tensor with numpy arrays. Their workaround converts the numpy arrays to torch tensors before indexing, recovering performance. They also tried converting all numpy ops to torch but that regressed perf by adding an extra sync point, so they kept the minimal conversion.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/backends/utils.py`

### Diff
```diff
diff --git a/vllm/v1/attention/backends/utils.py b/vllm/v1/attention/backends/utils.py
index b286a4ba9fe5..8e3d530fc1f9 100644
--- a/vllm/v1/attention/backends/utils.py
+++ b/vllm/v1/attention/backends/utils.py
@@ -542,7 +542,14 @@ def make_local_attention_virtual_batches(
                                                    1)
     batch_indices = np.repeat(np.arange(actual_batch_size, dtype=np.int32),
                               local_blocks * pages_per_local_batch)
-    block_table_local = block_table[batch_indices, block_indices]\
+
+    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
+    # regression when using numpy arrays (batch and block indices) to index into
+    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
+    # tensor first, which recovers perf.
+    batch_indices_torch = torch.from_numpy(batch_indices)
+    block_indices_torch = torch.from_numpy(block_indices)
+    block_table_local = block_table[batch_indices_torch, block_indices_torch]\
         .view(virtual_batches, -1)
 
     query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)

```
