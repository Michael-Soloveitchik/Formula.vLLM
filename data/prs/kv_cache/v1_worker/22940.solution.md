# PR #22940: [Core] direct indexing on self.block_table_np in compute_slot_mapping

- field: kv_cache  | area: v1_worker
- author: linzebing
- state: MERGED  created: 2025-08-14T22:00:36Z  merged: 2025-08-15T19:12:13Z
- url: https://github.com/vllm-project/vllm/pull/22940
- labels: ready, v1

# Issue

## Problem

In the V1 worker's `compute_slot_mapping`, block numbers were looked up by first fetching the CPU-side block table as a Torch tensor, calling `.flatten()` on it, fancy-indexing with `block_table_indices`, then converting the result back to NumPy with `.numpy()`. `torch.Tensor.flatten()` forces a full copy of the block-table tensor on every call, and the Torch↔NumPy round-trip adds extra conversion overhead, making a per-step hot-path operation needlessly expensive.

## Example

Profiling `facebook/opt-125m` with input-len 800 / output-len 75, `compute_slot_mapping` cost 400+μs per invocation before the change.

## Use case

Runs every decode/prefill step to map token positions into KV-cache slots, so it sits directly in the per-step serving hot path.

## Resources

CPU compute and memory bandwidth — redundant full-tensor copies and dtype/framework conversions burn CPU cycles and allocations on the critical path.

## Type of users

Anyone serving with the vLLM V1 engine; impact scales with throughput-bound workloads where per-step CPU overhead limits request rate.

## Kind of inputs

High request-count, high-throughput batches where `compute_slot_mapping` fires frequently; cost grows with block-table size (num requests × max_num_blocks_per_req).

## Mechanism with the issue

`block_table.py:compute_slot_mapping` built `block_numbers` via `self.get_cpu_tensor().flatten()[block_table_indices].numpy()`. `flatten()` copies the whole block-table tensor and `.numpy()` adds a conversion step, both redundant since a NumPy view of the same buffer already exists.

# Solution

## Fix in words

Diff drops the Torch path entirely. Before: `block_table_cpu = self.get_cpu_tensor(); block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()`. After: `block_numbers = self.block_table_np.ravel()[block_table_indices]`. `np.ravel()` returns a view (no copy) of the existing NumPy block table, and indexing stays in NumPy so the Torch flatten-copy and `.numpy()` conversion are eliminated. Cuts `compute_slot_mapping` from 400+μs to 15-30μs and lifts opt-125m throughput ~2.07%.

## Author's solution

Author replaced Torch tensor flattening and conversion with direct NumPy indexing via `ravel` on `self.block_table_np`, eliminating redundant copies and framework conversions in slot-mapping computation.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/worker/block_table.py`

### Diff
```diff
diff --git a/vllm/v1/worker/block_table.py b/vllm/v1/worker/block_table.py
index bf38e88f0c2a..5662fc350e19 100644
--- a/vllm/v1/worker/block_table.py
+++ b/vllm/v1/worker/block_table.py
@@ -91,8 +91,7 @@ def compute_slot_mapping(self, req_indices: np.ndarray,
         # block_size.
         block_table_indices = (req_indices * self.max_num_blocks_per_req +
                                positions // self.block_size)
-        block_table_cpu = self.get_cpu_tensor()
-        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
+        block_numbers = self.block_table_np.ravel()[block_table_indices]
         block_offsets = positions % self.block_size
         np.add(block_numbers * self.block_size,
                block_offsets,

```
