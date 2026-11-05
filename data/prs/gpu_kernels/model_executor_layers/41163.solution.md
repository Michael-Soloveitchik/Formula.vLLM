# PR #41163: [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark

- field: gpu_kernels  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2026-04-28T19:41:29Z  merged: 2026-04-29T15:11:05Z
- url: https://github.com/vllm-project/vllm/pull/41163
- labels: ready

# Issue

## Problem

AllPool.forward materializes per-request slices with a Python list comprehension that first calls `.tolist()` on two GPU tensors (`first_token_indices_gpu` and `last_token_indices_gpu`), then builds one slice per request. The two `.tolist()` calls each force a host-visible device sync, stalling the CPU on the GPU, and producing N individual slice ops instead of one bulk operation. This makes the method-level cost dominated by launch/sync overhead rather than useful work.

## Example

Benchmark with 64 requests of length 128, operating on a 32-request subgroup (offset 16, size 32), hidden size 4096, bf16: median forward time 59.89 us on main vs 39.65 us after the fix (~51% faster).

## Use case

Token-wise pooling (e.g. token_classify) in the pooling/embedding serving path, where DispatchPooler hands the full hidden_states tensor to AllPool for a subgroup of requests.

## Resources

Kernel-launch / host-device sync overhead and CPU time: repeated GPU→CPU transfers (`.tolist()`) and N separate slice kernels stress the CPU-GPU synchronization boundary, not raw GPU compute.

## Type of users

Users serving pooling / classification / embedding models through vLLM's token-wise pooler path, especially with many short requests per batch where per-request overhead dominates.

## Kind of inputs

Batches with many requests (high request count, modest per-request sequence length) routed to token-wise pooling; the more requests in the subgroup, the more the two per-batch syncs plus N slices cost.

## Mechanism with the issue

In `vllm/model_executor/layers/pooler/tokwise/methods.py`, `AllPool.forward` built `hidden_states_lst` by zipping `first_token_indices_gpu.tolist()` and `last_token_indices_gpu.tolist()` and slicing `hidden_states[first:last+1]` per request. The two `.tolist()` calls are blocking device syncs, and the comprehension issues one slice per request — O(N) ops plus two syncs every forward.

# Solution

## Fix in words

The diff replaces the two `.tolist()` syncs and the per-request slice comprehension with a single bulk path. It reads `num_scheduled_tokens_cpu.tolist()` (already on CPU, no device sync) to get `split_sizes`, computes the subgroup bounds with two scalar `.item()` calls (`group_start` from `first_token_indices_gpu[0]`, `group_end` from `last_token_indices_gpu[-1]+1`), slices the full tensor to the subgroup once, then calls `hidden_states_group.split(split_sizes)` to produce the per-request list in one op. An empty-`split_sizes` guard returns `[]`. Net effect: fewer host-device syncs and one slice+split instead of N slices, yielding ~51% speedup at the method level.

## Author's solution

Slice the subgroup out of the full hidden_states tensor once using its group start/end bounds, then split that contiguous slice by per-request token counts (`num_scheduled_tokens_cpu`) instead of iterating and slicing per request. This avoids the extra host-visible syncs from `.tolist()`/`.item()` in the hot path and cuts method-level time from 59.89 us to 39.65 us (~51% faster).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/pooler/tokwise/methods.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/pooler/tokwise/methods.py b/vllm/model_executor/layers/pooler/tokwise/methods.py
index 523e8e4c9a9b..d3fefb745cfe 100644
--- a/vllm/model_executor/layers/pooler/tokwise/methods.py
+++ b/vllm/model_executor/layers/pooler/tokwise/methods.py
@@ -47,13 +47,17 @@ def forward(
         pooling_metadata: PoolingMetadata,
     ) -> list[TokenPoolingMethodOutputItem]:
         pooling_cursor = pooling_metadata.get_pooling_cursor()
-        hidden_states_lst = [
-            hidden_states[first : last + 1]
-            for first, last in zip(
-                pooling_cursor.first_token_indices_gpu.tolist(),
-                pooling_cursor.last_token_indices_gpu.tolist(),
-            )
-        ]
+        split_sizes = pooling_cursor.num_scheduled_tokens_cpu.tolist()
+        if split_sizes:
+            # DispatchPooler passes the full hidden_states tensor.
+            # slice out the subgroup once, then split it by
+            # per-request token counts
+            group_start = int(pooling_cursor.first_token_indices_gpu[0].item())
+            group_end = int(pooling_cursor.last_token_indices_gpu[-1].item()) + 1
+            hidden_states_group = hidden_states[group_start:group_end]
+            hidden_states_lst = list(hidden_states_group.split(split_sizes))
+        else:
+            hidden_states_lst = []
 
         if not self.enable_chunked_prefill:
             return hidden_states_lst

```
