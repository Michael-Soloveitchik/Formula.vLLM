# PR #13625: [Kernel] Optimize moe intermediate_cache usage

- field: gpu_kernels  | area: model_executor_layers
- author: mgoin
- state: MERGED  created: 2025-02-20T18:34:43Z  merged: 2025-03-03T21:29:53Z
- url: https://github.com/vllm-project/vllm/pull/13625
- labels: ready

# Issue

## Problem

fused_experts_impl allocates three separate intermediate buffers (cache1, cache2, cache3) for the MoE expert computation. cache1 holds the gate/up projection output and cache3 holds the down projection output, but they are never live at the same time: cache1 is fully consumed before cache3 is written. Allocating both independently wastes HBM proportional to tokens x top_k x intermediate_size.

## Example

DeepSeek R1 / DeepSeek-Coder-V2-Lite at 64k-token prefill OOMs; manual measurement of fused_moe shows 3586 MB peak, of which ~512 MB (15%) is the redundant cache3 allocation.

## Use case

Long-sequence prefill serving of large MoE models where many tokens x top_k experts blow up the intermediate activation footprint.

## Resources

GPU HBM / memory capacity — peak allocation at prefill, not compute throughput.

## Type of users

Users serving MoE models (DeepSeek R1, DeepSeek-Coder-V2) at long context lengths.

## Kind of inputs

Large M = num_tokens x top_k (e.g. 64k tokens, topk=2), big intermediate_size, fp16/bf16 activations driving large intermediate_cache tensors.

## Mechanism with the issue

In fused_experts_impl (fused_moe.py), intermediate_cache1 shape (M, top_k, N) and intermediate_cache3 shape (M, top_k, w2.shape[1]) are allocated via separate torch.empty calls. Since cache1 feeds the silu+mul into cache2 and is dead before cache3 is produced by the second GEMM, holding both simultaneously doubles that portion of the activation memory for no data-dependency reason.

# Solution

## Fix in words

Before: cache1 and cache3 each got their own torch.empty. After: one backing buffer cache13 = torch.empty(M * top_k * max(N, w2.shape[1])) is allocated, then both intermediate_cache1 (view as (M, top_k, N)) and intermediate_cache3 (view as (M, top_k, w2.shape[1])) alias slices of it. cache2 stays a separate allocation since it is live concurrently with cache1. Because cache1 is done being used by the time cache3 is written, the shared buffer is safe and cuts peak fused_moe memory ~15% (3586 MB -> 3074 MB at 64k prefill) with no perf change.

## Author's solution

Author first tried an inplace silu+mul kernel (torch.ops._C.silu_and_mul_inplace) but found it simpler to reuse memory between cache1 and cache3 since there is no data dependency between them. They back both with a single tensor, measuring 15% peak memory reduction for 64k prefill fused_moe and confirming gsm8k eval accuracy is unchanged.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe.py b/vllm/model_executor/layers/fused_moe/fused_moe.py
index bc9573b36df7..a9f7429373bc 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -1231,15 +1231,20 @@ def fused_experts_impl(hidden_states: torch.Tensor,
 
     config = get_config_func(M)
 
-    intermediate_cache1 = torch.empty((M, top_k_num, N),
-                                      device=hidden_states.device,
-                                      dtype=hidden_states.dtype)
+    # We can reuse the memory between these because by the time we need
+    # cache3, we're done with cache1
+    cache13 = torch.empty(M * top_k_num * max(N, w2.shape[1]),
+                          device=hidden_states.device,
+                          dtype=hidden_states.dtype)
+    intermediate_cache1 = cache13[:M * top_k_num * N].view(
+        (M, topk_ids.shape[1], N))
+    intermediate_cache3 = cache13[:M * top_k_num * w2.shape[1]].view(
+        (M, topk_ids.shape[1], w2.shape[1]))
+
+    # This needs separate memory since it's used concurrently with cache1
     intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                       device=hidden_states.device,
                                       dtype=hidden_states.dtype)
-    intermediate_cache3 = torch.empty((M, top_k_num, w2.shape[1]),
-                                      device=hidden_states.device,
-                                      dtype=hidden_states.dtype)
 
     if hidden_states.dtype == torch.bfloat16:
         compute_type = tl.bfloat16

```
