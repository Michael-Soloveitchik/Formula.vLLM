# PR #37347: [Perf] Optimize token_embed for pooling models, 1.0% token throughput improvement

- field: scheduling  | area: v1_pool
- author: yewentao256
- state: MERGED  created: 2026-03-17T21:50:59Z  merged: 2026-03-19T02:27:51Z
- url: https://github.com/vllm-project/vllm/pull/37347
- labels: ready, v1

# Issue

## Problem

When pooling models build the PoolingCursor, build_pooling_cursor always recomputed the per-sequence token offset array from scratch: it allocated a new int64 tensor of length n_seq+1 on CPU, ran torch.cumsum over num_scheduled_tokens, then copied the result to the GPU. This cumulative sum is identical to query_start_loc, which the model runner has already computed for the same forward pass. Recomputing it duplicates CPU work, a tensor allocation, and a host-to-device copy on every pooling step.

## Example

Serving intfloat/multilingual-e5-small as a pooling runner with max-num-seqs=128, batch=16, 256 tokens/prompt: the redundant cumsum + H2D copy sits on the critical path of every /pooling request, measurably adding to latency.

## Use case

Online embedding/pooling serving (token_embed, embeddings, classification) where each request batch triggers a pooling step after the model forward.

## Resources

CPU cumsum compute, a CPU tensor allocation, and PCIe host-to-device copy / kernel-launch overhead per pooling step.

## Type of users

Users serving pooling/embedding models (embeddings, token classification, multi-vector retrieval) on GPU.

## Kind of inputs

Batched short/medium sequences across many sequences per step; impact grows with number of sequences (n_seq) per scheduled batch.

## Mechanism with the issue

In vllm/v1/pool/metadata.py, build_pooling_cursor unconditionally allocated cumsum = zeros(n_seq+1) on CPU and ran torch.cumsum(num_scheduled_tokens_cpu) then .to(device). The resulting first/last token index tensors equal slices of query_start_loc, which gpu_model_runner already holds (self.query_start_loc.gpu), so the whole recompute-and-copy was wasted duplicate work.

# Solution

## Fix in words

The diff adds an optional query_start_loc_gpu argument to build_pooling_cursor. Before, cumsum was always computed on CPU and copied to device; after, when query_start_loc_gpu is supplied the function reuses it directly as cumsum (with validation that its length == n_seq+1 and that it sits on the same device as the hidden states), skipping the CPU allocation, cumsum, and H2D copy. gpu_model_runner._pool now passes self.query_start_loc.gpu[:num_reqs+1] into the call. The CPU fallback path is retained for when no precomputed cursor is available, yielding ~1% token throughput improvement.

## Author's solution

Reuse the already-computed cumsum (query_start_loc_gpu) instead of recomputing the per-sequence offset prefix sum inside build_pooling_cursor. The author initially also proposed replacing the Python list-comprehension split with a single hidden_states.split(split_sizes), but after review kept only option 2 (cumsum reuse), which alone gave ~1% perf improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/pool/metadata.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/v1/pool/metadata.py b/vllm/v1/pool/metadata.py
index 0764d5e6f7a7..cb386deccfb6 100644
--- a/vllm/v1/pool/metadata.py
+++ b/vllm/v1/pool/metadata.py
@@ -101,6 +101,7 @@ def build_pooling_cursor(
         num_scheduled_tokens_np: np.ndarray,
         seq_lens_cpu: torch.Tensor,
         device: torch.device,
+        query_start_loc_gpu: torch.Tensor | None = None,
     ):
         n_seq = len(num_scheduled_tokens_np)
         prompt_lens = self.prompt_lens
@@ -109,11 +110,25 @@ def build_pooling_cursor(
 
         index = list(range(n_seq))
         num_scheduled_tokens_cpu = torch.from_numpy(num_scheduled_tokens_np)
-        cumsum = torch.zeros(
-            n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"
-        )
-        torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])
-        cumsum = cumsum.to(device, non_blocking=True)
+        if query_start_loc_gpu is None:
+            cumsum = torch.zeros(
+                n_seq + 1, dtype=torch.int64, pin_memory=pin_memory, device="cpu"
+            )
+            torch.cumsum(num_scheduled_tokens_cpu, dim=0, out=cumsum[1:])
+            cumsum = cumsum.to(device, non_blocking=True)
+        else:
+            if query_start_loc_gpu.shape[0] != n_seq + 1:
+                raise ValueError(
+                    "query_start_loc_gpu length does not match "
+                    f"the number of sequences: {query_start_loc_gpu.shape[0]} "
+                    f"!= {n_seq + 1}."
+                )
+            if query_start_loc_gpu.device != device:
+                raise ValueError(
+                    "query_start_loc_gpu must be on the same device as the "
+                    f"hidden states: {query_start_loc_gpu.device} != {device}."
+                )
+            cumsum = query_start_loc_gpu
         self.pooling_cursor = PoolingCursor(
             index=index,
             first_token_indices_gpu=cumsum[:n_seq],
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index af5dca71f9c0..595e8cc39b70 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -2928,7 +2928,10 @@ def _pool(
 
         pooling_metadata = self.input_batch.get_pooling_metadata()
         pooling_metadata.build_pooling_cursor(
-            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
+            num_scheduled_tokens_np,
+            seq_lens_cpu,
+            device=hidden_states.device,
+            query_start_loc_gpu=self.query_start_loc.gpu[: num_reqs + 1],
         )
 
         model = cast(VllmModelForPooling, self.model)

```
