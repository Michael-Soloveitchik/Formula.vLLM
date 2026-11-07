# PR #38559: [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement

- field: scheduling  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2026-03-30T15:46:57Z  merged: 2026-04-01T03:54:58Z
- url: https://github.com/vllm-project/vllm/pull/38559
- labels: ready

# Issue

## Problem

Mean pooling computed a full `torch.cumsum` over the entire `hidden_states` tensor in float32, then gathered per-sequence sums via `cumsum[end] - cumsum[start] + hidden_states[start]`. The cumsum allocates a large temporary float32 tensor the same shape as all tokens stacked together, and computes a prefix sum across every token even though only per-segment totals are needed. This is redundant compute plus a big transient memory spike.

## Example

Serving `intfloat/multilingual-e5-base` with max-model-len 512 and batch size 1 at concurrency 64, 2000 prompts of 512 tokens each (1,024,000 input tokens total). The cumsum temp tensor spans all stacked tokens × hidden_size in float32.

## Use case

Embedding/pooling inference where each request's token hidden states must be mean-pooled into one vector per sequence (text embedding serving).

## Resources

GPU HBM (large float32 cumsum temp tensor) and GPU compute (full prefix-sum pass over all tokens).

## Type of users

Users serving pooling/embedding models (`--runner pooling`) with mean pooling, especially high-concurrency batched embedding workloads.

## Kind of inputs

Batches of many sequences being pooled together; long or numerous token rows make the cumsum temp tensor and prefix-sum pass costly.

## Mechanism with the issue

In `MeanPool.forward` (vllm/model_executor/layers/pooler/seqwise/methods.py), `torch.cumsum(hidden_states, dim=0, dtype=torch.float32)` builds a float32 prefix-sum tensor over the full stacked token tensor, then indexes it with `first_token_indices_gpu`/`last_token_indices_gpu`. Computing a global cumsum is wasteful since only per-segment sums are needed, and the float32 cast of the whole tensor spikes memory.

# Solution

## Fix in words

Replaces the full-tensor `torch.cumsum` + index-difference approach with segment accumulation via `index_add_`. It builds `segment_ids` using `torch.repeat_interleave` (e.g. prompt_lens [2,1,3] → [0,0,1,2,2,2]) mapping each token row to its sequence, then accumulates into a `(num_seqs, hidden_size)` float32 `segment_sums` tensor. Accumulation iterates `hidden_states` in ~16MB chunks, casting each chunk to float32 and calling `segment_sums.index_add_`, bounding the transient float32 cast to one chunk instead of the whole tensor. Finally divides `segment_sums` by `prompt_lens` for the means. Adds an early return `new_empty((0, hidden_size))` for empty batches and casts `prompt_lens` to int64. Yields 5.9% E2E throughput gain by avoiding the redundant global prefix sum and large temp allocation.

## Author's solution

Optimize mean pooling using chunks and `index_add`, avoiding redundant calculation from `cumsum` and the large temporary tensor. Accuracy covered by existing MTEB and online pooling unit tests; perf validated with `vllm bench serve` on `intfloat/multilingual-e5-base` showing ~5.9% E2E throughput improvement (238.52 → 252.72 req/s).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/pooler/seqwise/methods.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/pooler/seqwise/methods.py b/vllm/model_executor/layers/pooler/seqwise/methods.py
index f3c7f29d6092..b967ff4ede7b 100644
--- a/vllm/model_executor/layers/pooler/seqwise/methods.py
+++ b/vllm/model_executor/layers/pooler/seqwise/methods.py
@@ -14,6 +14,8 @@
 
 SequencePoolingMethodOutput: TypeAlias = torch.Tensor | list[torch.Tensor]
 
+_MEAN_POOL_ACCUMULATION_CHUNK_BYTES = 16 * 1024 * 1024  # 16MB
+
 
 class SequencePoolingMethod(nn.Module, ABC):
     def get_supported_tasks(self) -> Set[PoolingTask]:
@@ -67,19 +69,41 @@ def forward(
         )
 
         prompt_lens = pooling_cursor.prompt_lens_cpu.to(
-            hidden_states.device, non_blocking=True
+            hidden_states.device, dtype=torch.int64, non_blocking=True
+        )
+
+        num_seqs = prompt_lens.numel()
+        hidden_size = hidden_states.shape[-1]
+
+        if num_seqs == 0:
+            # early return for empty batch
+            return hidden_states.new_empty((0, hidden_size), dtype=torch.float32)
+
+        # eg. [2, 1, 3] -> [0, 0, 1, 2, 2, 2]
+        segment_ids = torch.repeat_interleave(
+            torch.arange(num_seqs, device=hidden_states.device, dtype=torch.long),
+            prompt_lens,
+        )
+        segment_sums = torch.zeros(
+            (num_seqs, hidden_size),
+            dtype=torch.float32,
+            device=hidden_states.device,
         )
 
-        # Use float32 for torch.cumsum in MeanPool,
-        # otherwise precision will be lost significantly.
-        cumsum = torch.cumsum(hidden_states, dim=0, dtype=torch.float32)
+        bytes_per_token = hidden_size * torch.finfo(torch.float32).bits // 8
+        chunk_size = max(1, _MEAN_POOL_ACCUMULATION_CHUNK_BYTES // bytes_per_token)
 
-        start_indices = pooling_cursor.first_token_indices_gpu
-        end_indices = pooling_cursor.last_token_indices_gpu
+        # iterate over the batch in chunks
+        for start in range(0, hidden_states.shape[0], chunk_size):
+            end = min(start + chunk_size, hidden_states.shape[0])
+            # using index_add_ to accumulate for each segment
+            segment_sums.index_add_(
+                0,
+                segment_ids[start:end],
+                hidden_states[start:end].to(dtype=torch.float32),
+            )
 
-        return (
-            cumsum[end_indices] - cumsum[start_indices] + hidden_states[start_indices]
-        ) / prompt_lens.unsqueeze(1)
+        return segment_sums / prompt_lens.unsqueeze(1)
 
 
 def get_seq_pooling_method(pooling_type: SequencePoolingType | str):

```
