# PR #41043: [Perf][Spec Decode] Avoid per-step numpy allocation in prepare_next_t…

- field: decoding_tpot  | area: v1_spec_decode
- author: wangluochao902
- state: MERGED  created: 2026-04-27T19:16:14Z  merged: 2026-04-29T21:20:14Z
- url: https://github.com/vllm-project/vllm/pull/41043
- labels: speculative-decoding, ready, v1

# Issue

## Problem

In the speculative decoding path, `prepare_next_token_ids_padded` rebuilds the backup next-token IDs on every decode step using `.tolist()`, a Python list comprehension, and `np.array()`. These three temporaries are redundant because a pre-allocated pinned `CpuGpuBuffer` (`backup_next_token_ids`) already exists to hold the values. The extra per-step allocation wastes CPU work and creates garbage that triggers occasional Python GC pauses.

## Example

Llama-3.1-8B + Eagle3 (TP=4, num_speculative_tokens=3) on H200: mean TPOT 2.60→2.58 ms (0.8%) and P99 TPOT 3.24→2.94 ms (9.3%), the tail win coming from removing GC pauses caused by the per-step `np.array()`.

## Use case

Online serving with speculative decoding (Eagle/Eagle3, MTP, DFlash, draft_model) where every decode step must compute each request's next token ID before the drafter runs.

## Resources

CPU time and memory-allocator/GC pressure on the per-step host path; the transient allocations also add latency jitter (tail TPOT) rather than stressing GPU.

## Type of users

Users running speculative decoding, especially latency-sensitive low-concurrency setups where per-step CPU overhead and GC pauses show up in P99 TPOT.

## Kind of inputs

Decode-phase batches of typically 1–16 requests, hit every single decode step regardless of sequence length or dtype.

## Mechanism with the issue

`prepare_next_token_ids_padded` in `vllm/v1/spec_decode/llm_base_proposer.py` built `seq_lens_list = (num_tokens_no_spec[:num_reqs]-1).tolist()`, then a list comprehension over `requests[...].get_token_id(...)`, then wrapped it in `np.array(dtype=np.int32)` and assigned into the buffer. Three temporary objects are created and immediately copied/discarded each step, when the result could be written straight into the existing pinned numpy buffer. Because the method lives on `SpecDecodeBaseProposer`, the waste applies to all spec-decode methods.

# Solution

## Fix in words

The diff deletes the `.tolist()` call, the list comprehension, and the `np.array()` wrapper. It replaces them with a simple `for i in range(num_reqs)` loop that writes each `requests[gpu_input_batch.req_ids[i]].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)` directly into `self.backup_next_token_ids.np[i]`, the pre-allocated pinned buffer. No temporary Python lists or numpy array are created per step. Since spec-decode batches are small (1–16 requests) the loop is trivially cheap, and eliminating the allocation removes the per-step GC pauses that inflated P99 TPOT.

## Author's solution

The author writes directly into the pre-allocated pinned `CpuGpuBuffer` via a small Python loop over the requests, computing each backup token ID with `get_token_id(num_tokens_no_spec[i] - 1)` and storing it into `backup_next_token_ids.np[i]`. This removes the intermediate `.tolist()`, list comprehension, and `np.array()` allocation. They note the loop is trivially fast for typical spec-decode batch sizes (1–16) and the change applies to all spec-decode methods via `SpecDecodeBaseProposer`.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/spec_decode/llm_base_proposer.py`

### Diff
```diff
diff --git a/vllm/v1/spec_decode/llm_base_proposer.py b/vllm/v1/spec_decode/llm_base_proposer.py
index 1c59704de602..8d38038734b3 100644
--- a/vllm/v1/spec_decode/llm_base_proposer.py
+++ b/vllm/v1/spec_decode/llm_base_proposer.py
@@ -883,16 +883,12 @@ def prepare_next_token_ids_padded(
         is not sampled and comes from `request.get_token_id()` instead. This is denoted
         the "backup" token id. It also counts rejected tokens via `sampled_token_ids`.
         """
-        # Precompute get_token_id for when there is no valid next token
+        # Precompute backup token IDs for discarded requests.
         num_reqs = gpu_input_batch.num_reqs
-        seq_lens_list = (gpu_input_batch.num_tokens_no_spec[:num_reqs] - 1).tolist()
-        self.backup_next_token_ids.np[:num_reqs] = np.array(
-            [
-                requests[gpu_input_batch.req_ids[i]].get_token_id(seq_lens_list[i])
-                for i in range(num_reqs)
-            ],
-            dtype=np.int32,
-        )
+        for i in range(num_reqs):
+            self.backup_next_token_ids.np[i] = requests[
+                gpu_input_batch.req_ids[i]
+            ].get_token_id(gpu_input_batch.num_tokens_no_spec[i] - 1)
         self.backup_next_token_ids.copy_to_gpu(num_reqs)
         backup_tokens_gpu = self.backup_next_token_ids.gpu
 

```
