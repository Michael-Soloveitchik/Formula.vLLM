# PR #36723: [DSV3.2][MTP] Optimize Indexer MTP handling

- field: attention  | area: v1_attention
- author: benchislett
- state: MERGED  created: 2026-03-10T23:43:27Z  merged: 2026-03-11T04:16:56Z
- url: https://github.com/vllm-project/vllm/pull/36723
- labels: ready, v1

# Issue

## Problem

The MLA indexer's MTP batch-expansion path used torch.repeat_interleave without specifying output_size. Because the output shape is derived from the data-dependent tensor decode_lens, PyTorch must read that tensor's values on the CPU to size the output, forcing a GPU→CPU synchronization. This sync stalls the pipeline and breaks async (overlapped) scheduling, causing large throughput loss.

## Example

DSV3.2 NVFP4 with MTP=3 on 8xB200 TP8: removing the sync restored async scheduling and yielded ~30% speedup. GSM8k held at 95% accuracy with FP8 KV cache.

## Use case

Speculative decoding (multi-token prediction) serving of DeepSeek-V3.2 with the MLA indexer attention backend, where decode batches are expanded per request.

## Resources

Kernel-launch/scheduling pipeline and GPU utilization — the forced GPU/CPU sync serializes host and device, idling the GPU and defeating async scheduling.

## Type of users

Users running DSV3.2 / MLA-indexer models with MTP speculative decoding, quantized (NVFP4 + FP8 KV cache), on multi-GPU TP setups (e.g. 8xB200 TP8).

## Kind of inputs

Decode batches with per-request speculative query lengths, where decode_lens is a GPU tensor whose contents determine the expanded buffer size.

## Mechanism with the issue

In indexer.py build(), three torch.repeat_interleave calls (expanded_base, expanded_starts, expanded_block_table) inferred output length from the GPU tensor decode_lens. Without output_size, repeat_interleave must sync the device to CPU to allocate the correctly-sized output, blocking the async scheduler each step. Introduced by #34552 (v0.17.0).

# Solution

## Fix in words

Diff adds output_size=actual_expanded to all three torch.repeat_interleave calls in indexer.py build(): expanded_base, expanded_starts, and the expanded_block_table_buffer (dim=0) expansion. actual_expanded is already known on the host, so passing it lets repeat_interleave allocate the output without reading decode_lens on CPU. This eliminates the GPU→CPU sync, so the call becomes non-blocking and async scheduling is fully restored. Result: ~30% speedup with unchanged accuracy.

## Author's solution

Author first considered a non-blocking implementation exploiting uniform query lengths (consistent shapes in the common speculative-decoding case), but adopted MatthewBonanni's simpler suggestion: pass output_size to each repeat_interleave call to avoid the sync, which works just as well. Verified via instrumentation, profiling, and benchmarking that async scheduling was fully restored, with ~30% speedup on DSV3.2 NVFP4 MTP=3 8xB200 TP8 and GSM8k at 95%.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/backends/mla/indexer.py`

### Diff
```diff
diff --git a/vllm/v1/attention/backends/mla/indexer.py b/vllm/v1/attention/backends/mla/indexer.py
index d94055cbe46b..f8ff2fc2e76d 100644
--- a/vllm/v1/attention/backends/mla/indexer.py
+++ b/vllm/v1/attention/backends/mla/indexer.py
@@ -384,12 +384,14 @@ def build(
 
                 # [7, 6, 8, 0] -> [7, 7, 7, 6, 8, 8, 8, 8]
                 expanded_base = torch.repeat_interleave(
-                    seq_lens - decode_lens, decode_lens
+                    seq_lens - decode_lens, decode_lens, output_size=actual_expanded
                 )
 
                 # [0, 3, 4, 8] -> [0, 0, 0, 3, 4, 4, 4, 4]
                 expanded_starts = torch.repeat_interleave(
-                    common_attn_metadata.query_start_loc[:num_decodes], decode_lens
+                    common_attn_metadata.query_start_loc[:num_decodes],
+                    decode_lens,
+                    output_size=actual_expanded,
                 )
 
                 # [0, 1, 2, 0, 0, 1, 2, 3]
@@ -407,7 +409,9 @@ def build(
                 # Give each of the flattened entries the same block table row as the
                 # original request.
                 self.expanded_block_table_buffer[:actual_expanded] = (
-                    torch.repeat_interleave(block_table, decode_lens, dim=0)
+                    torch.repeat_interleave(
+                        block_table, decode_lens, dim=0, output_size=actual_expanded
+                    )
                 )
                 if actual_expanded < num_decode_tokens:
                     self.expanded_block_table_buffer[

```
