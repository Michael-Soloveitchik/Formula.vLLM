# PR #31167: [Perf] Remove blocking copy in GDN Attention

- field: attention  | area: v1_attention
- author: benchislett
- state: MERGED  created: 2025-12-22T17:45:10Z  merged: 2025-12-22T22:25:22Z
- url: https://github.com/vllm-project/vllm/pull/31167
- labels: performance, ready, v1

# Issue

## Problem

In the GDN (Gated Delta Net) attention backend's metadata build path, `context_lens` is copied from CPU to the GPU with a default blocking `.to()` call. A blocking host-to-device copy forces a CUDA synchronization point on the critical path, which stalls the ability to overlap CPU-side scheduling with GPU execution. This defeats async scheduling for Qwen3-Next, where metadata prep should run ahead of the current step.

## Example

Running Qwen3-Next at batch size 1 on 4xB200, the blocking copy in attention metadata build serializes CPU and GPU work; removing it nets a measured +1% end-to-end speedup.

## Use case

Serving Qwen3-Next with async scheduling (and MTP) enabled, where the scheduler tries to build next-step attention metadata while the GPU is still computing the current step.

## Resources

PCIe/host-to-device transfer path and CPU-GPU synchronization/kernel-launch overlap; the blocking copy wastes latency by forcing a sync rather than stressing raw bandwidth.

## Type of users

Users serving Qwen3-Next (GDN/linear-attention hybrid models) on multi-GPU setups like 4xB200 who rely on async scheduling and MTP for low-latency decode.

## Kind of inputs

Low batch size decode (BS1) workloads where per-step CPU scheduling latency is a large fraction of step time and any sync bubble is visible.

## Mechanism with the issue

In `build()` in `vllm/v1/attention/backends/gdn_attn.py`, `context_lens_tensor = context_lens.to(query_start_loc.device)` issues a synchronous H2D copy of `num_computed_tokens_cpu`. Default `.to()` is blocking, inserting an implicit device sync that prevents async scheduling from running metadata prep off the critical path.

# Solution

## Fix in words

Diff changes the single H2D copy in `gdn_attn.py` `build()` from `context_lens.to(query_start_loc.device)` to `context_lens.to(query_start_loc.device, non_blocking=True)`. Before: blocking copy forces a CUDA sync, serializing CPU and GPU. After: non-blocking copy lets the transfer queue on the stream without stalling the host, so scheduling can overlap with GPU work. Effect: removes a sync bubble on the critical path, enabling async scheduling and yielding +1% at BS1 on 4xB200.

## Author's solution

Author removes the blocking copy in GDN attention by making the context_lens host-to-device transfer non-blocking, so it no longer stalls async scheduling for Qwen3-Next. Stated as a first step to unblock making async scheduling + MTP fully non-blocking for Qwen3-Next, with more follow-ups planned. GSM8k accuracy verified unchanged.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/backends/gdn_attn.py`

### Diff
```diff
diff --git a/vllm/v1/attention/backends/gdn_attn.py b/vllm/v1/attention/backends/gdn_attn.py
index ace2cbb0564c..fcde986f48d4 100644
--- a/vllm/v1/attention/backends/gdn_attn.py
+++ b/vllm/v1/attention/backends/gdn_attn.py
@@ -143,7 +143,7 @@ def build(  # type: ignore[override]
 
         query_start_loc = m.query_start_loc
         context_lens = m.num_computed_tokens_cpu
-        context_lens_tensor = context_lens.to(query_start_loc.device)
+        context_lens_tensor = context_lens.to(query_start_loc.device, non_blocking=True)
         nums_dict, batch_ptr, token_chunk_offset_ptr = None, None, None
 
         if (

```
