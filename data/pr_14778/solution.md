# PR #14778: [Model] Mamba2 Prefill Performance Tweaks: Fixing Flurry of Unnecessary Memory Copies 

- field: attention  | area: model_executor_layers
- author: cyang49
- state: MERGED  created: 2025-03-13T19:20:57Z  merged: 2025-03-14T20:36:18Z
- url: https://github.com/vllm-project/vllm/pull/14778
- labels: ready

# Issue

## Problem

In the Mamba2 mixer's `forward_cuda` prefill path, two per-sequence Python `for` loops each issued a separate CUDA kernel/memory operation per batch element. One loop zeroed `ssm_state[idx]` for every sequence lacking an initial state; the other copied `varlen_state[i]` into `ssm_state[idx]` for every sequence. These element-wise `.zero_()` / `.copy_()` calls produced a flurry of tiny, independent device memory copies instead of one batched op, inflating kernel-launch overhead and prefill latency.

## Example

Offline prefill, batch size 64, short prompts: per-Mamba-layer latency was 5ms, dominated by the two copy phases visible in the Nsight Systems trace; after fusing, 3ms.

## Use case

Serving/prefilling Mamba2-based SSM models (e.g. Bamba-9B) where every decoder layer touches the SSM state cache once per forward.

## Resources

HBM/memory-copy bandwidth and CPU-side kernel-launch overhead — many small device ops serialize and stall the GPU rather than saturating memory bandwidth.

## Type of users

Users serving hybrid/SSM (Mamba2) models like Bamba-9B on single GPU (H100-80GB) in batched prefill.

## Kind of inputs

Batched prefill with many concurrent sequences (batch 64, 1000 requests) short-to-medium prompts; cost scales with batch size since each sequence spawns its own loop iteration.

## Mechanism with the issue

`mamba_mixer2.py` `forward_cuda`: the `for idx in state_indices_tensor[~has_initial_states]: ssm_state[idx].zero_()` loop and the `for i, idx in enumerate(state_indices_tensor): ssm_state[idx].copy_(varlen_state[i])` loop each dispatch one kernel per index. Python-level iteration over a `(batch,)` index tensor = O(batch) serial launches. Slow because launch + small-copy overhead dominates the actual data movement.

# Solution

## Fix in words

Both per-element Python loops are replaced with `torch.vmap`-batched functions. Zero-init: `batched_zero_init_func = torch.vmap(lambda idx: ssm_state[idx].zero_())` applied over `state_indices_tensor[~has_initial_states].unsqueeze(-1)`. State update: `batched_copy = torch.vmap(lambda idx, source_state: ssm_state[idx].copy_(source_state))` applied over `state_indices_tensor.unsqueeze(-1)` and `varlen_state`. Indices are unsqueezed to a trailing dim to dodge vmap's ban on `.item()`. The `any(...)` guard also becomes `torch.any(...)`. Effect: the flurry of small copies fuses into batched ops, cutting prefill Mamba-layer latency 5ms→3ms and lifting total token throughput ~8%.

## Author's solution

The author profiled Bamba-9B in Nsight Systems and identified two per-layer phases of frequent, unnecessary memory copies that could be removed or fused. They fused/vectorized these copies to reduce the count, dropping prefill Mamba-layer latency from 5ms to 3ms and improving total token throughput ~8% on an H100-80GB. They note a third similar sequential loop remains unfixed since their test case didn't exercise that path.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/mamba/mamba_mixer2.py b/vllm/model_executor/layers/mamba/mamba_mixer2.py
index b53a540ed662..5b19e3f3554a 100644
--- a/vllm/model_executor/layers/mamba/mamba_mixer2.py
+++ b/vllm/model_executor/layers/mamba/mamba_mixer2.py
@@ -466,10 +466,17 @@ def forward_cuda(
         if has_prefill:
 
             initial_states = None
-            if has_initial_states is not None and any(has_initial_states):
-                for idx in mamba_cache_params.state_indices_tensor[
-                        ~has_initial_states]:
-                    mamba_cache_params.ssm_state[idx].zero_()
+
+            if has_initial_states is not None and torch.any(
+                    has_initial_states):
+
+                # vectorized ssm_state zero init
+                batched_zero_init_func = torch.vmap(
+                    lambda idx: mamba_cache_params.ssm_state[idx].zero_())
+                batched_zero_init_func(
+                    mamba_cache_params.
+                    state_indices_tensor[~has_initial_states].unsqueeze(
+                        dim=-1), )
                 initial_states = mamba_cache_params.ssm_state[
                     mamba_cache_params.state_indices_tensor]
 
@@ -493,10 +500,17 @@ def forward_cuda(
                 dt_limit=(0.0, float("inf")),
             )
 
-            # update ssm states
-            # - varlen state is a (batch, nheads, headdim, dstate) tensor
-            for i, idx in enumerate(mamba_cache_params.state_indices_tensor):
-                mamba_cache_params.ssm_state[idx].copy_(varlen_state[i])
+            # vectorized ssm state update using vmap
+            # the 1d state_indices_tensor needs to be unsqueezed to avoid vmap
+            # limitation which doesn't allow use of `item()`
+            # Note: the lambda capture can happen where ssm_state is initialized
+            #       instead of here
+            batched_copy = torch.vmap(
+                lambda idx, source_state: mamba_cache_params.ssm_state[
+                    idx].copy_(source_state))
+            batched_copy(
+                mamba_cache_params.state_indices_tensor.unsqueeze(dim=-1),
+                varlen_state)
 
             # - reshape
             hidden_states = scan_output.view(seq_len, -1)

```
