# PR #38451: [Perf] Fix DBO overlap: capture DeepEP event before yield

- field: scheduling  | area: model_executor_layers
- author: czhu-cohere
- state: MERGED  created: 2026-03-29T03:47:48Z  merged: 2026-03-31T20:39:00Z
- url: https://github.com/vllm-project/vllm/pull/38451
- labels: ready

# Issue

## Problem

In dual-batch overlap (DBO) for the DeepEP high-throughput all2all backend, compute and communication were not actually overlapping. The DeepEP dispatch/combine kernels receive a `previous_event` captured on the compute stream, and DeepEP waits for that event before launching. Because the event was captured AFTER yielding from compute to comm, the compute stream tail already contained work enqueued by the OTHER ubatch, so DeepEP waited on unrelated compute. This serialized dispatch/combine against the other ubatch's compute, defeating the point of DBO.

## Example

Serving Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 on 4xB200 with DEP4 (TP1, DP4, expert-parallel, deepep_high_throughput), only one ubatch's combine overlapped with the other ubatch's non-MoE compute; prefill benchmark (input-len 8192, 640 prompts, concurrency 64) took 53.22s before vs 40.33s after.

## Use case

Prefill-heavy MoE inference on multi-GPU expert-parallel deployments using the DeepEP HT all2all backend with dual-batch overlap enabled.

## Resources

GPU compute and inter-GPU communication bandwidth (NVLink/RDMA) — the two were running serially instead of overlapped, wasting both.

## Type of users

Users serving large MoE models with expert parallelism (EP/DEP) across multiple GPUs (e.g. B200), FP8 quantized, using DeepEP high-throughput backend with DBO.

## Kind of inputs

Long-input prefill batches (input-len ~8192) with enough concurrency to form two microbatches for DBO; MoE routing with bf16 combine tensors.

## Mechanism with the issue

In `deepep_ht.py`, `_do_dispatch` and `_finalize` called `dbo_yield_and_switch_from_compute_to_comm()` and then `dbo_get_previous_event(self.buffer.capture)`. After the yield, the compute stream had the other ubatch's kernels enqueued, so the captured event spanned that unrelated work. DeepEP honors `previous_event` as an internal dependency, so dispatch/combine blocked until the other ubatch's compute finished, serializing the streams.

# Solution

## Fix in words

The diff moves the `previous_event = dbo_get_previous_event(self.buffer.capture)` call to BEFORE the `dbo_yield_and_switch_from_compute_to_comm()` call, in both `_do_dispatch` and `_finalize`. Before: event captured after yield, so it covered the current ubatch's compute plus the other ubatch's enqueued compute. After: event captures only the current ubatch's compute tail, so DeepEP's internal dependency only waits on the work it actually needs. This lets dispatch/combine launch without waiting for the other ubatch, enabling real compute/comm overlap while preserving the true data dependency.

## Author's solution

Author found via profiling that the DeepEP HT backend wasn't fully overlapping compute and comm because the `previous_event` passed to dispatch/combine was captured after yielding, so it waited on the whole compute stream including the other ubatch's work. Their fix moves the DeepEP event capture to right before the yield so it correctly captures only the current ubatch's compute tail, increasing overlap while keeping the core dependency intact.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py b/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
index 63312557d85d..61777fd99031 100644
--- a/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
+++ b/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py
@@ -107,15 +107,17 @@ def _do_dispatch(
     ) -> Callable:
         has_scales = token_scales is not None
 
+        # Capture a DeepEP event on the compute stream before yielding.
+        # This must happen before the yield so the event only covers this
+        # ubatch's compute work. If captured after, the compute stream tail
+        # may include the other ubatch's work, preventing overlap.
+        previous_event = dbo_get_previous_event(self.buffer.capture)
+
         # We yield before launching the dispatch kernel since the dispatch
         # kernel will block the CPU so we want to queue up all the compute
         # for the other ubatch before the dispatch kernel starts.
         dbo_yield_and_switch_from_compute_to_comm()
 
-        # capture a DeepEP event and pass it as previous_event so
-        # DeepEP honors the dependency internally.
-        previous_event = dbo_get_previous_event(self.buffer.capture)
-
         (
             num_tokens_per_rank,
             num_tokens_per_rdma_rank,
@@ -357,11 +359,11 @@ def _finalize(
                 topk_ids=topk_ids,
                 apply_router_weight_on_input=apply_router_weight_on_input,
             )
+        previous_event = dbo_get_previous_event(self.buffer.capture)
         dbo_yield_and_switch_from_compute_to_comm()
         assert fused_expert_output.dtype == torch.bfloat16, (
             f"Expected fused_expert_output bfloat16, got {fused_expert_output.dtype}"
         )
-        previous_event = dbo_get_previous_event(self.buffer.capture)
         combined_x, _, event = self.buffer.combine(
             # HT combine only supports BF16
             x=fused_expert_output,

```
