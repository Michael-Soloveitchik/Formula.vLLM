# PR #24573: [EPLB] Reduce EPLB Inference Overhead

- field: parallelism  | area: model_executor_layers
- author: abmfy
- state: MERGED  created: 2025-09-10T10:28:37Z  merged: 2025-09-22T16:31:05Z
- url: https://github.com/vllm-project/vllm/pull/24573
- labels: ready

# Issue

## Problem

EPLB (Expert Parallelism Load Balancer) replicates logical experts into multiple physical replicas, requiring per-token replica selection and expert-load recording after MoE routing. The original implementation used `torch.rand_like` to pick a replica, which is slow at inference time and not `torch.compile`-friendly, so this EPLB logic stayed uncompiled and added measurable per-decode overhead. It also masked invalid local expert ids before recording load, adding extra tensor ops.

## Example

On DeepSeek-V3-0324 in a DP16 serving setup with 1000 prompts (1000 input / 100 output tokens), enabling EPLB on main raised Mean TPOT from 184.22 ms (no EPLB) to 192.38 ms and dropped total throughput from 31685 to 30427 tok/s.

## Use case

Serving large MoE models (DeepSeek-V3 class) with expert parallelism and EPLB load balancing enabled across many data-parallel ranks.

## Resources

GPU compute plus kernel-launch and `torch.compile` graph-break overhead on the per-token decode path; the uncompiled `torch.rand`-based selection added eager-mode launches each forward.

## Type of users

Users serving large MoE models with expert parallelism / EPLB on multi-GPU TP/DP deployments.

## Kind of inputs

Every routed token in every forward pass triggers it; overhead grows with batch size and `topk_ids` tensor size, independent of sequence content.

## Mechanism with the issue

In `select_experts` (layer.py), the inline block computed `replica_indices = (torch.rand_like(topk_ids) * logical_replica_count[...]).long()` to pick a replica, then did `masked_fill_` on invalid ids and `scatter_add_` to record load. `torch.rand_like` is non-deterministic, slow, and forces a graph break, so the whole EPLB section ran in eager mode on the hot decode path.

# Solution

## Fix in words

The diff extracts all EPLB replica-selection and load-recording logic out of `select_experts` into a new standalone function `eplb_map_to_physical_and_record` decorated with `@torch.compile(dynamic=True, backend=...simple_compile_backend)`. It replaces non-deterministic `torch.rand_like` replica selection with a deterministic modulo scheme: `pos_indices % replica_count` using a flattened `torch.arange` token-position index, so the replica is chosen purely from token position and routing rank `k`. It drops the `masked_fill_`/invalid-mask handling and records load with a plain `scatter_add_` of ones (noted as a compilable substitute for `torch.bincount`) now that global physical expert-load metrics are collected. A CPU/non-CUDA fallback `eplb_map_to_physical_and_record` is added that just returns `topk_ids` unchanged. Net effect: the EPLB path becomes compile-friendly and deterministic, cutting per-token inference overhead.

## Author's solution

Author reduces EPLB overhead three ways: (1) switch from `torch.rand` to a simple modulo-based pseudo-random replica selection keyed on token index and routing rank `k`; (2) remove unnecessary masking of local experts since global physical expert-load metrics are now collected; (3) extract all EPLB logic from `select_experts` into a `torch.compile`-friendly function. Tested with EPLB enabled, `num_redundant_experts=0`, and no rearrangement to isolate replica-selection and load-recording cost on DeepSeek-V3-0324 DP16.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe.py`
- `vllm/model_executor/layers/fused_moe/layer.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe.py b/vllm/model_executor/layers/fused_moe/fused_moe.py
index 6c2a5bda7cba..0e334fdf2404 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -1017,6 +1017,79 @@ def grouped_topk(
     return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
 
 
+@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
+def eplb_map_to_physical_and_record(
+        topk_ids: torch.Tensor,
+        expert_load_view: torch.Tensor,
+        logical_to_physical_map: torch.Tensor,
+        logical_replica_count: torch.Tensor,
+        indices_type: Optional[torch.dtype] = None) -> torch.Tensor:
+    '''
+    Map the logical expert ids to physical expert ids
+    and record the expert load metrics.
+
+    This will select a pseudo-random replica for each logical expert.
+    Only used for EPLB.
+
+    Args:
+        topk_ids: The logical expert ids.
+        expert_load_view: The expert load view.
+        logical_to_physical_map: The logical to physical map.
+        logical_replica_count: The logical replica count.
+        indices_type: The indices type.
+
+    Returns:
+        The physical expert ids.
+    '''
+
+    # 1. Convert the logical expert ids to physical expert ids
+    # Directly select a random replica for each logical expert
+
+    # In case `indices_type` is not `torch.long` or `torch.int`,
+    # e.g. `torch.uint32` as required by dispatch/combine kernels
+    topk_ids_long = topk_ids.long()
+    # Use (token position) modulo (replica count)
+    # to deterministically choose a replica
+    replica_count = logical_replica_count[topk_ids_long]
+    # Flatten-position based index, reshaped back to `topk_ids` shape
+    pos_indices = torch.arange(topk_ids.numel(),
+                               device=topk_ids.device,
+                               dtype=torch.long).reshape_as(topk_ids)
+    # Compute pseudo-random indices by modulo
+    replica_indices = (pos_indices % replica_count).unsqueeze(-1)
+    physical_ids = logical_to_physical_map[topk_ids_long].gather(
+        -1, replica_indices).squeeze(-1)
+
+    topk_ids = physical_ids
+
+    # 2. Record expert load metrics.
+
+    # TODO(bowen): When using `FusedMoEModularKernel`, this
+    # can be done in a more unified way, since
+    # `FusedMoEPrepareAndFinalize` will return the expert
+    # token count, in some cases directly from the kernel.
+    # However, now there are many code paths not using
+    # the modular kernel, e.g. calling `fused_experts`,
+    # so we decide to keep the logic here.
+    #
+    # If later refactor moved all the MoE kernel calls
+    # to the modular kernel, we can move this logic there
+    # to achieve better efficiency.
+
+    # `expert_load_view`: (num_physical_experts,)
+
+    # `torch.bincount` is not compilable, so use `scatter_add_` instead.
+    topk_ids_flatten = topk_ids.flatten()
+    expert_load_view.scatter_add_(
+        dim=0,
+        index=topk_ids_flatten.long(),
+        src=torch.ones_like(topk_ids_flatten).to(expert_load_view))
+
+    if indices_type is not None:
+        topk_ids = topk_ids.to(dtype=indices_type)
+    return topk_ids
+
+
 def fused_grouped_topk(
     hidden_states: torch.Tensor,
     gating_output: torch.Tensor,
diff --git a/vllm/model_executor/layers/fused_moe/layer.py b/vllm/model_executor/layers/fused_moe/layer.py
index da513d75da4d..17ad75584a3f 100644
--- a/vllm/model_executor/layers/fused_moe/layer.py
+++ b/vllm/model_executor/layers/fused_moe/layer.py
@@ -43,7 +43,8 @@
 
 if current_platform.is_cuda_alike():
     from .fused_batched_moe import BatchedTritonExperts
-    from .fused_moe import TritonExperts, fused_experts
+    from .fused_moe import (TritonExperts, eplb_map_to_physical_and_record,
+                            fused_experts)
     if has_pplx():
         from .pplx_prepare_finalize import (PplxPrepareAndFinalize,
                                             pplx_hidden_dim_scale_bytes)
@@ -55,6 +56,16 @@
     fused_experts = None  # type: ignore
     FusedMoEPermuteExpertsUnpermute = None  # type: ignore
     FusedMoEPrepareAndFinalize = None  # type: ignore
+
+    def eplb_map_to_physical_and_record(
+            topk_ids: torch.Tensor, expert_load_view: torch.Tensor,
+            logical_to_physical_map: torch.Tensor,
+            logical_replica_count: torch.Tensor,
+            indices_type: Optional[torch.dtype]) -> torch.Tensor:
+        # CPU fallback: no EPLB so just return as is
+        return topk_ids
+
+
 if is_rocm_aiter_moe_enabled():
     from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
         rocm_aiter_grouped_topk as grouped_topk)
@@ -1616,55 +1627,13 @@ def select_experts(
             assert logical_to_physical_map is not None
             assert logical_replica_count is not None
 
-            # 1. Convert the logical expert ids to physical expert ids
-            # Directly select a random replica for each logical expert
-
-            # TODO: maybe optimize this by using specified kernels,
-            # or compute pseudo-random indices by modulo
-
-            # In case `indices_type` is not `torch.long` or `torch.int`,
-            # e.g. `torch.uint32` as required by dispatch/combine kernels
-            topk_ids_long = topk_ids.long()
-            replica_indices = (
-                torch.rand_like(topk_ids, dtype=torch.float) *
-                logical_replica_count[topk_ids_long]).long().unsqueeze(-1)
-            physical_ids = logical_to_physical_map[topk_ids_long].gather(
-                -1, replica_indices).squeeze(-1)
-
-            topk_ids = physical_ids
-
-            # 2. Record expert load metrics.
-
-            # TODO(bowen): When using `FusedMoEModularKernel`, this
-            # can be done in a more unified way, since
-            # `FusedMoEPrepareAndFinalize` will return the expert
-            # token count, in some cases directly from the kernel.
-            # However, now there are many code paths not using
-            # the modular kernel, e.g. calling `fused_experts`,
-            # so we decide to keep the logic here.
-            #
-            # If later refactor moved all the MoE kernel calls
-            # to the modular kernel, we can move this logic there
-            # to achieve better efficiency.
-
-            # `expert_load_view`: (num_physical_experts,)
-
-            topk_ids_flatten = topk_ids.flatten()
-
-            # Performance optimization:
-            # `masked_fill` is significantly faster than `masked_select`
-            invalid_mask = topk_ids_flatten < 0
-            # Replace invalid expert ids with 0 (just a dummy position)
-            # to avoid out-of-bounds errors in scatter_add_
-            index = topk_ids_flatten.masked_fill_(invalid_mask, 0)
-            # `src` is the valid mask, which is 1 for valid and 0 for invalid
-            src = ~invalid_mask
-
-            expert_load_view.scatter_add_(dim=0,
-                                          index=index.long(),
-                                          src=src.to(expert_load_view))
-
-            topk_ids = topk_ids.to(dtype=indices_type)
+            topk_ids = eplb_map_to_physical_and_record(
+                topk_ids=topk_ids,
+                expert_load_view=expert_load_view,
+                logical_to_physical_map=logical_to_physical_map,
+                logical_replica_count=logical_replica_count,
+                indices_type=indices_type,
+            )
 
         assert topk_ids.dtype == indices_type or indices_type is None
 

```
