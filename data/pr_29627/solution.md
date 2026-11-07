# PR #29627: [Attention] Cache attention metadata builds across hybrid KV-cache groups

- field: attention  | area: v1_attention
- author: LucasWilkinson
- state: MERGED  created: 2025-11-27T21:56:41Z  merged: 2025-12-16T22:10:16Z
- url: https://github.com/vllm-project/vllm/pull/29627
- labels: ready, v1

# Issue

## Problem

In hybrid KV-cache setups, the model runner rebuilds attention metadata independently for every KV-cache group on every forward step. When multiple hybrid groups share the same metadata builder type and the same KVCacheSpec, the only thing that actually differs between their metadata is the block table; everything else (query_start_loc, seq_lens, slot mapping layout, cudagraph buffers) is recomputed redundantly. This redundant per-group build is pure CPU/host overhead that scales with the number of KV-cache groups and is paid every decode step.

## Example

Running ibm-granite/granite-4.0-tiny-preview (hybrid attention + Mamba layers) with a decode-heavy workload (input-len 128, output-len 2048, batch-size 8). With APC enabled, caching the metadata build cut avg latency from 17.66s to 15.24s (13.7%); with APC disabled, from 14.84s to 14.49s (2.4%).

## Use case

Serving hybrid models (e.g. Granite 4.0, Mamba/attention mixes) where multiple KV-cache groups exist per model and decode steps dominate runtime.

## Resources

CPU host time and per-step kernel-launch/metadata-build overhead on the critical decode path, not GPU compute. The work bottlenecks the host while the GPU waits.

## Type of users

Users serving hybrid KV-cache models (attention + Mamba/SSM or chunked local attention), especially with prefix caching (APC) enabled and decode-heavy generation.

## Kind of inputs

Small-batch, long-output decode workloads (many short decode steps) on models that instantiate multiple hybrid KV-cache groups sharing identical builder/spec but distinct block tables.

## Mechanism with the issue

In gpu_model_runner._build_attn_group_metadata, each (kv_cache_gid, attn_gid) calls builder.build() fresh. For groups whose (KVCacheSpec, builder type) match a prior group, build() regenerates the entire CommonAttentionMetadata-derived structure when only block_table/slot_mapping differ, wasting host cycles per step.

# Solution

## Fix in words

Adds a supports_update_block_table flag and an update_block_table() method to AttentionMetadataBuilder (default raises/False), implemented for FlashAttention and Mamba2 via a shallow copy.copy of existing metadata that swaps in the new block_table/slot_mapping (Mamba2 also handles prefix-caching state indices and the CUDA-graph persistent buffer). In gpu_model_runner, a per-step cached_attn_metadata dict keyed by (KVCacheSpec, builder type) stores the first full build; subsequent groups with the same key and a supporting builder skip build() and instead call update_block_table() to patch just the block table. make_local_attention_virtual_batches now returns a (metadata, make_block_table) tuple so chunked local attention can rebuild its virtual-batch block table on update, and the chunked_local_attention backend wires this through update_block_table. The VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE env default is flipped from 0 to 1. Effect: metadata is built once per unique spec/builder and cheaply block-table-patched for the rest, cutting decode latency up to 13.7%.

## Author's solution

Scaled-back generalization of prior PRs (#22788, #29444) that caches attention metadata builds across hybrid KV-cache groups, recognizing the only per-group difference is the block table. Validated on a decode-heavy granite-4.0-tiny workload showing 13.7% latency reduction with APC enabled and 2.4% with APC disabled. Depends on #29628 landing first.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/v1/attention/test_chunked_local_attention.py`
- `vllm/attention/layers/chunked_local_attention.py`
- `vllm/envs.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/mamba2_attn.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/tests/v1/attention/test_chunked_local_attention.py b/tests/v1/attention/test_chunked_local_attention.py
index faace3473a28..4529c2cfc29b 100644
--- a/tests/v1/attention/test_chunked_local_attention.py
+++ b/tests/v1/attention/test_chunked_local_attention.py
@@ -172,7 +172,7 @@ def test_local_attention_virtual_batches(test_data: LocalAttentionTestData):
     )
 
     # Call the function
-    result = make_local_attention_virtual_batches(
+    result, _ = make_local_attention_virtual_batches(
         attn_chunk_size, common_attn_metadata, block_size
     )
 
diff --git a/vllm/attention/layers/chunked_local_attention.py b/vllm/attention/layers/chunked_local_attention.py
index 0ced0028ded9..7e3794d40833 100644
--- a/vllm/attention/layers/chunked_local_attention.py
+++ b/vllm/attention/layers/chunked_local_attention.py
@@ -4,7 +4,7 @@
 
 import torch
 
-from vllm.attention.backends.abstract import AttentionBackend, AttentionMetadata
+from vllm.attention.backends.abstract import AttentionBackend
 from vllm.attention.layer import Attention
 from vllm.attention.selector import get_attn_backend
 from vllm.config import CacheConfig
@@ -51,11 +51,19 @@ def build(
             common_prefix_len: int,
             common_attn_metadata: CommonAttentionMetadata,
             fast_build: bool = False,
-        ) -> AttentionMetadata:
-            common_attn_metadata = make_local_attention_virtual_batches(
+        ):
+            cm, make_virtual_batches_block_table = make_local_attention_virtual_batches(
                 attention_chunk_size, common_attn_metadata, block_size
             )
-            return super().build(common_prefix_len, common_attn_metadata, fast_build)
+            metadata = super().build(common_prefix_len, cm, fast_build)
+            metadata.make_virtual_batches_block_table = make_virtual_batches_block_table
+            return metadata
+
+        def update_block_table(
+            self, metadata, blk_table: torch.Tensor, slot_mapping: torch.Tensor
+        ):
+            blk_table = metadata.make_virtual_batches_block_table(blk_table)
+            return super().update_block_table(metadata, blk_table, slot_mapping)
 
     attn_backend = subclass_attention_backend(
         name_prefix=prefix,
diff --git a/vllm/envs.py b/vllm/envs.py
index cb75ba1a62de..b0a78db10496 100755
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -207,7 +207,7 @@
     VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL: bool = False
     VLLM_ENABLE_CUDAGRAPH_GC: bool = False
     VLLM_LOOPBACK_IP: str = ""
-    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
+    VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = True
     VLLM_ENABLE_RESPONSES_API_STORE: bool = False
     VLLM_USE_TRTLLM_ATTENTION: str | None = None
     VLLM_NVFP4_GEMM_BACKEND: str | None = None
@@ -1429,7 +1429,7 @@ def get_vllm_port() -> int | None:
     # kv-cache memory usage and enable longer contexts)
     # TODO(lucas): Remove this flag once latency regression is resolved.
     "VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE": lambda: bool(
-        int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "0"))
+        int(os.getenv("VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE", "1"))
     ),
     # Enables support for the "store" option in the OpenAI Responses API.
     # When set to 1, vLLM's OpenAI server will retain the input and output
diff --git a/vllm/v1/attention/backends/flash_attn.py b/vllm/v1/attention/backends/flash_attn.py
index f5ad98cf2125..3445e998d637 100755
--- a/vllm/v1/attention/backends/flash_attn.py
+++ b/vllm/v1/attention/backends/flash_attn.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Attention layer with FlashAttention."""
 
+import copy
 from dataclasses import dataclass
 from typing import ClassVar
 
@@ -250,6 +251,7 @@ class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetad
         if get_flash_attn_version() == 3
         else AttentionCGSupport.UNIFORM_BATCH
     )
+    supports_update_block_table: bool = True
 
     def __init__(
         self,
@@ -493,6 +495,17 @@ def schedule(
         )
         return attn_metadata
 
+    def update_block_table(
+        self,
+        metadata: FlashAttentionMetadata,
+        blk_table: torch.Tensor,
+        slot_mapping: torch.Tensor,
+    ) -> FlashAttentionMetadata:
+        new_metadata = copy.copy(metadata)
+        new_metadata.block_table = blk_table
+        new_metadata.slot_mapping = slot_mapping
+        return new_metadata
+
     def use_cascade_attention(self, *args, **kwargs) -> bool:
         return use_cascade_attention(*args, **kwargs)
 
diff --git a/vllm/v1/attention/backends/mamba2_attn.py b/vllm/v1/attention/backends/mamba2_attn.py
index bf1d8f09ab0a..f923371283aa 100644
--- a/vllm/v1/attention/backends/mamba2_attn.py
+++ b/vllm/v1/attention/backends/mamba2_attn.py
@@ -1,5 +1,6 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import copy
 import itertools
 from dataclasses import dataclass
 
@@ -134,6 +135,8 @@ class Mamba2AttentionMetadata:
 class Mamba2AttentionMetadataBuilder(
     BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
 ):
+    supports_update_block_table: bool = True
+
     def __init__(
         self,
         kv_cache_spec: AttentionSpec,
@@ -346,3 +349,27 @@ def build(
             num_computed_tokens_p=num_computed_tokens_p,
         )
         return attn_metadata
+
+    def update_block_table(
+        self,
+        metadata: Mamba2AttentionMetadata,
+        blk_table: torch.Tensor,
+        slot_mapping: torch.Tensor,
+    ) -> Mamba2AttentionMetadata:
+        new_metadata = copy.copy(metadata)
+        prefix_caching = self.vllm_config.cache_config.enable_prefix_caching
+        state_indices_t = blk_table if prefix_caching else blk_table[:, 0]
+        num_reqs = blk_table.shape[0]
+
+        # For CUDA graphs, copy to persistent buffer
+        if (
+            metadata.num_prefills == 0
+            and num_reqs <= self.decode_cudagraph_max_bs
+            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
+        ):
+            persistent_state_indices_t = self.state_indices_tensor[:num_reqs]
+            persistent_state_indices_t.copy_(state_indices_t, non_blocking=True)
+            state_indices_t = persistent_state_indices_t
+
+        new_metadata.state_indices_tensor = state_indices_t
+        return new_metadata
diff --git a/vllm/v1/attention/backends/utils.py b/vllm/v1/attention/backends/utils.py
index 79a1f7d4757d..d95f1637a001 100644
--- a/vllm/v1/attention/backends/utils.py
+++ b/vllm/v1/attention/backends/utils.py
@@ -4,6 +4,7 @@
 import enum
 import functools
 from abc import abstractmethod
+from collections.abc import Callable
 from dataclasses import dataclass, field, fields, make_dataclass
 from typing import (
     TYPE_CHECKING,
@@ -313,6 +314,9 @@ class AttentionMetadataBuilder(abc.ABC, Generic[M]):
     # If not, set this to None. Otherwise set it to the query
     # length that will be pulled into the front of the batch.
     reorder_batch_threshold: int | None = None
+    # Does this backend/builder support updating the block table in existing
+    # metadata
+    supports_update_block_table: bool = False
 
     @abstractmethod
     def __init__(
@@ -383,6 +387,21 @@ def build(
         """
         raise NotImplementedError
 
+    def update_block_table(
+        self,
+        metadata: M,
+        blk_table: torch.Tensor,
+        slot_mapping: torch.Tensor,
+    ) -> M:
+        """
+        Update the block table for the attention metadata.
+        Faster when theres multiple kv-cache groups that create virtually the
+        same metadata but just with different block tables.
+
+        Only needs to be implemented if supports_update_block_table is True.
+        """
+        raise NotImplementedError
+
     def build_for_cudagraph_capture(
         self, common_attn_metadata: CommonAttentionMetadata
     ) -> M:
@@ -599,7 +618,7 @@ def make_local_attention_virtual_batches(
     attn_chunk_size: int,
     common_attn_metadata: CommonAttentionMetadata,
     block_size: int = 0,
-) -> CommonAttentionMetadata:
+) -> tuple[CommonAttentionMetadata, Callable[[torch.Tensor], torch.Tensor]]:
     query_start_loc_np = common_attn_metadata.query_start_loc_cpu.numpy()
     seq_lens_np = common_attn_metadata.seq_lens_cpu.numpy()
     block_table = common_attn_metadata.block_table_tensor
@@ -711,9 +730,12 @@ def make_local_attention_virtual_batches(
     # tensor first, which recovers perf.
     batch_indices_torch = torch.from_numpy(batch_indices)
     block_indices_torch = torch.from_numpy(block_indices)
-    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
-        virtual_batches, -1
-    )
+
+    # Save as a lambda so we can return this for update_block_table
+    make_block_table = lambda block_table: block_table[
+        batch_indices_torch, block_indices_torch
+    ].view(virtual_batches, -1)
+    block_table_local = make_block_table(block_table)
 
     query_start_loc_cpu = torch.from_numpy(cu_seqlens_q_local)
     seq_lens_cpu = torch.from_numpy(seqlens_k_local)
@@ -732,7 +754,7 @@ def make_local_attention_virtual_batches(
         causal=True,
         _seq_lens_cpu=seq_lens_cpu,
         _num_computed_tokens_cpu=torch.from_numpy(num_computed_tokens_local),
-    )
+    ), make_block_table
 
 
 def make_kv_sharing_fast_prefill_common_attn_metadata(
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 3f20296c27ba..d03cfc8046bd 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -1628,6 +1628,15 @@ def _get_block_table_and_slot_mapping(kv_cache_gid: int):
                 logits_indices
             )
 
+        # Cache attention metadata builds across hybrid KV-cache groups
+        # The only thing that changes between different hybrid KV-cache groups when the
+        # same metadata builder and KVCacheSpec is the same is the block table, so we
+        # can cache the attention metadata builds and just update the block table using
+        # `builder.update_block_table` if the builder supports it.
+        cached_attn_metadata: dict[
+            tuple[KVCacheSpec, type[AttentionMetadataBuilder]], AttentionMetadata
+        ] = {}
+
         def _build_attn_group_metadata(
             kv_cache_gid: int,
             attn_gid: int,
@@ -1635,13 +1644,15 @@ def _build_attn_group_metadata(
             ubid: int | None = None,
         ) -> None:
             attn_group = self.attn_groups[kv_cache_gid][attn_gid]
+            builder = attn_group.get_metadata_builder(ubid or 0)
+            cache_key = (kv_cache_groups[kv_cache_gid].kv_cache_spec, type(builder))
+
             cascade_attn_prefix_len = (
                 cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                 if cascade_attn_prefix_lens
                 else 0
             )
 
-            builder = attn_group.get_metadata_builder(ubid or 0)
             extra_attn_metadata_args = {}
             if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                 assert ubid is None, "UBatching not supported with GDN yet"
@@ -1656,12 +1667,23 @@ def _build_attn_group_metadata(
                 attn_metadata_i = builder.build_for_cudagraph_capture(
                     common_attn_metadata
                 )
+            elif (
+                cache_key in cached_attn_metadata
+                and builder.supports_update_block_table
+            ):
+                attn_metadata_i = builder.update_block_table(
+                    cached_attn_metadata[cache_key],
+                    common_attn_metadata.block_table_tensor,
+                    common_attn_metadata.slot_mapping,
+                )
             else:
                 attn_metadata_i = builder.build(
                     common_prefix_len=cascade_attn_prefix_len,
                     common_attn_metadata=common_attn_metadata,
                     **extra_attn_metadata_args,
                 )
+                if builder.supports_update_block_table:
+                    cached_attn_metadata[cache_key] = attn_metadata_i
 
             if ubid is None:
                 assert isinstance(attn_metadata, dict)

```
