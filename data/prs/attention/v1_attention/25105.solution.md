# PR #25105: [PERF] Add `conv1d` metadata to GDN attn

- field: attention  | area: v1_attention
- author: vadiklyutiy
- state: MERGED  created: 2025-09-17T22:15:45Z  merged: 2025-09-18T14:27:50Z
- url: https://github.com/vllm-project/vllm/pull/25105
- labels: ready, v1, qwen

# Issue

## Problem

The Qwen3-Next GDN (gated delta net) attention path calls `causal_conv1d_fn` without the convolution metadata that the Triton kernel uses to precompute per-batch indexing. Lacking this metadata, the kernel path falls back to logic that performs repeated GPU-to-host memory transfers on every invocation. These synchronous transfers stall the GPU and sharply lower utilization during the convolution step.

## Example

Serving Qwen/Qwen3-Next-80B-A3B-Instruct on H200 with tp=4, random-input 32768, output 1, concurrency 8: mean TTFT is ~2600 ms and total token throughput ~57.5k tok/s before the fix, bottlenecked on the conv1d host syncs.

## Use case

Prefill of long-context prompts in the GDN/linear-attention layers of Qwen3-Next during online serving.

## Resources

Stresses PCIe host<->device bandwidth and forces host-device synchronization, which idles GPU compute (low SM utilization).

## Type of users

Users serving Qwen3-Next (hybrid Mamba/GDN MoE) models, especially multi-GPU tensor-parallel long-context inference on high-end GPUs (H200).

## Kind of inputs

Long prefill sequences (e.g. 32k input tokens) with num_prefills > 0 batched across multiple concurrent requests.

## Mechanism with the issue

In `qwen3_next.py` `_forward`, the prefill branch calls `causal_conv1d_fn(mixed_qkv_non_spec.transpose(0,1), ...)` with no `metadata` argument. `GDNAttentionMetadata` had no conv1d metadata fields (`nums_dict`, `cu_seqlen`, `batch_ptr`, `token_chunk_offset_ptr`), unlike `Mamba2AttentionMetadata`, so the Triton `causal_conv1d` could not use its precomputed batch pointers and instead recomputed indexing with host transfers each call.

# Solution

## Fix in words

Adds the four conv1d metadata fields (`nums_dict`, `cu_seqlen`, `batch_ptr`, `token_chunk_offset_ptr`) to `GDNAttentionMetadata`, mirroring `Mamba2AttentionMetadata`. In `qwen3_next.py`, the prefill branch now hoists the transpose into `mixed_qkv_non_spec_T` and, when `conv_metadata.cu_seqlen is None`, calls `update_metadata(...)` to populate the metadata once, then passes `metadata=conv_metadata` into `causal_conv1d_fn`. `update_metadata` is extended to accept `GDNAttentionMetadata` in its type union. With precomputed batch pointers, the Triton kernel skips the per-call GPU->host transfers, raising GPU utilization and cutting TTFT (~2600->~2068 ms, ~26% speedup). Also fixes `torch.tensor` -> `torch.Tensor` annotations across mamba2/gdn/short_conv metadata.

## Author's solution

Add conv metadata to the GDN attention path similar to mamba2, so `causal_conv1d_fn` uses precomputed metadata and avoids the GPU<->host memory transfers that were throttling GDN attention utilization for Qwen3-Next. Also corrected the `torch.tensor` -> `torch.Tensor` type annotations. Benchmarks show a 26% speedup.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/mamba/mamba2_metadata.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/v1/attention/backends/mamba2_attn.py`
- `vllm/v1/attention/backends/short_conv_attn.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/mamba/mamba2_metadata.py b/vllm/model_executor/layers/mamba/mamba2_metadata.py
index 368bfe3af1d3..c926e17a2c19 100644
--- a/vllm/model_executor/layers/mamba/mamba2_metadata.py
+++ b/vllm/model_executor/layers/mamba/mamba2_metadata.py
@@ -11,6 +11,7 @@
     PlaceholderAttentionMetadata)
 from vllm.attention.backends.utils import PAD_SLOT_ID
 from vllm.platforms import current_platform
+from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
 from vllm.v1.attention.backends.mamba2_attn import (
     Mamba2AttentionMetadata, _query_start_loc_to_chunk_indices_offsets)
 
@@ -45,8 +46,8 @@ class Mamba2Metadata:
     """
     nums_dict: Optional[dict] = None
     cu_seqlen: Optional[int] = None
-    batch_ptr: Optional[torch.tensor] = None
-    token_chunk_offset_ptr: Optional[torch.tensor] = None
+    batch_ptr: Optional[torch.Tensor] = None
+    token_chunk_offset_ptr: Optional[torch.Tensor] = None
 
 
 def get_platform_metadata_classes() -> tuple[type[AttentionMetadata], ...]:
@@ -117,7 +118,8 @@ def prepare_mamba2_metadata(
 
 def update_metadata(x: torch.Tensor, query_start_loc: torch.Tensor,
                     mamba2_metadata: Union[Mamba2Metadata,
-                                           Mamba2AttentionMetadata]):
+                                           Mamba2AttentionMetadata,
+                                           GDNAttentionMetadata]):
     """
     this is triggered upon handling a new input at the first layer
     """
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index fe63e9303235..121ce3349eca 100644
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -36,6 +36,7 @@
                                                RowParallelLinear)
 from vllm.model_executor.layers.logits_processor import LogitsProcessor
 from vllm.model_executor.layers.mamba.abstract import MambaBase
+from vllm.model_executor.layers.mamba.mamba2_metadata import update_metadata
 from vllm.model_executor.layers.mamba.mamba_mixer2 import (
     mamba_v2_sharded_weight_loader)
 from vllm.model_executor.layers.mamba.mamba_utils import (
@@ -406,6 +407,7 @@ def _forward(
 
         assert isinstance(attn_metadata, dict)
         attn_metadata = attn_metadata[self.prefix]
+        conv_metadata = attn_metadata
         assert isinstance(attn_metadata, GDNAttentionMetadata)
         has_initial_state = attn_metadata.has_initial_state
         spec_query_start_loc = attn_metadata.spec_query_start_loc
@@ -472,10 +474,15 @@ def _forward(
 
         # 2.2: process the remaining part
         if attn_metadata.num_prefills > 0:
+            mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
+            if conv_metadata.cu_seqlen is None:
+                conv_metadata = update_metadata(mixed_qkv_non_spec_T,
+                                                non_spec_query_start_loc,
+                                                conv_metadata)
             # - "cache_indices" updates the conv_state cache in positions
             #   pointed to by "mamba_cache_params.state_indices_tensor"
             mixed_qkv_non_spec = causal_conv1d_fn(
-                mixed_qkv_non_spec.transpose(0, 1),
+                mixed_qkv_non_spec_T,
                 conv_weights,
                 self.conv1d.bias,
                 activation=self.activation,
@@ -483,6 +490,7 @@ def _forward(
                 has_initial_state=has_initial_state,
                 cache_indices=non_spec_state_indices_tensor,
                 query_start_loc=non_spec_query_start_loc,
+                metadata=conv_metadata,
             ).transpose(0, 1)
         elif attn_metadata.num_decodes > 0:
             mixed_qkv_non_spec = causal_conv1d_update(
diff --git a/vllm/v1/attention/backends/gdn_attn.py b/vllm/v1/attention/backends/gdn_attn.py
index ba89f93e8b56..5dadc52d0fb1 100644
--- a/vllm/v1/attention/backends/gdn_attn.py
+++ b/vllm/v1/attention/backends/gdn_attn.py
@@ -50,6 +50,12 @@ class GDNAttentionMetadata:
         Tensor] = None  # shape: [num_prefill_tokens + num_decode_tokens,]
     num_accepted_tokens: Optional[torch.Tensor] = None  # shape: [batch,]
 
+    # The following attributes are for triton implementation of causal_conv1d
+    nums_dict: Optional[dict] = None
+    cu_seqlen: Optional[int] = None
+    batch_ptr: Optional[torch.Tensor] = None
+    token_chunk_offset_ptr: Optional[torch.Tensor] = None
+
 
 class GDNAttentionMetadataBuilder(
         AttentionMetadataBuilder[GDNAttentionMetadata]):
diff --git a/vllm/v1/attention/backends/mamba2_attn.py b/vllm/v1/attention/backends/mamba2_attn.py
index 359bad1ea9de..2fe1f14ca1db 100644
--- a/vllm/v1/attention/backends/mamba2_attn.py
+++ b/vllm/v1/attention/backends/mamba2_attn.py
@@ -132,8 +132,8 @@ class Mamba2AttentionMetadata:
     # The following attributes are for triton implementation of causal_conv1d
     nums_dict: Optional[dict] = None
     cu_seqlen: Optional[int] = None
-    batch_ptr: Optional[torch.tensor] = None
-    token_chunk_offset_ptr: Optional[torch.tensor] = None
+    batch_ptr: Optional[torch.Tensor] = None
+    token_chunk_offset_ptr: Optional[torch.Tensor] = None
 
 
 class Mamba2AttentionMetadataBuilder(
diff --git a/vllm/v1/attention/backends/short_conv_attn.py b/vllm/v1/attention/backends/short_conv_attn.py
index f5ad65b02b4d..717c40b37ecf 100644
--- a/vllm/v1/attention/backends/short_conv_attn.py
+++ b/vllm/v1/attention/backends/short_conv_attn.py
@@ -34,8 +34,8 @@ class ShortConvAttentionMetadata:
     # For causal_conv1d
     nums_dict: Optional[dict] = None
     cu_seqlen: Optional[int] = None
-    batch_ptr: Optional[torch.tensor] = None
-    token_chunk_offset_ptr: Optional[torch.tensor] = None
+    batch_ptr: Optional[torch.Tensor] = None
+    token_chunk_offset_ptr: Optional[torch.Tensor] = None
 
 
 class ShortConvAttentionMetadataBuilder(

```
