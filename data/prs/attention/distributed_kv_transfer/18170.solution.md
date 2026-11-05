# PR #18170: [V1][P/D] Local attention optimization for NIXL

- field: attention  | area: distributed_kv_transfer
- author: mgoin
- state: MERGED  created: 2025-05-14T21:04:22Z  merged: 2025-05-17T01:16:33Z
- url: https://github.com/vllm-project/vllm/pull/18170
- labels: ready, v1

# Issue

## Problem

The V1 NixlConnector transferred the full KV cache for every layer during disaggregated prefill/decode (P/D) KV transfers, ignoring that some models use local (chunked) attention where only a recent window of blocks is actually needed. For Llama 4, which mixes global (NoPE) and local chunked (RoPE) attention layers, this means transferring far more KV data over the network than the local-attention layers require. The result is inflated P/D transfer volume and degraded TTFT in multi-node setups.

## Example

For a 100k-token prefill on Llama-4-Scout, 3 of every 4 layers use local attention with attention_chunk_size=8192, so only ~8192 tokens of KV are needed per local layer instead of 100k. The connector transferred the full 100k-token cache for all layers; limiting local layers to their window drops transferred data to (100000*1/4)+(8192*3/4)=31,144 token-equivalents, a 69% reduction.

## Use case

Disaggregated prefill/decode serving where the prefill worker ships computed KV cache over NIXL to a separate decode worker, specifically for long-context Llama 4 inference.

## Resources

Network/interconnect bandwidth for inter-node KV transfer (NIXL over the fabric between P and D nodes), which dominates TTFT overhead in multi-node P/D.

## Type of users

Users running multi-node disaggregated P/D serving on Llama 4 models (local/global hybrid attention), e.g. 4xH200 Llama-4-Scout deployments.

## Kind of inputs

Long input sequences (100k tokens) where local-attention layers only need the most recent chunk_size (8192) tokens, so the gap between full-sequence blocks and window blocks is large.

## Mechanism with the issue

In `NixlConnectorWorker._read_blocks`, `_get_block_descs_ids` was called once with the full `local_block_ids`/`remote_block_ids` across all NIXL regions, generating descriptors for every block of every layer. It had no per-layer awareness and no notion of an attention window, so chunked local-attention layers still got descriptors for the entire sequence of blocks.

# Solution

## Fix in words

The worker now receives `vllm_config` and, in `register_kv_caches`, detects `model_type == "llama4"` and precomputes `block_window_per_layer`: for each layer it uses `no_rope_layers` to tag local (RoPE) layers with `chunk_block_size = ceil(attention_chunk_size / block_size)` and global (NoPE) layers with `None`. In `_read_blocks`, if `block_window_per_layer` is set, it iterates per layer and truncates block IDs to the last `block_window` blocks for local layers (full list for global), building descriptors per layer instead of once globally. `_get_block_descs_ids` gains a `layer_idx` arg that selects only that layer's NIXL regions (K_i/V_i for non-MLA via `2*layer_idx`, single region for MLA), so descriptors map only to that layer's memory and selected blocks. Non-Llama4 models fall through the default global-attention path unchanged.

## Author's solution

During `register_kv_caches`, pre-calculate the per-layer attention window in blocks from Llama 4's `no_rope_layers` and `attention_chunk_size`: RoPE layers get a local window, NoPE layers use the full cache. In `start_load_kv`, iterate layers and truncate the physical block ID list to the window (most recent blocks) for local-attention layers, and have `_get_block_descs_ids` use a `layer_idx` to emit NIXL descriptors only for that layer's memory regions and selected blocks. This cuts transferred data for Llama 4 RoPE layers, improving TTFT, and sets a pattern for future attention types like sliding window.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`

### Diff
```diff
diff --git a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
index c0c03efcdbf4..e6c83a0fc5bd 100644
--- a/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
+++ b/vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py
@@ -96,7 +96,8 @@ def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole):
             self.connector_worker: Optional[NixlConnectorWorker] = None
         elif role == KVConnectorRole.WORKER:
             self.connector_scheduler = None
-            self.connector_worker = NixlConnectorWorker(str(self.engine_id))
+            self.connector_worker = NixlConnectorWorker(
+                vllm_config, str(self.engine_id))
 
     ############################################################
     # Scheduler Side Methods
@@ -302,7 +303,7 @@ def request_finished(
 class NixlConnectorWorker:
     """Implementation of Worker side methods"""
 
-    def __init__(self, engine_id: str):
+    def __init__(self, vllm_config: VllmConfig, engine_id: str):
         if NixlWrapper is None:
             logger.error("NIXL is not available")
             raise RuntimeError("NIXL is not available")
@@ -329,6 +330,7 @@ def __init__(self, engine_id: str):
         # Number of NIXL regions. Currently one region per cache
         # (so 1 per layer for MLA, otherwise 2 per layer)
         self.num_regions = 0
+        self.num_layers = 0
 
         # nixl_prepped_dlist_handle (int).
         self.src_xfer_side_handle: int = 0
@@ -355,6 +357,14 @@ def __init__(self, engine_id: str):
         # Background thread for establishing new connections.
         self._nixl_handshake_listener_t: Optional[threading.Thread] = None
 
+        self.vllm_config = vllm_config
+        self.block_size = vllm_config.cache_config.block_size
+
+        # TODO(mgoin): remove this once we have hybrid memory allocator
+        # Optimization for models with local attention (Llama 4)
+        # List of block window sizes for each layer for local attention
+        self.block_window_per_layer: list[Optional[int]] = []
+
     @staticmethod
     def _nixl_handshake_listener(metadata: NixlAgentMetadata,
                                  ready_event: threading.Event, rank: int):
@@ -465,6 +475,27 @@ def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
                 kv_caches_base_addr.append(base_addr)
         self.kv_caches_base_addr[self.engine_id] = kv_caches_base_addr
         self.num_regions = len(caches_data)
+        self.num_layers = len(self.kv_caches.keys())
+
+        # TODO(mgoin): remove this once we have hybrid memory allocator
+        # Optimization for models with local attention (Llama 4)
+        if self.vllm_config.model_config.hf_config.model_type == "llama4":
+            from transformers import Llama4TextConfig
+            assert isinstance(self.vllm_config.model_config.hf_text_config,
+                              Llama4TextConfig)
+            llama4_config = self.vllm_config.model_config.hf_text_config
+            no_rope_layers = llama4_config.no_rope_layers
+            chunk_size = llama4_config.attention_chunk_size
+            chunk_block_size = math.ceil(chunk_size / self.block_size)
+            for layer_idx in range(self.num_layers):
+                # no_rope_layers[layer_idx] == 0 means NoPE (global)
+                # Any other value means RoPE (local chunked)
+                is_local_attention = no_rope_layers[layer_idx] != 0
+                block_window = chunk_block_size if is_local_attention else None
+                self.block_window_per_layer.append(block_window)
+            logger.debug("Llama 4 block window per layer mapping: %s",
+                         self.block_window_per_layer)
+            assert len(self.block_window_per_layer) == self.num_layers
 
         descs = self.nixl_wrapper.get_reg_descs(caches_data, "VRAM")
         logger.debug("Registering descs: %s", caches_data)
@@ -699,10 +730,39 @@ def _read_blocks(
         remote_xfer_side_handle = self.dst_xfer_side_handles[dst_engine_id]
 
         # Get descs ids.
-        remote_block_descs_ids = self._get_block_descs_ids(
-            dst_engine_id, remote_block_ids)
-        local_block_descs_ids = self._get_block_descs_ids(
-            self.engine_id, local_block_ids)
+        local_block_descs_ids: list[int] = []
+        remote_block_descs_ids: list[int] = []
+        if not self.block_window_per_layer:
+            # Default case: assume global attention
+            remote_block_descs_ids = self._get_block_descs_ids(
+                dst_engine_id, remote_block_ids)
+            local_block_descs_ids = self._get_block_descs_ids(
+                self.engine_id, local_block_ids)
+        else:
+            # TODO(mgoin): remove this once we have hybrid memory allocator
+            # Optimization for models with local attention (Llama 4)
+            for layer_idx, block_window in enumerate(
+                    self.block_window_per_layer):
+                # For each layer:
+                if block_window is None:
+                    # If not chunked, we just use the
+                    # full block lists (global attention)
+                    layer_local_block_ids = local_block_ids
+                    layer_remote_block_ids = remote_block_ids
+                else:
+                    # If chunked, get the last block_window blocks
+                    layer_local_block_ids = local_block_ids[-block_window:]
+                    layer_remote_block_ids = remote_block_ids[-block_window:]
+
+                # Get descs ids for the layer.
+                layer_local_desc_ids = self._get_block_descs_ids(
+                    self.engine_id, layer_local_block_ids, layer_idx)
+                layer_remote_desc_ids = self._get_block_descs_ids(
+                    dst_engine_id, layer_remote_block_ids, layer_idx)
+
+                local_block_descs_ids.extend(layer_local_desc_ids)
+                remote_block_descs_ids.extend(layer_remote_desc_ids)
+
         assert len(local_block_descs_ids) == len(remote_block_descs_ids)
 
         # Prepare transfer with Nixl.
@@ -721,12 +781,31 @@ def _read_blocks(
         # Use handle to check completion in future step().
         self._recving_transfers[request_id].append(handle)
 
-    def _get_block_descs_ids(self, engine_id: str,
-                             block_ids: list[int]) -> list[int]:
-        """Get the descs ids for a set of block ids."""
+    def _get_block_descs_ids(self,
+                             engine_id: str,
+                             block_ids: list[int],
+                             layer_idx: Optional[int] = None) -> list[int]:
+        """
+        Get the descs ids for a set of block ids.
+        If layer_idx is provided, we use the region_ids for the given layer.
+        Otherwise, we use all regions.
+        """
+
+        if layer_idx is None:
+            region_ids = range(self.num_regions)
+        else:
+            assert layer_idx < self.num_layers
+            if self.num_layers < self.num_regions:
+                # If we have more regions than layers, we assume that
+                # the regions are organized as [K0, V0, K1, V1, ...]
+                # and we select K_i and V_i
+                assert 2 * self.num_layers == self.num_regions
+                region_ids = range(2 * layer_idx, 2 * layer_idx + 2)
+            else:
+                # Otherwise, we assume we have MLA and select i-th layer
+                assert self.num_layers == self.num_regions
+                region_ids = range(layer_idx, layer_idx + 1)
 
-        # range(1) for MLA, range(2) otherwise.
-        region_ids = range(self.num_regions)
         num_blocks = self.dst_num_blocks[engine_id]
 
         # Compute the desc ids for each block.

```
