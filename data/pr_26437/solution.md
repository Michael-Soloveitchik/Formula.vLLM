# PR #26437: [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h)

- field: decoding_tpot  | area: model_executor_layers
- author: vadiklyutiy
- state: MERGED  created: 2025-10-08T18:08:37Z  merged: 2025-10-16T04:18:32Z
- url: https://github.com/vllm-project/vllm/pull/26437
- labels: ready, v1, qwen

# Issue

## Problem

Qwen3-next MTP (multi-token prediction) speculative decoding used boolean mask indexing (`tensor[bool_mask]`) in the Gated DeltaNet (GDN) attention path to split tokens into spec and non-spec groups. Boolean mask indexing requires the GPU to count the number of `True` elements to size the output tensor, which forces a device-to-host (d2h) memory transfer and a synchronization stall. These transfers made MTP slower than standard token prediction on some inputs, especially during the prefill phase.

## Example

Serving Qwen/Qwen3-Next-80B-A3B-Instruct with TP=4, 512 concurrency, random 1024-in/1024-out, MTP with num_speculative_tokens=2. Before fix: 15947 tok/s output throughput, mean TTFT 5594 ms. After fix: 17872 tok/s, mean TTFT 3953 ms.

## Use case

Online serving of a large MoE hybrid-attention model with speculative decoding (MTP) enabled under high concurrency, where every GDN attention layer splits tokens between speculative and non-speculative branches.

## Resources

PCIe/host-device bandwidth and GPU sync: the d2h transfers stall the CUDA stream and serialize kernel launches, wasting GPU compute cycles waiting on the host.

## Type of users

Users running Qwen3-Next (and similar GDN/linear-attention hybrid models) with MTP speculative decoding on multi-GPU TP setups, e.g. B200 TP=4.

## Kind of inputs

Mixed batches containing both speculative-decode sequences and prefill/non-spec tokens, at high batch size / concurrency, where the spec/non-spec token split must be computed every layer.

## Mechanism with the issue

In `qwen3_next.py` `_forward`, `mixed_qkv[spec_token_masks]`, `g[:, spec_token_masks]`, `beta[:, ...]`, and the `core_attn_out[:, mask] = ...` scatter all used boolean masks built in `gdn_attn.py`'s `GDNAttentionMetadata`. Each boolean index op triggers a d2h copy of the `True` count to allocate the result tensor. Additionally `@tensor_cache` in `fla/ops/utils.py` had cache_size=4, too small for the 2 calls x 3 grouping per GDN attn, causing cache thrashing.

# Solution

## Fix in words

The metadata now carries precomputed integer index tensors `spec_token_indx` and `non_spec_token_indx` (int32) instead of a single boolean `spec_token_masks`; they are built once in `GDNAttentionMetadata.build` via `torch.argsort(spec_token_masks)` splitting at `num_non_spec_tokens`. The forward path replaces boolean indexing with `index_select(dim, indx)` for `mixed_qkv`, `g`, and `beta`, and replaces the masked scatter assignment with `index_copy_`, eliminating the d2h transfers and stream syncs. `cache_size` in `@tensor_cache` is raised from 4 to 8 so the 2x3 grouped GDN calls all stay cached. Finally `rearrange_mixed_qkv` returns `query/key/value` as `.contiguous()` to enable better torch.compile fusion.

## Author's solution

Remove boolean indexing from GDN and replace with direct index_select/index_copy indexing to avoid the d2h transfer caused by `tensor[bool_mask]` needing the True-element count. Increase `@tensor_cache` cache size since there are 2 calls per GDN attn grouped by 3, requiring at least 2*3 entries. Make q, k, v `.contiguous()` early in GDN; not helpful now but aids torch.compile fusion.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fla/ops/utils.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/v1/attention/backends/gdn_attn.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fla/ops/utils.py b/vllm/model_executor/layers/fla/ops/utils.py
index 1ed82c6086bb..7e86145da4fb 100644
--- a/vllm/model_executor/layers/fla/ops/utils.py
+++ b/vllm/model_executor/layers/fla/ops/utils.py
@@ -45,7 +45,7 @@ def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]
     """
 
     cache_entries: tuple[tuple | None, dict | None, Any] = []
-    cache_size = 4
+    cache_size = 8
 
     @functools.wraps(fn)
     def wrapper(*args: Any, **kwargs: Any) -> Any:
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index a29def57c4a0..af7f3ac9f8d4 100644
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -423,7 +423,7 @@ def rearrange_mixed_qkv(self, mixed_qkv):
             (query, key),
         )
         value = rearrange(value, "l (h d) -> 1 l h d", d=self.head_v_dim)
-        return query, key, value
+        return query.contiguous(), key.contiguous(), value.contiguous()
 
     def forward(
         self,
@@ -455,7 +455,8 @@ def _forward(
         spec_query_start_loc = attn_metadata.spec_query_start_loc
         non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
         spec_sequence_masks = attn_metadata.spec_sequence_masks
-        spec_token_masks = attn_metadata.spec_token_masks
+        spec_token_indx = attn_metadata.spec_token_indx
+        non_spec_token_indx = attn_metadata.non_spec_token_indx
         spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
         non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
         self_kv_cache = self.kv_cache[forward_context.virtual_engine]
@@ -463,8 +464,6 @@ def _forward(
         ssm_state = self_kv_cache[1]
         num_actual_tokens = attn_metadata.num_actual_tokens
         num_accepted_tokens = attn_metadata.num_accepted_tokens
-        if spec_token_masks is not None:
-            spec_token_masks = spec_token_masks[:num_actual_tokens]
 
         # 1. Set up dimensions for reshapes later
         projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states[:num_actual_tokens])
@@ -487,8 +486,8 @@ def _forward(
                 mixed_qkv_spec = mixed_qkv
                 mixed_qkv_non_spec = None
             else:
-                mixed_qkv_spec = mixed_qkv[spec_token_masks]
-                mixed_qkv_non_spec = mixed_qkv[~spec_token_masks]
+                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
+                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
         else:
             mixed_qkv_spec = None
             mixed_qkv_non_spec = mixed_qkv
@@ -558,10 +557,10 @@ def _forward(
                 g_non_spec = None
                 beta_non_spec = None
             else:
-                g_spec = g[:, spec_token_masks]
-                beta_spec = beta[:, spec_token_masks]
-                g_non_spec = g[:, ~spec_token_masks]
-                beta_non_spec = beta[:, ~spec_token_masks]
+                g_spec = g.index_select(1, spec_token_indx)
+                beta_spec = beta.index_select(1, spec_token_indx)
+                g_non_spec = g.index_select(1, non_spec_token_indx)
+                beta_non_spec = beta.index_select(1, non_spec_token_indx)
         else:
             g_spec = None
             beta_spec = None
@@ -638,8 +637,9 @@ def _forward(
                 dtype=core_attn_out_non_spec.dtype,
                 device=core_attn_out_non_spec.device,
             )
-            core_attn_out[:, spec_token_masks] = core_attn_out_spec
-            core_attn_out[:, ~spec_token_masks] = core_attn_out_non_spec
+            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
+            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
+
         elif spec_sequence_masks is not None:
             core_attn_out = core_attn_out_spec
         else:
diff --git a/vllm/v1/attention/backends/gdn_attn.py b/vllm/v1/attention/backends/gdn_attn.py
index 1deda1ccd78a..acfefde129f6 100644
--- a/vllm/v1/attention/backends/gdn_attn.py
+++ b/vllm/v1/attention/backends/gdn_attn.py
@@ -47,9 +47,9 @@ class GDNAttentionMetadata:
         None  # shape: [batch - num_spec_decodes,]
     )
     spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
-    spec_token_masks: torch.Tensor | None = (
-        None  # shape: [num_prefill_tokens + num_decode_tokens,]
-    )
+    spec_token_indx: torch.Tensor | None = None
+    non_spec_token_indx: torch.Tensor | None = None
+
     num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
 
     # The following attributes are for triton implementation of causal_conv1d
@@ -105,9 +105,14 @@ def __init__(
             dtype=torch.bool,
             device=device,
         )
-        self.spec_token_masks = torch.empty(
+        self.spec_token_indx = torch.empty(
             (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
-            dtype=torch.bool,
+            dtype=torch.int32,
+            device=device,
+        )
+        self.non_spec_token_indx = torch.empty(
+            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),
+            dtype=torch.int32,
             device=device,
         )
         self.spec_query_start_loc = torch.empty(
@@ -166,7 +171,8 @@ def build(  # type: ignore[override]
                 split_decodes_and_prefills(m, decode_threshold=1)
             )
             num_spec_decode_tokens = 0
-            spec_token_masks = None
+            spec_token_indx = None
+            non_spec_token_indx = None
             spec_state_indices_tensor = None
             non_spec_state_indices_tensor = m.block_table_tensor[:, 0]
             spec_query_start_loc = None
@@ -180,18 +186,23 @@ def build(  # type: ignore[override]
             num_prefills = non_spec_query_lens.size(0) - num_decodes
             num_decode_tokens = num_decodes
             num_prefill_tokens = non_spec_query_lens.sum().item() - num_decode_tokens
+            num_spec_decode_tokens = (
+                query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
+            )
 
             if num_prefills == 0 and num_decodes == 0:
-                spec_token_masks = torch.ones(
-                    (
-                        min(
-                            num_spec_decodes * (self.num_spec + 1),
-                            query_start_loc[-1].item(),
-                        )
-                    ),
-                    dtype=torch.bool,
+                spec_token_size = min(
+                    num_spec_decodes * (self.num_spec + 1),
+                    query_start_loc[-1].item(),
+                )
+                spec_token_indx = torch.arange(
+                    spec_token_size,
+                    dtype=torch.int32,
                     device=query_start_loc.device,
                 )
+                non_spec_token_indx = torch.empty(
+                    0, dtype=torch.int32, device=query_start_loc.device
+                )
                 spec_state_indices_tensor = m.block_table_tensor[:, : self.num_spec + 1]
                 non_spec_state_indices_tensor = None
                 spec_query_start_loc = query_start_loc
@@ -200,6 +211,11 @@ def build(  # type: ignore[override]
                 spec_token_masks = torch.repeat_interleave(
                     spec_sequence_masks, query_lens
                 )
+                index = torch.argsort(spec_token_masks)
+                num_non_spec_tokens = num_prefill_tokens + num_decode_tokens
+                non_spec_token_indx = index[:num_non_spec_tokens]
+                spec_token_indx = index[num_non_spec_tokens:]
+
                 spec_state_indices_tensor = m.block_table_tensor[
                     spec_sequence_masks, : self.num_spec + 1
                 ]
@@ -226,9 +242,6 @@ def build(  # type: ignore[override]
                     out=non_spec_query_start_loc[1:],
                 )
 
-            num_spec_decode_tokens = (
-                query_lens.sum().item() - num_prefill_tokens - num_decode_tokens
-            )
             assert num_accepted_tokens is not None
             num_accepted_tokens = num_accepted_tokens[spec_sequence_masks]
 
@@ -274,12 +287,18 @@ def build(  # type: ignore[override]
             spec_sequence_masks = self.spec_sequence_masks[:batch_size]
             spec_sequence_masks[num_spec_decodes:].fill_(False)
 
-            assert spec_token_masks is not None
-            self.spec_token_masks[: spec_token_masks.size(0)].copy_(
-                spec_token_masks, non_blocking=True
+            assert non_spec_token_indx is not None and spec_token_indx is not None
+            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(
+                non_spec_token_indx, non_blocking=True
+            )
+            non_spec_token_indx = self.non_spec_token_indx[
+                : non_spec_token_indx.size(0)
+            ]
+
+            self.spec_token_indx[: spec_token_indx.size(0)].copy_(
+                spec_token_indx, non_blocking=True
             )
-            spec_token_masks = self.spec_token_masks[:num_actual_tokens]
-            spec_token_masks[spec_token_masks.size(0) :].fill_(False)
+            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]
 
             self.spec_query_start_loc[: num_spec_decodes + 1].copy_(
                 spec_query_start_loc, non_blocking=True
@@ -332,7 +351,8 @@ def build(  # type: ignore[override]
             spec_state_indices_tensor=spec_state_indices_tensor,
             non_spec_state_indices_tensor=non_spec_state_indices_tensor,
             spec_sequence_masks=spec_sequence_masks,
-            spec_token_masks=spec_token_masks,
+            spec_token_indx=spec_token_indx,
+            non_spec_token_indx=non_spec_token_indx,
             num_accepted_tokens=num_accepted_tokens,
             nums_dict=nums_dict,
             batch_ptr=batch_ptr,

```
