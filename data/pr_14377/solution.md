# PR #14377: [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync

- field: gpu_kernels  | area: model_executor_models
- author: cynthieye
- state: MERGED  created: 2025-03-06T17:51:47Z  merged: 2025-03-11T07:39:56Z
- url: https://github.com/vllm-project/vllm/pull/14377
- labels: ready

# Issue

## Problem

In Qwen2VisionTransformer (both qwen2_vl and qwen2_5_vl), each transformer block recomputed the same sequence-length metadata from cu_seqlens inside every block's attention forward. The xFormers branch called `(cu_seqlens[1:] - cu_seqlens[:-1]).tolist()` and the Flash Attention branch called `(cu_seqlens[1:] - cu_seqlens[:-1]).max().item()` once per block. Both `.tolist()` and `.item()` force a GPU->CPU copy, triggering a synchronous cudaMemcpyAsync + device sync on every block, adding latency.

## Example

A vision transformer with N blocks copies the same cu_seqlens-derived value from GPU to CPU N times per forward pass, where every copy stalls the stream. All copies return identical results since cu_seqlens is constant across blocks.

## Use case

Online multimodal serving of Qwen2-VL / Qwen2.5-VL image/video inputs, where the vision encoder runs on every forward and per-block sync overhead directly caps throughput.

## Resources

PCIe/host-device transfer and GPU stream synchronization — each `.item()`/`.tolist()` is a blocking D2H copy that serializes the GPU pipeline (kernel-launch/sync overhead, not raw compute).

## Type of users

Users serving Qwen2-VL / Qwen2.5-VL vision-language models at scale, on either the Flash Attention or xFormers ViT attention backend.

## Kind of inputs

Any multimodal request with image/video tokens producing a multi-block vision transformer pass; cost scales with number of ViT blocks and is independent of the actual sequence content.

## Mechanism with the issue

Inside `Qwen2VisionBlock.forward` -> attention `forward`, the per-block recomputation of `max_seqlen` (Flash Attn) and `seqlens` (xFormers) from `cu_seqlens` each issued a D2H sync. Because cu_seqlens does not change between blocks, every block produced the identical value, so the repeated synchronous copies were pure redundant overhead.

# Solution

## Fix in words

The diff hoists the cu_seqlens-derived values out of the per-block loop and into `Qwen2VisionTransformer.forward`, computing them exactly once before iterating `self.blocks`. A new `self.attn_backend = get_vit_attn_backend(support_fa=True)` selects which value to compute: `max_seqlen = (cu_seqlens[1:]-cu_seqlens[:-1]).max().item()` only for FLASH_ATTN, or `seqlens = (...).tolist()` only for XFORMERS. These are threaded as new optional `max_seqlen`/`seqlens` params through `Qwen2VisionBlock.forward` into the attention `forward`, where the inline recomputations were deleted. Result: one D2H sync per forward instead of one per block, removing the redundant cudaMemcpyAsync stalls.

## Author's solution

The author observed that the xFormers branch repeatedly calls `.tolist()` (and the Flash Attn branch `.item()`) to force GPU->CPU copies on every forward, triggering cudaMemcpyAsync. Since the inputs and outputs are identical across the repeated calls, they compute it once and reuse the first result for the rest. They report the online xFormers branch QPS improved ~15% and the Flash Attn branch ~7%.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/qwen2_5_vl.py`
- `vllm/model_executor/models/qwen2_vl.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/qwen2_5_vl.py b/vllm/model_executor/models/qwen2_5_vl.py
index ef3d28c8087d..ae48c779481f 100644
--- a/vllm/model_executor/models/qwen2_5_vl.py
+++ b/vllm/model_executor/models/qwen2_5_vl.py
@@ -255,10 +255,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
         return q, k, v
 
     def forward(
-        self,
-        x: torch.Tensor,
-        cu_seqlens: torch.Tensor,
-        rotary_pos_emb: torch.Tensor,
+            self,
+            x: torch.Tensor,
+            cu_seqlens: torch.Tensor,
+            rotary_pos_emb: torch.Tensor,
+            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
+            seqlens: Optional[list[int]] = None,  # Only used for xFormers
     ) -> torch.Tensor:
         # [s, b, c] --> [s, b, head * 3 * head_dim]
         x, _ = self.qkv(x)
@@ -285,7 +287,6 @@ def forward(
 
             q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
 
-            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
             output = flash_attn_varlen_func(q,
                                             k,
                                             v,
@@ -321,7 +322,6 @@ def forward(
             from xformers import ops as xops
             from xformers.ops.fmha.attn_bias import BlockDiagonalMask
 
-            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
             attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                        kv_seqlen=None,
                                                        device=q.device)
@@ -364,11 +364,20 @@ def __init__(
                                      quant_config=quant_config,
                                      prefix=f"{prefix}.mlp")
 
-    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
-                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
+    def forward(
+            self,
+            x: torch.Tensor,
+            cu_seqlens: torch.Tensor,
+            rotary_pos_emb: torch.Tensor,
+            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
+            seqlens: Optional[list[int]] = None,  # Only used for xFormers
+    ) -> torch.Tensor:
         x = x + self.attn(self.norm1(x),
                           cu_seqlens=cu_seqlens,
-                          rotary_pos_emb=rotary_pos_emb)
+                          rotary_pos_emb=rotary_pos_emb,
+                          max_seqlen=max_seqlen,
+                          seqlens=seqlens)
+
         x = x + self.mlp(self.norm2(x))
         return x
 
@@ -528,6 +537,7 @@ def __init__(
             quant_config=quant_config,
             prefix=f"{prefix}.merger",
         )
+        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
 
     @property
     def dtype(self) -> torch.dtype:
@@ -633,14 +643,25 @@ def forward(
 
         # transformers
         hidden_states = hidden_states.unsqueeze(1)
+
+        max_seqlen = None
+        seqlens = None
+        if self.attn_backend == _Backend.FLASH_ATTN:
+            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
+        elif self.attn_backend == _Backend.XFORMERS:
+            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
         for layer_num, blk in enumerate(self.blocks):
             if layer_num in self.fullatt_block_indexes:
                 cu_seqlens_now = cu_seqlens
             else:
                 cu_seqlens_now = cu_window_seqlens
-            hidden_states = blk(hidden_states,
-                                cu_seqlens=cu_seqlens_now,
-                                rotary_pos_emb=rotary_pos_emb)
+            hidden_states = blk(
+                hidden_states,
+                cu_seqlens=cu_seqlens_now,
+                rotary_pos_emb=rotary_pos_emb,
+                max_seqlen=max_seqlen,
+                seqlens=seqlens,
+            )
 
         # For Qwen2.5-VL-3B, float16 will overflow at last block
         # for long visual tokens sequences.
diff --git a/vllm/model_executor/models/qwen2_vl.py b/vllm/model_executor/models/qwen2_vl.py
index ac3d154dd881..0e9fa7183c89 100644
--- a/vllm/model_executor/models/qwen2_vl.py
+++ b/vllm/model_executor/models/qwen2_vl.py
@@ -303,10 +303,12 @@ def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
         return q, k, v
 
     def forward(
-        self,
-        x: torch.Tensor,
-        cu_seqlens: torch.Tensor,
-        rotary_pos_emb: torch.Tensor,
+            self,
+            x: torch.Tensor,
+            cu_seqlens: torch.Tensor,
+            rotary_pos_emb: torch.Tensor,
+            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
+            seqlens: Optional[list[int]] = None,  # Only used for xFormers
     ) -> torch.Tensor:
 
         # [s, b, c] --> [s, b, 3 * head * head_dim]
@@ -329,7 +331,6 @@ def forward(
 
             q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
 
-            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
             output = flash_attn_varlen_func(q,
                                             k,
                                             v,
@@ -365,7 +366,6 @@ def forward(
             from xformers import ops as xops
             from xformers.ops.fmha.attn_bias import BlockDiagonalMask
 
-            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
             attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seqlens,
                                                        kv_seqlen=None,
                                                        device=q.device)
@@ -409,11 +409,22 @@ def __init__(
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")
 
-    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
-                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
-        x = x + self.attn(self.norm1(x),
-                          cu_seqlens=cu_seqlens,
-                          rotary_pos_emb=rotary_pos_emb)
+    def forward(
+            self,
+            x: torch.Tensor,
+            cu_seqlens: torch.Tensor,
+            rotary_pos_emb: torch.Tensor,
+            max_seqlen: Optional[int] = None,  # Only used for Flash Attention
+            seqlens: Optional[list[int]] = None,  # Only used for xFormers
+    ) -> torch.Tensor:
+        x = x + self.attn(
+            self.norm1(x),
+            cu_seqlens=cu_seqlens,
+            rotary_pos_emb=rotary_pos_emb,
+            max_seqlen=max_seqlen,
+            seqlens=seqlens,
+        )
+
         x = x + self.mlp(self.norm2(x))
         return x
 
@@ -570,6 +581,7 @@ def __init__(
             quant_config=quant_config,
             prefix=f"{prefix}.merger",
         )
+        self.attn_backend: _Backend = get_vit_attn_backend(support_fa=True)
 
     @property
     def dtype(self) -> torch.dtype:
@@ -624,8 +636,21 @@ def forward(
 
         # transformers
         x = x.unsqueeze(1)
+
+        max_seqlen = None
+        seqlens = None
+        if self.attn_backend == _Backend.FLASH_ATTN:
+            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
+        elif self.attn_backend == _Backend.XFORMERS:
+            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
         for blk in self.blocks:
-            x = blk(x, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
+            x = blk(
+                x,
+                cu_seqlens=cu_seqlens,
+                rotary_pos_emb=rotary_pos_emb,
+                max_seqlen=max_seqlen,
+                seqlens=seqlens,
+            )
 
         # adapter
         x = self.merger(x)

```
