# PR #21126: [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available

- field: attention  | area: model_executor_layers
- author: mgoin
- state: MERGED  created: 2025-07-17T16:12:23Z  merged: 2025-09-19T20:06:49Z
- url: https://github.com/vllm-project/vllm/pull/21126
- labels: ready

# Issue

## Problem

RotaryEmbedding.forward_cuda always dispatched to vLLM's custom `ops.rotary_embedding` kernel, which is ~2x slower than FlashInfer's RoPE kernel per microbenchmarks. No path existed to use FlashInfer's `apply_rope_with_cos_sin_cache_inplace` even when FlashInfer was installed and the config (dtype, head_size, platform) supported it. Separately, the cos/sin cache device/dtype matching logic was duplicated and inconsistent across the various RoPE subclasses.

## Example

Llama-3.1-8B-Instruct on CUDA with head_size=128 and bf16/fp16: each decode/prefill step runs the custom RoPE kernel instead of FlashInfer, leaving ~2x per-op kernel speedup on the table (though gsm8k end-to-end throughput was ~unchanged: 34.6 vs 33.5 it/s).

## Use case

Any CUDA inference where rotary embedding is applied to Q/K every attention layer every forward pass — high-frequency small kernel in the hot path.

## Resources

GPU compute and kernel-launch overhead; RoPE is a cheap pointwise op so kernel efficiency dominates its cost.

## Type of users

Users serving transformer models on CUDA GPUs with FlashInfer installed, using fp16/bf16 and standard head sizes (64/128/256/512).

## Kind of inputs

fp16 or bf16 query/key tensors with head_size in {64,128,256,512}, neox or non-neox style, on CUDA platform.

## Mechanism with the issue

`base.py::forward_cuda` unconditionally imported `_custom_ops` and called `ops.rotary_embedding`. The cos_sin_cache was always cast to `dtype` at init, and each subclass (deepseek, llama4_vision, mrope) hand-rolled its own `cos_sin_cache.to(...)` device/dtype fixups, some only matching device not dtype — slow `nn.Module.__setattr__` calls and inconsistent behavior.

# Solution

## Fix in words

Adds a `self.use_flashinfer` gate in `__init__` (enabled + bf16/fp16 + CUDA + `has_flashinfer()` + head_size in {64,128,256,512}); when set, the cos_sin_cache is NOT pre-cast to dtype. `forward_cuda` now checks `use_flashinfer` first and calls the new `torch.ops.vllm.flashinfer_rotary_embedding` custom op (wrapping FlashInfer's `apply_rope_with_cos_sin_cache_inplace`, registered via `direct_register_custom_op` with in-place query/key mutation and a no-op fake impl), returning early. The duplicated cache fixup logic is consolidated into a single `_match_cos_sin_cache_dtype` helper used by forward_cuda, forward_xpu, and the deepseek/llama4_vision/mrope subclasses. Net effect: optional ~2x-faster FlashInfer RoPE path plus unified, cheaper cache device/dtype matching.

## Author's solution

Enable FlashInfer RoPE for the cuda forward impl when the rotary_embedding custom op is set; microbenchmarks show ~2x speedup over the custom op. Because lm-eval (llama 8b, gsm8k) showed no noticeable end-to-end difference, the author left it not enabled by default, and documented the one-line `VllmConfig.__post_init__` change needed to turn it on.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/rotary_embedding/base.py`
- `vllm/model_executor/layers/rotary_embedding/common.py`
- `vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`
- `vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py`
- `vllm/model_executor/layers/rotary_embedding/mrope.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/rotary_embedding/base.py b/vllm/model_executor/layers/rotary_embedding/base.py
index db50eb08db3f..3dc249ae9adb 100644
--- a/vllm/model_executor/layers/rotary_embedding/base.py
+++ b/vllm/model_executor/layers/rotary_embedding/base.py
@@ -6,6 +6,8 @@
 import torch
 
 from vllm.model_executor.custom_op import CustomOp
+from vllm.platforms import current_platform
+from vllm.utils.flashinfer import has_flashinfer
 
 from .common import apply_rotary_emb_torch
 
@@ -30,9 +32,17 @@ def __init__(
         self.base = base
         self.is_neox_style = is_neox_style
         self.dtype = dtype
+        # Flashinfer only supports head_size=64, 128, 256, 512.
+        # https://github.com/flashinfer-ai/flashinfer/blob/ebfd655efe830048dba5d582aaa61d61d1cf9a87/include/flashinfer/utils.cuh#L174-L202
+        self.use_flashinfer = (self.enabled()
+                               and dtype in (torch.float16, torch.bfloat16)
+                               and current_platform.is_cuda()
+                               and has_flashinfer()
+                               and self.head_size in [64, 128, 256, 512])
 
         cache = self._compute_cos_sin_cache()
-        cache = cache.to(dtype)
+        if not self.use_flashinfer:
+            cache = cache.to(dtype)
         self.cos_sin_cache: torch.Tensor
         self.register_buffer("cos_sin_cache", cache, persistent=False)
 
@@ -57,6 +67,14 @@ def _compute_cos_sin_cache(self) -> torch.Tensor:
         cache = torch.cat((cos, sin), dim=-1)
         return cache
 
+    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
+        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
+        # is expensive, so avoid calling it if possible
+        if self.cos_sin_cache.device != query.device or \
+            self.cos_sin_cache.dtype != query.dtype:
+            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
+                                                       dtype=query.dtype)
+
     def forward_native(
         self,
         positions: torch.Tensor,
@@ -94,15 +112,16 @@ def forward_cuda(
         query: torch.Tensor,
         key: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
-        from vllm import _custom_ops as ops
+        if self.use_flashinfer:
+            torch.ops.vllm.flashinfer_rotary_embedding(positions, query, key,
+                                                       self.head_size,
+                                                       self.cos_sin_cache,
+                                                       self.is_neox_style)
+            return query, key
 
-        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
-        # is expensive, so avoid calling it if possible
-        if self.cos_sin_cache.device != query.device or \
-            self.cos_sin_cache.dtype != query.dtype:
-            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
-                                                       dtype=query.dtype)
+        from vllm import _custom_ops as ops
 
+        self._match_cos_sin_cache_dtype(query)
         # ops.rotary_embedding() is an in-place operation
         # that updates the query and key tensors.
         ops.rotary_embedding(positions, query, key, self.head_size,
@@ -117,8 +136,7 @@ def forward_xpu(
     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
         from vllm._ipex_ops import ipex_ops as ops
 
-        self.cos_sin_cache = self.cos_sin_cache.to(positions.device,
-                                                   dtype=query.dtype)
+        self._match_cos_sin_cache_dtype(query)
         # ops.rotary_embedding() is an in-place operation
         # that updates the query and key tensors.
         if key is None:
diff --git a/vllm/model_executor/layers/rotary_embedding/common.py b/vllm/model_executor/layers/rotary_embedding/common.py
index 8d821bea19e3..e3cd0a8e788e 100644
--- a/vllm/model_executor/layers/rotary_embedding/common.py
+++ b/vllm/model_executor/layers/rotary_embedding/common.py
@@ -6,6 +6,7 @@
 import torch
 
 from vllm.platforms import current_platform
+from vllm.utils import direct_register_custom_op
 
 if current_platform.is_cuda():
     from vllm.vllm_flash_attn.layers.rotary import apply_rotary_emb
@@ -103,3 +104,48 @@ def yarn_get_mscale(scale: float = 1) -> float:
     if scale <= 1:
         return 1.0
     return 0.1 * math.log(scale) + 1.0
+
+
+def _flashinfer_rotary_embedding(
+    positions: torch.Tensor,
+    query: torch.Tensor,
+    key: torch.Tensor,
+    head_size: int,
+    cos_sin_cache: torch.Tensor,
+    is_neox: bool,
+) -> None:
+    """Custom op wrapper for flashinfer's rotary embedding.
+    
+    This is an in-place operation that modifies query and key tensors directly.
+    """
+    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
+
+    apply_rope_with_cos_sin_cache_inplace(
+        positions=positions,
+        query=query,
+        key=key,
+        head_size=head_size,
+        cos_sin_cache=cos_sin_cache,
+        is_neox=is_neox,
+    )
+
+
+def _flashinfer_rotary_embedding_fake(
+    positions: torch.Tensor,
+    query: torch.Tensor,
+    key: torch.Tensor,
+    head_size: int,
+    cos_sin_cache: torch.Tensor,
+    is_neox: bool,
+) -> None:
+    return
+
+
+# Register flashinfer rotary embedding custom op
+direct_register_custom_op(
+    op_name="flashinfer_rotary_embedding",
+    op_func=_flashinfer_rotary_embedding,
+    mutates_args=["query", "key"],  # These tensors are modified in-place
+    fake_impl=_flashinfer_rotary_embedding_fake,
+    dispatch_key=current_platform.dispatch_key,
+)
diff --git a/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py b/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
index 7ac2e4bb6c34..736ec2c1dd3a 100644
--- a/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
+++ b/vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py
@@ -97,15 +97,13 @@ def forward_native(
     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
         """PyTorch-native implementation equivalent to forward()."""
         assert key is not None
+        self._match_cos_sin_cache_dtype(query)
         query_rot = query[..., :self.rotary_dim]
         key_rot = key[..., :self.rotary_dim]
         if self.rotary_dim < self.head_size:
             query_pass = query[..., self.rotary_dim:]
             key_pass = key[..., self.rotary_dim:]
 
-        if self.cos_sin_cache.device != positions.device:
-            self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(
-                positions.device)
         cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
                                      if offsets is not None else positions]
         cos, sin = cos_sin.chunk(2, dim=-1)
diff --git a/vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py b/vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
index 37ead43e22bc..871728035306 100644
--- a/vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
+++ b/vllm/model_executor/layers/rotary_embedding/llama4_vision_rope.py
@@ -59,7 +59,7 @@ def forward_native(  # type: ignore[override]
         key: Optional[torch.Tensor] = None,
     ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
         assert key is not None
-        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
+        self._match_cos_sin_cache_dtype(query)
         query_ = torch.view_as_complex(query.float().reshape(
             *query.shape[:-1], -1, 2))
         key_ = torch.view_as_complex(key.float().reshape(
diff --git a/vllm/model_executor/layers/rotary_embedding/mrope.py b/vllm/model_executor/layers/rotary_embedding/mrope.py
index ccc59bbbe233..17d04a1ad715 100644
--- a/vllm/model_executor/layers/rotary_embedding/mrope.py
+++ b/vllm/model_executor/layers/rotary_embedding/mrope.py
@@ -245,6 +245,7 @@ def forward_native(
         assert positions.ndim == 1 or positions.ndim == 2
         assert key is not None
 
+        self._match_cos_sin_cache_dtype(query)
         num_tokens = positions.shape[-1]
         cos_sin = self.cos_sin_cache[positions]
         cos, sin = cos_sin.chunk(2, dim=-1)
@@ -293,6 +294,7 @@ def forward_cuda(
         assert positions.ndim == 1 or positions.ndim == 2
         assert key is not None
 
+        self._match_cos_sin_cache_dtype(query)
         num_tokens = positions.shape[-1]
         cos_sin = self.cos_sin_cache[positions]
         cos, sin = cos_sin.chunk(2, dim=-1)

```
