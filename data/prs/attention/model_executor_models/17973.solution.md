# PR #17973: [PERF] Speed up Qwen2.5-VL model by speed up rotary position embedding const…

- field: attention  | area: model_executor_models
- author: vadiklyutiy
- state: MERGED  created: 2025-05-12T01:42:53Z  merged: 2025-05-16T06:31:02Z
- url: https://github.com/vllm-project/vllm/pull/17973
- labels: ready, qwen

# Issue

## Problem

Qwen2.5-VL vision transformer built all rotary position embedding and window-index constant tensors at the start of every `forward()` call. Computation mixed CPU and GPU tensors, causing many small host-device transfers (cuMemcpy) back and forth. These constants depend only on `grid_thw` shape, yet were recomputed from scratch each forward with no caching.

## Example

Serving Qwen2.5-VL-3B on H100 at 50 reqs/s with one 512x512 image per request. Profile showed a pink `tmp` region at start of `Qwen2_5_VisionTransformer.forward()` before main transformer began, dominated by constant-tensor setup.

## Use case

Online serving of vision-language model where many requests share same image resolutions, so same `grid_thw` recurs across requests.

## Resources

CPU compute plus PCIe host-device bandwidth (many tiny cuMemcpy ops), and kernel-launch overhead for scattered small tensor ops on GPU.

## Type of users

Users serving Qwen2.5-VL multimodal inference, especially high-throughput image workloads on single or multi-GPU.

## Kind of inputs

Image/video requests with repeated resolutions (same t,h,w grid dims), high request rate, batched small images.

## Mechanism with the issue

`rot_pos_emb` and `get_window_index` looped over all `grid_thw` rows building pos_ids, window indices, cu_seqlens with tensors defaulting to GPU device. Per-forward recompute plus `.item()`/`.tolist()` calls forced device syncs. No memoization across identical grid shapes.

# Solution

## Fix in words

Diff splits per-grid logic into `rotary_pos_emb_thw(t,h,w)` and `get_window_index_thw(grid_t,grid_h,grid_w)` operating on single grid tuples, wraps combined result in `get_rope_by_thw` decorated with `@lru_cache(maxsize=1024)` so identical (t,h,w) grids reuse cached CPU tensors. `inv_freq` and all constant tensors pinned to `device='cpu'`. `forward` now iterates `grid_thw` as a Python `list[list[int]]`, accumulates per-grid results, and moves final `cu_seqlens`, `cu_window_seqlens`, `rotary_pos_emb`, `window_index` to GPU once via `.to(device=..., non_blocking=True)`. Net effect: constant-mrope construction ~5x faster, 2.46% E2E throughput gain, correctness unchanged.

## Author's solution

Refactored so all tensors needed to create constant mrope data live on CPU (mirroring how language-model mrope works), eliminating the CPU/GPU mix and small back-and-forth transfers. Regrouped the calculation by `grid_thw` line and cached results so repeated grid shapes reuse prior computation.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/qwen2_5_vl.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/qwen2_5_vl.py b/vllm/model_executor/models/qwen2_5_vl.py
index 5904ad1f1f24..68dd07820189 100644
--- a/vllm/model_executor/models/qwen2_5_vl.py
+++ b/vllm/model_executor/models/qwen2_5_vl.py
@@ -25,7 +25,7 @@
 # limitations under the License.
 """Inference-only Qwen2.5-VL model compatible with HuggingFace weights."""
 from collections.abc import Iterable, Mapping
-from functools import partial
+from functools import lru_cache, partial
 from typing import Callable, Literal, Optional, TypedDict, Union
 
 import torch
@@ -478,8 +478,8 @@ def __init__(self, dim: int, theta: float = 10000.0) -> None:
         super().__init__()
         self.dim = dim
         self.theta = theta
-        inv_freq = 1.0 / (theta
-                          **(torch.arange(0, dim, 2, dtype=torch.float) / dim))
+        inv_freq = 1.0 / (theta**(
+            torch.arange(0, dim, 2, dtype=torch.float, device='cpu') / dim))
         self.register_buffer("inv_freq", inv_freq, persistent=False)
         self._seq_len_cached = 0
         self._freqs_cached = None
@@ -520,7 +520,7 @@ def __init__(
         self.hidden_size = vision_config.hidden_size
         self.num_heads = vision_config.num_heads
 
-        # args for get_window_index
+        # args for get_window_index_thw
         self.window_size = vision_config.window_size
         self.patch_size = vision_config.patch_size
         self.spatial_merge_size = vision_config.spatial_merge_size
@@ -567,65 +567,71 @@ def dtype(self) -> torch.dtype:
     def device(self) -> torch.device:
         return self.patch_embed.proj.weight.device
 
-    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
-        pos_ids = []
-        for t, h, w in grid_thw:
-            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
-            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
-            hpos_ids = hpos_ids.reshape(
-                h // self.spatial_merge_size,
-                self.spatial_merge_size,
-                w // self.spatial_merge_size,
-                self.spatial_merge_size,
-            ).permute(0, 2, 1, 3).flatten()
-            wpos_ids = wpos_ids.reshape(
-                h // self.spatial_merge_size,
-                self.spatial_merge_size,
-                w // self.spatial_merge_size,
-                self.spatial_merge_size,
-            ).permute(0, 2, 1, 3).flatten()
-            pos_ids.append(
-                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
-        pos_ids = torch.cat(pos_ids, dim=0)
-        max_grid_size = grid_thw[:, 1:].max()
-        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
+    def rotary_pos_emb_thw(self, t, h, w):
+        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
+        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
+        hpos_ids = hpos_ids.reshape(
+            h // self.spatial_merge_size,
+            self.spatial_merge_size,
+            w // self.spatial_merge_size,
+            self.spatial_merge_size,
+        ).permute(0, 2, 1, 3).flatten()
+        wpos_ids = wpos_ids.reshape(
+            h // self.spatial_merge_size,
+            self.spatial_merge_size,
+            w // self.spatial_merge_size,
+            self.spatial_merge_size,
+        ).permute(0, 2, 1, 3).flatten()
+        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
+        max_size = max(h, w)
+        rotary_pos_emb_full = self.rotary_pos_emb(max_size)
         rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
+        rotary_pos_emb = rotary_pos_emb.reshape(
+            rotary_pos_emb.shape[0] // self.spatial_merge_unit,
+            self.spatial_merge_unit, -1)
+
         return rotary_pos_emb
 
-    def get_window_index(self, grid_thw):
-        window_index: list = []
-        cu_window_seqlens: list = [0]
-        window_index_id = 0
+    def get_window_index_thw(self, grid_t, grid_h, grid_w):
         vit_merger_window_size = (self.window_size //
                                   self.spatial_merge_size // self.patch_size)
 
-        for grid_t, grid_h, grid_w in grid_thw:
-            llm_grid_h = grid_h // self.spatial_merge_size
-            llm_grid_w = grid_w // self.spatial_merge_size
-            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
-                grid_t, llm_grid_h, llm_grid_w)
-            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
-            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
-            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
-            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
-            index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
-            index_padded = index_padded.reshape(grid_t, num_windows_h,
-                                                vit_merger_window_size,
-                                                num_windows_w,
-                                                vit_merger_window_size)
-            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
-                grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
-                vit_merger_window_size)
-            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
-            index_padded = index_padded.reshape(-1)
-            index_new = index_padded[index_padded != -100]
-            window_index.append(index_new + window_index_id)
-            cu_seqlens_tmp = seqlens.cumsum(
-                0) * self.spatial_merge_unit + cu_window_seqlens[-1]
-            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
-            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
-        window_index = torch.cat(window_index, dim=0)
-        return window_index, cu_window_seqlens
+        llm_grid_h = grid_h // self.spatial_merge_size
+        llm_grid_w = grid_w // self.spatial_merge_size
+        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
+            grid_t, llm_grid_h, llm_grid_w)
+        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
+        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
+        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
+        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
+        index_padded = F.pad(index, (0, pad_w, 0, pad_h), 'constant', -100)
+        index_padded = index_padded.reshape(grid_t, num_windows_h,
+                                            vit_merger_window_size,
+                                            num_windows_w,
+                                            vit_merger_window_size)
+        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
+            grid_t, num_windows_h * num_windows_w, vit_merger_window_size,
+            vit_merger_window_size)
+        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
+        index_padded = index_padded.reshape(-1)
+        index_new = index_padded[index_padded != -100]
+        cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit
+        cu_seqlens_tmp = cu_seqlens_tmp.to(dtype=torch.int32)
+        cu_seqlens_tmp = torch.unique_consecutive(cu_seqlens_tmp)
+
+        return index_new, cu_seqlens_tmp
+
+    @lru_cache(maxsize=1024)  # noqa: B019
+    def get_rope_by_thw(self, t, h, w):
+        window_index_thw, cu_seqlens_window_thw = self.get_window_index_thw(
+            t, h, w)
+        rotary_pos_emb_thw = self.rotary_pos_emb_thw(t, h, w)
+        rotary_pos_emb_thw = rotary_pos_emb_thw[window_index_thw, :, :]
+        rotary_pos_emb_thw = rotary_pos_emb_thw.flatten(start_dim=0, end_dim=1)
+        cu_seqlens_thw = torch.repeat_interleave(
+            torch.tensor([h * w], dtype=torch.int32), t)
+        return (rotary_pos_emb_thw, window_index_thw, cu_seqlens_window_thw,
+                cu_seqlens_thw)
 
     def compute_attn_mask_seqlen(
         self,
@@ -641,45 +647,74 @@ def compute_attn_mask_seqlen(
     def forward(
         self,
         x: torch.Tensor,
-        grid_thw: torch.Tensor,
+        grid_thw: list[list[int]],
     ) -> torch.Tensor:
         # patchify
+        seq_len, _ = x.size()
+        rotary_pos_emb = []
+        window_index: list = []
+        cu_window_seqlens: list = [torch.tensor([0], dtype=torch.int32)]
+        cu_seqlens: list = []
+
         hidden_states = x.to(device=self.device, dtype=self.dtype)
         hidden_states = self.patch_embed(hidden_states)
 
-        # compute position embedding
-        rotary_pos_emb = self.rot_pos_emb(grid_thw)
+        window_index_id = 0
+        cu_window_seqlens_last = 0
+        for t, h, w in grid_thw:
+            t, h, w = int(t), int(h), int(w)
+            llm_h = h // self.spatial_merge_size
+            llm_w = w // self.spatial_merge_size
+
+            (
+                rotary_pos_emb_thw,
+                window_index_thw,
+                cu_seqlens_window_thw,
+                cu_seqlens_thw,
+            ) = self.get_rope_by_thw(t, h, w)
+
+            window_index.append(window_index_thw + window_index_id)
+            window_index_id += (t * llm_h * llm_w)
+
+            cu_seqlens_window_thw = (cu_seqlens_window_thw +
+                                     cu_window_seqlens_last)
+            cu_window_seqlens_last = cu_seqlens_window_thw[-1]
+            cu_window_seqlens.append(cu_seqlens_window_thw)
 
-        # windows attention
-        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
-        cu_window_seqlens = torch.tensor(
-            cu_window_seqlens,
-            device=hidden_states.device,
-            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32)
+            rotary_pos_emb.append(rotary_pos_emb_thw)
+
+            cu_seqlens.append(cu_seqlens_thw)
+
+        rotary_pos_emb = torch.cat(rotary_pos_emb)
+        window_index = torch.cat(window_index)
+        cu_window_seqlens = torch.cat(cu_window_seqlens)
         cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
-        seq_len, _ = hidden_states.size()
-        hidden_states = hidden_states.reshape(
-            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
-        hidden_states = hidden_states[window_index, :, :]
-        hidden_states = hidden_states.reshape(seq_len, -1)
-        rotary_pos_emb = rotary_pos_emb.reshape(
-            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
-        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
-        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
-        # compute cu_seqlens
-        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
-                                             grid_thw[:, 0]).cumsum(
-                                                 dim=0, dtype=torch.int32)
+        cu_seqlens = torch.cat(cu_seqlens)
+        cu_seqlens = torch.cumsum(cu_seqlens, dim=0, dtype=torch.int32)
         cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)
 
         # transformers
-        hidden_states = hidden_states.unsqueeze(1)
-
         # pre-compute seqlens for window/full attn to reduce cuMemcpy operations
         max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(
             cu_seqlens)
         max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(
             cu_window_seqlens)
+
+        cu_seqlens = cu_seqlens.to(device=self.device, non_blocking=True)
+        cu_window_seqlens = cu_window_seqlens.to(device=self.device,
+                                                 non_blocking=True)
+        rotary_pos_emb = rotary_pos_emb.to(device=self.device,
+                                           non_blocking=True)
+        window_index = window_index.to(device=hidden_states.device,
+                                       non_blocking=True)
+
+        hidden_states = hidden_states.reshape(
+            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
+        hidden_states = hidden_states[window_index, :, :]
+        hidden_states = hidden_states.reshape(seq_len, -1)
+
+        hidden_states = hidden_states.unsqueeze(1)
+
         for layer_num, blk in enumerate(self.blocks):
             if layer_num in self.fullatt_block_indexes:
                 cu_seqlens_now = cu_seqlens
@@ -932,12 +967,13 @@ def _process_image_input(
 
         grid_thw = image_input["image_grid_thw"]
         assert grid_thw.ndim == 2
+        grid_thw_list = grid_thw.tolist()
 
         if image_input["type"] == "image_embeds":
             image_embeds = image_input["image_embeds"].type(self.visual.dtype)
         else:
             pixel_values = image_input["pixel_values"].type(self.visual.dtype)
-            image_embeds = self.visual(pixel_values, grid_thw=grid_thw)
+            image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
 
         # Split concatenated embeddings for each image item.
         merge_size = self.visual.spatial_merge_size
@@ -951,13 +987,15 @@ def _process_video_input(
 
         grid_thw = video_input["video_grid_thw"]
         assert grid_thw.ndim == 2
+        grid_thw_list = grid_thw.tolist()
 
         if video_input["type"] == "video_embeds":
             video_embeds = video_input["video_embeds"].type(self.visual.dtype)
         else:
             pixel_values_videos = video_input["pixel_values_videos"].type(
                 self.visual.dtype)
-            video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)
+            video_embeds = self.visual(pixel_values_videos,
+                                       grid_thw=grid_thw_list)
 
         # Split concatenated embeddings for each video item.
         merge_size = self.visual.spatial_merge_size

```
