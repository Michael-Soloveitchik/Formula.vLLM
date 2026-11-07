# PR #24511: [Multi Modal][Performance] Fused Q,K's apply_rope into one

- field: attention  | area: model_executor_models
- author: wwl2755
- state: MERGED  created: 2025-09-09T14:10:40Z  merged: 2025-09-14T08:10:21Z
- url: https://github.com/vllm-project/vllm/pull/24511
- labels: ready, qwen

# Issue

## Problem

In Qwen2.5-VL vision attention, rotary position embeddings were applied to the query and key tensors in two separate `apply_rotary_pos_emb_vision` calls. Each call launches its own set of GPU kernels (cos/sin broadcast, elementwise multiply/add), so the rope step pays double the kernel-launch and memory-traffic overhead for two tensors of identical shape. Additionally, the attention output reshape path issued redundant `rearrange` operations, with an extra rearrange on the common path.

## Example

Running Qwen/Qwen2.5-VL-7B-Instruct, the Q,K rope step cost 731 µs wall / 485 µs GPU; a single fused call dropped it to 378 µs / 255 µs. The flash_attn output reshape merge saved a further ~12 µs.

## Use case

Prefill (TTFT) of multimodal requests where the vision encoder runs rope over image patch tokens before each attention layer.

## Resources

GPU compute and kernel-launch overhead, plus HBM memory traffic from reading/writing Q and K separately instead of as one batched tensor.

## Type of users

Users serving vision-language models (Qwen2.5-VL) with image/video inputs, where the vision tower runs per-layer rope.

## Kind of inputs

Image/video prompts producing many patch tokens, driving large Q/K tensors of shape [b, s, heads, head_dim] through the vision attention stack.

## Mechanism with the issue



# Solution

## Fix in words

Before, rope ran as two calls: `q = apply_rotary_pos_emb_vision(q, ...)` and `k = apply_rotary_pos_emb_vision(k, ...)`. After, Q and K are concatenated along dim 0 into `qk_concat` of shape [2*b, s, heads, head_dim], passed through a single `apply_rotary_pos_emb_vision` call, then split back with `torch.chunk(..., 2, dim=0)` — halving kernel launches and fusing the memory traffic. Separately, the trailing output reshape is folded into each backend branch: the flash_attn path now does `rearrange(output, "(b s) h d -> s b (h d)", ...).contiguous()` directly, and the TORCH_SDPA/XFORMERS branches get their own final rearrange, removing the single shared post-branch rearrange and saving ~12 µs on the flash_attn path.

## Author's solution

Author fused Q and K rope into one call in Qwen2.5-VL, cutting rope cost from 731 µs to 378 µs (GPU 485 µs to 255 µs), and merged multiple rearranges in the flash_attn code path for a further ~12 µs, yielding ~11% TTFT improvement. Author offered to extend the change to other models if reviewers approve.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/models/qwen2_5_vl.py`

### Diff
```diff
diff --git a/vllm/model_executor/models/qwen2_5_vl.py b/vllm/model_executor/models/qwen2_5_vl.py
index fc028aa2287a..dbf486374bcf 100644
--- a/vllm/model_executor/models/qwen2_5_vl.py
+++ b/vllm/model_executor/models/qwen2_5_vl.py
@@ -363,8 +363,10 @@ def forward(
         q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous()
                    for x in (q, k, v))
         if rotary_pos_emb is not None:
-            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
-            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
+            # [2 * b, s, heads, head_dim]
+            qk_concat = torch.cat([q, k], dim=0)
+            qk_rotated = apply_rotary_pos_emb_vision(qk_concat, rotary_pos_emb)
+            q, k = torch.chunk(qk_rotated, 2, dim=0)
 
         if self.is_flash_attn_backend:
             if self.attn_backend == _Backend.ROCM_AITER_FA:
@@ -388,8 +390,8 @@ def forward(
                                             causal=False)
 
             context_layer = rearrange(output,
-                                      "(b s) ... -> b s ...",
-                                      b=batch_size)
+                                      "(b s) h d -> s b (h d)",
+                                      b=batch_size).contiguous()
         elif self.attn_backend == _Backend.TORCH_SDPA:
             # Execute attention entry by entry for speed & less VRAM.
             outputs = []
@@ -408,6 +410,8 @@ def forward(
                 output_i = rearrange(output_i, "b h s d -> b s h d ")
                 outputs.append(output_i)
             context_layer = torch.cat(outputs, dim=1)
+            context_layer = rearrange(context_layer,
+                                      "b s h d -> s b (h d)").contiguous()
         elif self.attn_backend == _Backend.XFORMERS:
             from xformers import ops as xops
             from xformers.ops.fmha.attn_bias import BlockDiagonalMask
@@ -418,8 +422,8 @@ def forward(
 
             context_layer = xops.memory_efficient_attention_forward(
                 q, k, v, attn_bias=attn_bias, p=0, scale=None)
-        context_layer = rearrange(context_layer,
-                                  "b s h d -> s b (h d)").contiguous()
+            context_layer = rearrange(context_layer,
+                                      "b s h d -> s b (h d)").contiguous()
 
         output, _ = self.proj(context_layer)
         return output

```
