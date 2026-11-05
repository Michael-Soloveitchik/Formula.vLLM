# PR #42311: [Model] [Perf] Use flatten for Qwen3.5's GDN output projection

- field: attention  | area: model_executor_layers
- author: rishaps
- state: MERGED  created: 2026-05-11T10:51:46Z  merged: 2026-05-18T08:14:37Z
- url: https://github.com/vllm-project/vllm/pull/42311
- labels: ready, qwen

# Issue

## Problem

Qwen3.5's GDN (gated delta net) output projection merges the last two tensor dimensions (head and head-dim) using einops `rearrange(core_attn_out, "... h d -> ... (h d)")`. einops parses and validates the pattern string in Python on every call, which is expensive relative to the trivial reshape it performs. This Python-level overhead is incurred on every GDN forward pass.

## Example

Profiling `Qwen/Qwen3.5-0.8B` in eager mode: einops `rearrange` averaged 23.261 us/call on CUDA (RTX 3050) over 576 calls and 41.741 us/call on CPU over 144 calls, versus 2.828 us and 1.949 us for `aten::flatten` — ~8x (CUDA) to ~21x (CPU) per-call slower.

## Use case

Autoregressive serving/inference of Qwen3.5 models, where the GDN layer's output projection runs once per layer per forward step across prefill and every decode token.

## Resources

CPU / Python interpreter time (string parsing and validation overhead), plus kernel-launch/dispatch latency — the overhead is host-side, most visible in eager mode.

## Type of users

Users serving or benchmarking Qwen3.5 (GDN / linear-attention) models, on both CUDA GPUs and CPU, especially in eager mode where per-call Python overhead is not hidden by graph capture.

## Kind of inputs

Any input to Qwen3.5 GDN layers regardless of sequence length, batch size, or dtype — the overhead is per-forward-call and independent of tensor contents; it accumulates with number of forward steps (tokens * layers).

## Mechanism with the issue

In `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, the `_output_projection`, `forward_xpu`, and `forward_cpu` paths call `rearrange(core_attn_out, "... h d -> ... (h d)")` right before `self.out_proj`. einops re-parses the pattern string and validates shapes in Python on each invocation, which dominates the cost of what is just a contiguous merge of the final two dims.

# Solution

## Fix in words

The diff replaces `core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")` with `core_attn_out = core_attn_out.flatten(-2)` in three code paths (`_output_projection`, `forward_xpu`, `forward_cpu`), keeping the original pattern as an inline comment. `flatten(-2)` is the native PyTorch equivalent that merges the last two dimensions without any Python-level string parsing or validation. The reshape semantics are identical, so model behavior is unchanged, while eliminating einops overhead yields ~8x (CUDA) to ~21x (CPU) per-call speedup on this operation.

## Author's solution

Replace einops's `rearrange(core_attn_out, "... h d -> ... (h d)")` with the PyTorch-native equivalent `core_attn_out.flatten(-2)` to merge the last two dimensions, avoiding expensive Python-level pattern string parsing and validation on every GDN forward call. The author verified correctness via PPL comparison on Qwen3.5-0.8B (19.510 flatten vs 19.510 rearrange, both ~0.63% from HF expected), confirming the reshape does not alter tensor ordering or model output.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/mamba/gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn_linear_attn.py
index 10760d6116cc..160b81f3f9fa 100644
--- a/vllm/model_executor/layers/mamba/gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn_linear_attn.py
@@ -697,7 +697,7 @@ def _output_projection(
         z = z.reshape(-1, z.shape[-1])
         core_attn_out = self.norm(core_attn_out, z)
         core_attn_out = core_attn_out.reshape(z_shape_og)
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
         output[:num_tokens], _ = self.out_proj(core_attn_out)
 
     def forward_hip(
@@ -857,7 +857,7 @@ def forward_xpu(
         z = z.reshape(-1, z.shape[-1])
         core_attn_out = self.norm(core_attn_out, z)
         core_attn_out = core_attn_out.reshape(z_shape_og)
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
         output[:num_tokens], _ = self.out_proj(core_attn_out)
 
     def forward_cpu(
@@ -907,7 +907,7 @@ def forward_cpu(
         z = z.reshape(-1, z.shape[-1])
         core_attn_out = self.norm(core_attn_out, z)
         core_attn_out = core_attn_out.reshape(z_shape_og)
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
         output[:num_tokens], _ = self.out_proj(core_attn_out)
 
     def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None:

```
