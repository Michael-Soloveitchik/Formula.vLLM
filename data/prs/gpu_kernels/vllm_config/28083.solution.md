# PR #28083: [PERF] Decouple projections from GDN custom op. Attempt 2

- field: gpu_kernels  | area: vllm_config
- author: vadiklyutiy
- state: MERGED  created: 2025-11-05T00:50:08Z  merged: 2025-11-06T01:01:12Z
- url: https://github.com/vllm-project/vllm/pull/28083
- labels: ready, qwen

# Issue

## Problem

In Qwen3-Next GDN (gated delta net) attention, the entire forward pass—input projections, convolution+recurrent core attention, gated RMSNorm, and output projection—was wrapped inside a single `vllm::gdn_attention` custom op. Custom ops are opaque to `torch.compile` (registered as no-compile boundaries), so the large matmul-heavy input/output projections and the gated norm never got compiled or fused. This left significant performance on the table, especially in decode where kernel-launch overhead and un-fused elementwise ops dominate.

## Example

Serving Qwen/Qwen3-Next-80B-A3B-Instruct with tp=4, expert parallel, FLASH_ATTN backend: decode throughput at concurrency 1024 was 16586 tok/s before and 18953 after (+14.3%); another decode config gained +16.5% (19212 -> 22384 tok/s).

## Use case

High-concurrency online serving of a large MoE + linear-attention (GDN) model, where many small decode steps run back-to-back and compile-fused kernels matter.

## Resources

GPU compute and kernel-launch/dispatch overhead; un-compiled projections and a custom triton gated-rmsnorm kernel wasted launch time and missed fusion opportunities.

## Type of users

Users serving Qwen3-Next-class models (gated delta net linear attention + MoE) on multi-GPU tensor-parallel setups, particularly decode-heavy workloads.

## Kind of inputs

Short-input/long-output decode traffic (e.g. random-input 32, random-output 1024) at high max-concurrency (512-1024), plus large-prefill batches (input 8192).

## Mechanism with the issue

`Qwen3NextGatedDeltaNet.forward` called `torch.ops.vllm.gdn_attention`, whose `_forward` did `in_proj_qkvz`/`in_proj_ba`, the conv+recurrent core, `self.norm` (custom triton gated rmsnorm), and `out_proj` all inside the opaque custom op. Because the op is a `torch.compile` no-compile layer, none of those projections or the norm were ever compiled/fused.

# Solution

## Fix in words

The diff splits the monolithic op into three pieces: `forward` now runs the input projections (`in_proj_qkvz`, `in_proj_ba`, qkv reorder/cat) and the output projection (gated `self.norm` + `out_proj`) in plain Python/eager so `torch.compile` can compile and fuse them, while only the conv + recurrent computation stays behind a narrowed custom op renamed `gdn_attention_core` (taking `mixed_qkv, b, a, core_attn_out` and mutating `core_attn_out`). The compilation config's splitting-op list is updated `vllm::gdn_attention` -> `vllm::gdn_attention_core`. A new native-PyTorch `RMSNormGated` CustomOp is added in `layernorm.py` with a `forward_native` (torch rsqrt-based gated/group rmsnorm) that `torch.compile` turns into efficient code, replacing the fla triton version for the norm. Net effect: projections and gated norm become compiled/fused, yielding ~1% prefill and 5-16% decode throughput gains with equal-or-better gsm8k accuracy.

## Author's solution

Refactor GDN to widen `torch.compile` coverage by separating the forward pass into three distinct pieces—input projection, core attention, output projection—so that projections, previously buried in the GDN custom op and uncompiled, are now covered by `torch.compile`. Additionally add an `RMSNormGated` class implementing a torch-native gated rmsnorm for GDN, which `torch.compile` compiles into code even better than the previous custom triton kernel. This is the second attempt at #27512, which was reverted after unstable/unrelated CI failures (CUDART init, OOM).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/config/compilation.py`
- `vllm/model_executor/layers/layernorm.py`
- `vllm/model_executor/models/qwen3_next.py`

### Diff
```diff
diff --git a/vllm/config/compilation.py b/vllm/config/compilation.py
index 00e8cbfd7319..650104b62d3f 100644
--- a/vllm/config/compilation.py
+++ b/vllm/config/compilation.py
@@ -462,7 +462,7 @@ class CompilationConfig:
         "vllm::short_conv",
         "vllm::linear_attention",
         "vllm::plamo2_mamba_mixer",
-        "vllm::gdn_attention",
+        "vllm::gdn_attention_core",
         "vllm::kda_attention",
         "vllm::sparse_attn_indexer",
     ]
diff --git a/vllm/model_executor/layers/layernorm.py b/vllm/model_executor/layers/layernorm.py
index 65432c0fb2d4..a883ac81f41e 100644
--- a/vllm/model_executor/layers/layernorm.py
+++ b/vllm/model_executor/layers/layernorm.py
@@ -369,6 +369,109 @@ def forward_cuda(
         return self.forward_native(x, residual)
 
 
+@CustomOp.register("rms_norm_gated")
+class RMSNormGated(CustomOp):
+    """RMS Normalization with optional gating.
+
+    This is a native PyTorch implementation that supports:
+    - Standard RMS normalization
+    - Group RMS normalization
+    - Optional gating with SiLU activation
+    """
+
+    def __init__(
+        self,
+        hidden_size: int,
+        eps: float = 1e-5,
+        group_size: int | None = None,
+        norm_before_gate: bool = False,
+        device: torch.device | None = None,
+        dtype: torch.dtype | None = None,
+    ):
+        """Initialize RMSNormGated.
+
+        Args:
+            hidden_size: Size of the hidden dimension
+            eps: Epsilon for numerical stability
+            group_size: If not None, do GroupNorm with each group
+                        having group_size elements.
+                        group_size=None is equivalent to group_size=hidden_size
+                        (i.e. there's only 1 group).
+            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
+                              If False and z is provided: out = norm(x * silu(z))
+            device: Device to create parameters on
+            dtype: Data type for parameters
+        """
+        factory_kwargs = {"device": device, "dtype": dtype}
+        super().__init__()
+        self.eps = eps
+        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
+        self.register_parameter("bias", None)
+        self.group_size = group_size
+        self.norm_before_gate = norm_before_gate
+        self.reset_parameters()
+
+    def reset_parameters(self):
+        torch.nn.init.ones_(self.weight)
+
+    def forward_native(
+        self, x: torch.Tensor, z: torch.Tensor | None = None
+    ) -> torch.Tensor:
+        """
+        Native PyTorch implementation of RMS normalization with gating.
+
+        Args:
+            x: Input tensor
+            z: Optional gating tensor
+
+        Returns:
+            Normalized (and optionally gated) tensor
+
+        If z is not None:
+            - norm_before_gate=True: out = norm(x) * silu(z)
+            - norm_before_gate=False: out = norm(x * silu(z))
+        """
+        # Apply gating before normalization if needed
+        if z is not None and not self.norm_before_gate:
+            x = x * F.silu(z)
+
+        # RMS Normalization
+        if self.group_size is None:
+            # Standard RMS norm across the last dimension
+            variance = x.pow(2).mean(dim=-1, keepdim=True)
+            x_normed = x * torch.rsqrt(variance + self.eps)
+            out = x_normed * self.weight
+        else:
+            # Group RMS norm
+            from einops import rearrange
+
+            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
+            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
+            x_normed = x_group * torch.rsqrt(variance + self.eps)
+            out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight
+
+        # Apply gating after normalization if needed
+        if z is not None and self.norm_before_gate:
+            out = out * F.silu(z)
+
+        return out
+
+    def forward_cuda(
+        self, x: torch.Tensor, z: torch.Tensor | None = None
+    ) -> torch.Tensor:
+        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
+
+        return rmsnorm_fn(
+            x,
+            self.weight,
+            self.bias,
+            z=z,
+            eps=self.eps,
+            group_size=self.group_size,
+            norm_before_gate=self.norm_before_gate,
+        )
+
+
 class LayerNorm(nn.Module):
     """
     Layer Normalization.
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index 1dd67bb66d65..b095c79dc954 100644
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -30,12 +30,14 @@
 from vllm.forward_context import ForwardContext, get_forward_context
 from vllm.logger import init_logger
 from vllm.model_executor.layers.fla.ops import (
-    RMSNormGated,
     chunk_gated_delta_rule,
     fused_recurrent_gated_delta_rule,
 )
 from vllm.model_executor.layers.fused_moe import SharedFusedMoE
-from vllm.model_executor.layers.layernorm import GemmaRMSNorm as Qwen3NextRMSNorm
+from vllm.model_executor.layers.layernorm import (
+    GemmaRMSNorm as Qwen3NextRMSNorm,
+)
+from vllm.model_executor.layers.layernorm import RMSNormGated
 from vllm.model_executor.layers.linear import (
     ColumnParallelLinear,
     QKVParallelLinear,
@@ -436,17 +438,66 @@ def forward(
         hidden_states: torch.Tensor,
         output: torch.Tensor,
     ):
-        return torch.ops.vllm.gdn_attention(
-            hidden_states,
-            output,
+        """
+        Forward pass with three parts:
+        1. Input projection
+        2. Core attention (custom op)
+        3. Output projection
+        """
+        num_tokens = hidden_states.size(0)
+
+        # ============================================================
+        # Part 1: Input Projection
+        # ============================================================
+        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
+        projected_states_ba, _ = self.in_proj_ba(hidden_states)
+        query, key, value, z, b, a = self.fix_query_key_value_ordering(
+            projected_states_qkvz, projected_states_ba
+        )
+        query, key, value = map(
+            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
+        )
+        mixed_qkv = torch.cat((query, key, value), dim=-1)
+
+        # ============================================================
+        # Part 2: Core Attention (Custom Op)
+        # ============================================================
+        core_attn_out = torch.zeros(
+            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
+            dtype=hidden_states.dtype,
+            device=hidden_states.device,
+        )
+
+        torch.ops.vllm.gdn_attention_core(
+            mixed_qkv,
+            b,
+            a,
+            core_attn_out,
             self.prefix,
         )
 
-    def _forward(
+        # ============================================================
+        # Part 3: Output Projection
+        # ============================================================
+        z_shape_og = z.shape
+        # Reshape input data into 2D tensor
+        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
+        z = z.reshape(-1, z.shape[-1])
+        core_attn_out = self.norm(core_attn_out, z)
+        core_attn_out = core_attn_out.reshape(z_shape_og)
+        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
+        output[:num_tokens], _ = self.out_proj(core_attn_out)
+
+    def _forward_core(
         self,
-        hidden_states: torch.Tensor,
-        output: torch.Tensor,
+        mixed_qkv: torch.Tensor,
+        b: torch.Tensor,
+        a: torch.Tensor,
+        core_attn_out: torch.Tensor,
     ):
+        """
+        Core attention computation (called by custom op).
+        """
         forward_context = get_forward_context()
         attn_metadata: AttentionMetadata = forward_context.attn_metadata
 
@@ -471,18 +522,11 @@ def _forward(
         num_actual_tokens = attn_metadata.num_actual_tokens
         num_accepted_tokens = attn_metadata.num_accepted_tokens
 
-        # 1. Set up dimensions for reshapes later
-        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states[:num_actual_tokens])
-        projected_states_ba, _ = self.in_proj_ba(hidden_states[:num_actual_tokens])
-        query, key, value, z, b, a = self.fix_query_key_value_ordering(
-            projected_states_qkvz, projected_states_ba
-        )
-        query, key, value = map(
-            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
-        )
-        mixed_qkv = torch.cat((query, key, value), dim=-1)
+        mixed_qkv = mixed_qkv[:num_actual_tokens]
+        b = b[:num_actual_tokens]
+        a = a[:num_actual_tokens]
 
-        # 2. Convolution sequence transformation
+        # 1. Convolution sequence transformation
         conv_weights = self.conv1d.weight.view(
             self.conv1d.weight.size(0), self.conv1d.weight.size(2)
         )
@@ -498,7 +542,7 @@ def _forward(
             mixed_qkv_spec = None
             mixed_qkv_non_spec = mixed_qkv
 
-        # 2.1: process the mutli-query part
+        # 1.1: Process the multi-query part
         if spec_sequence_masks is not None:
             mixed_qkv_spec = causal_conv1d_update(
                 mixed_qkv_spec,
@@ -515,7 +559,7 @@ def _forward(
                 validate_data=False,
             )
 
-        # 2.2: process the remaining part
+        # 1.2: Process the remaining part
         if attn_metadata.num_prefills > 0:
             mixed_qkv_non_spec_T = mixed_qkv_non_spec.transpose(0, 1)
             # - "cache_indices" updates the conv_state cache in positions
@@ -570,9 +614,9 @@ def _forward(
             g_non_spec = g
             beta_non_spec = beta
 
-        # 3. Recurrent attention
+        # 2. Recurrent attention
 
-        # 3.1: process the mutlti-query part
+        # 2.1: Process the multi-query part
         if spec_sequence_masks is not None:
             core_attn_out_spec, last_recurrent_state = fused_recurrent_gated_delta_rule(
                 q=query_spec,
@@ -590,7 +634,7 @@ def _forward(
         else:
             core_attn_out_spec, last_recurrent_state = None, None
 
-        # 3.2: process the remaining part
+        # 2.2: Process the remaining part
         if attn_metadata.num_prefills > 0:
             initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
             initial_state[~has_initial_state, ...] = 0
@@ -633,30 +677,20 @@ def _forward(
         else:
             core_attn_out_non_spec, last_recurrent_state = None, None
 
-        # Merge core attention output
+        # 3. Merge core attention output
         if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
-            core_attn_out = torch.empty(
+            merged_out = torch.empty(
                 (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                 dtype=core_attn_out_non_spec.dtype,
                 device=core_attn_out_non_spec.device,
             )
-            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
-            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
-
+            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
+            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
+            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
         elif spec_sequence_masks is not None:
-            core_attn_out = core_attn_out_spec
+            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
         else:
-            core_attn_out = core_attn_out_non_spec
-
-        z_shape_og = z.shape
-        # reshape input data into 2D tensor
-        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
-        z = z.reshape(-1, z.shape[-1])
-        core_attn_out = self.norm(core_attn_out, z)
-        core_attn_out = core_attn_out.reshape(z_shape_og)
-        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
-
-        output[:num_actual_tokens], _ = self.out_proj(core_attn_out)
+            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
 
 
 class Qwen3NextAttention(nn.Module):
@@ -1260,29 +1294,44 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
         return self.model.get_expert_mapping()
 
 
-def gdn_attention(
-    hidden_states: torch.Tensor,
-    output: torch.Tensor,
+def gdn_attention_core(
+    mixed_qkv: torch.Tensor,
+    b: torch.Tensor,
+    a: torch.Tensor,
+    core_attn_out: torch.Tensor,
     layer_name: str,
 ) -> None:
+    """
+    Custom op for the core attention computation.
+    Only handles the convolution + recurrent attention part.
+    Input/output projections are handled outside this op.
+    """
     forward_context: ForwardContext = get_forward_context()
     self = forward_context.no_compile_layers[layer_name]
-    self._forward(hidden_states=hidden_states, output=output)
+    self._forward_core(
+        mixed_qkv=mixed_qkv,
+        b=b,
+        a=a,
+        core_attn_out=core_attn_out,
+    )
 
 
-def gdn_attention_fake(
-    hidden_states: torch.Tensor,
-    output: torch.Tensor,
+def gdn_attention_core_fake(
+    mixed_qkv: torch.Tensor,
+    b: torch.Tensor,
+    a: torch.Tensor,
+    core_attn_out: torch.Tensor,
     layer_name: str,
 ) -> None:
+    """Fake implementation for torch.compile."""
     return
 
 
 direct_register_custom_op(
-    op_name="gdn_attention",
-    op_func=gdn_attention,
-    mutates_args=["output"],
-    fake_impl=gdn_attention_fake,
+    op_name="gdn_attention_core",
+    op_func=gdn_attention_core,
+    mutates_args=["core_attn_out"],
+    fake_impl=gdn_attention_core_fake,
 )
 
 

```
