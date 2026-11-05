# PR #28587: [Hybrid][torch.compile] Refactor mamba2 forward to avoid obscuring linear projections under custom op

- field: gpu_kernels  | area: model_executor_models
- author: tomeras91
- state: MERGED  created: 2025-11-12T20:11:14Z  merged: 2025-11-19T00:49:36Z
- url: https://github.com/vllm-project/vllm/pull/28587
- labels: performance, ready, torch.compile

# Issue

## Problem

The mamba_mixer2 custom op wrapped the entire Mamba2 block forward pass: input projection, conv+SSM token mixing, gated RMSNorm, and output projection. Only the token-mixing ops actually need to be in the custom op (they have input-dependent control flow like cu_seqlens that breaks piecewise CUDA graphs). Bundling the projections and RMSNorm inside the custom op hid them from torch.compile/Inductor, so trivial kernel-fusion opportunities were missed.

## Example

For nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8 at ISL/OSL/BS = 1K/1K/256 on a single H100, the obscured RMSNorm and projections left throughput and TTFT on the table (~12% throughput and ~22% TTFT recoverable once fused).

## Use case

Serving hybrid Mamba2/attention models (Nemotron-H, Bamba, Falcon-H1, GraniteMoE-hybrid, Zamba2) under continuous batching and chunked prefill with torch.compile enabled.

## Resources

GPU compute and HBM bandwidth wasted on unfused elementwise/norm ops plus extra kernel launches; also missed norm+quant fusion for quantized models.

## Type of users

Users serving hybrid state-space models with torch.compile, especially FP8/quantized deployments where RMSNorm could fuse with the quant op before projections.

## Kind of inputs

Mixed prefill/decode batches in continuous batching; large batch sizes and long prompts (1K ISL, BS 256) where projection + norm overhead is significant. Any dtype, with extra gains for FP8.

## Mechanism with the issue

In mamba_mixer2.py forward_cuda, in_proj, split, gated-norm (self.norm), and out_proj all ran inside torch.ops.vllm.mamba_mixer2. Because the custom op is opaque to Inductor, the many-op RMSNorm and the linear projections could not be fused or quant-fused, unlike the Attention pattern where QKV/out projections sit outside the custom op.

# Solution

## Fix in words

Splits the Mamba2 forward so only conv+SSM token mixing stays in the custom op. The new forward() runs in_proj (plus mup_vector scaling), allocates ssm_output, calls torch.ops.vllm.mamba_mixer2 on the projected_states, then runs the gated RMSNorm (self.norm) and out_proj outside the op before returning output. The custom op now dispatches to a renamed conv_ssm_forward that receives projected_states, does the hidden_states_B_C/dt split internally, writes results directly into the passed-in output buffer (removing the separate preallocated_ssm_out), and the gate is sliced from projected_states. conv_weights and the split_hidden_states_B_C_fn and TP-ed sizes are precomputed in __init__ (conv_weights registered as a non-persistent buffer). Callers in bamba/falcon_h1/granitemoehybrid/mamba2/nemotron_h/zamba2 drop the preallocated output arg and now use output = self.mamba(hidden_states). This exposes projections + norm to Inductor fusion, yielding ~10-15% throughput gain and ~22% TTFT reduction.

## Author's solution

Move everything except the token-mixing conv+SSM ops out of the custom op, mirroring how Attention keeps only the attention op custom while QKV/output projections live outside. This lets torch.compile/Inductor fuse the RMSNorm and projections (and enables norm+quant fusion for quantized models), while the custom op still provides piecewise CUDA graph breaks for the input-dependent control flow in token mixing.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`
- `vllm/model_executor/models/bamba.py`
- `vllm/model_executor/models/falcon_h1.py`
- `vllm/model_executor/models/granitemoehybrid.py`
- `vllm/model_executor/models/mamba2.py`
- `vllm/model_executor/models/nemotron_h.py`
- `vllm/model_executor/models/zamba2.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/mamba/mamba_mixer2.py b/vllm/model_executor/layers/mamba/mamba_mixer2.py
index fb45afa33dad..57313990b820 100644
--- a/vllm/model_executor/layers/mamba/mamba_mixer2.py
+++ b/vllm/model_executor/layers/mamba/mamba_mixer2.py
@@ -426,6 +426,10 @@ def __init__(
         # `ColumnParallelLinear` and `MergedColumnParallelLinear`,
         # and `set_weight_attrs` doesn't allow to override it
         self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
+        conv_weights = self.conv1d.weight.view(
+            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
+        )
+        self.register_buffer("conv_weights", conv_weights, persistent=False)
 
         # - these are TPed by heads to reduce the size of the
         #   temporal shape
@@ -459,6 +463,17 @@ def __init__(
             intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
         )
 
+        # - get hidden_states, B and C after depthwise convolution.
+        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
+            hidden_states_B_C,
+            [
+                self.intermediate_size // self.tp_size,
+                self.groups_ssm_state_size // self.tp_size,
+                self.groups_ssm_state_size // self.tp_size,
+            ],
+            dim=-1,
+        )
+
         compilation_config = get_current_vllm_config().compilation_config
         if prefix in compilation_config.static_forward_context:
             raise ValueError(f"Duplicate layer name: {prefix}")
@@ -470,10 +485,24 @@ def __init__(
         self.cache_config = cache_config
         self.prefix = prefix
 
+        # Pre-compute sizes for forward pass
+        self.tped_intermediate_size = self.intermediate_size // self.tp_size
+        self.tped_conv_size = self.conv_dim // self.tp_size
+        self.tped_dt_size = self.num_heads // self.tp_size
+
+        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
+            hidden_states_B_C,
+            [
+                self.tped_intermediate_size,
+                self.groups_ssm_state_size // self.tp_size,
+                self.groups_ssm_state_size // self.tp_size,
+            ],
+            dim=-1,
+        )
+
     def forward_native(
         self,
         hidden_states: torch.Tensor,
-        output: torch.Tensor,
         mup_vector: torch.Tensor | None = None,
     ):
         pass
@@ -481,22 +510,55 @@ def forward_native(
     def forward(
         self,
         hidden_states: torch.Tensor,
-        output: torch.Tensor,
         mup_vector: torch.Tensor | None = None,
     ):
+        # 1. Gated MLP's linear projection
+        projected_states, _ = self.in_proj(hidden_states)
+        if mup_vector is not None:
+            projected_states = projected_states * mup_vector
+
+        # 2. Prepare inputs for conv + SSM
+        ssm_output = torch.empty(
+            [
+                hidden_states.shape[0],
+                (self.num_heads // self.tp_size) * self.head_dim,
+            ],
+            dtype=hidden_states.dtype,
+            device=hidden_states.device,
+        )
+
+        # 3. conv + SSM
+        # (split `projected_states` into hidden_states_B_C, dt in the custom op to
+        # ensure it is not treated as an intermediate tensor by torch compile)
         torch.ops.vllm.mamba_mixer2(
-            hidden_states,
-            output,
+            projected_states,
+            ssm_output,
             self.prefix,
-            mup_vector,
         )
 
-    def forward_cuda(
+        # 4. gated MLP
+        # GatedRMSNorm internally applying SiLU to the gate
+        # SiLU is applied internally before normalization, unlike standard
+        # norm usage
+        gate = projected_states[..., : self.tped_intermediate_size]
+        hidden_states = self.norm(ssm_output, gate)
+
+        # 5. Final linear projection
+        output, _ = self.out_proj(hidden_states)
+
+        return output
+
+    def conv_ssm_forward(
         self,
-        hidden_states: torch.Tensor,
+        projected_states: torch.Tensor,
         output: torch.Tensor,
-        mup_vector: torch.Tensor | None = None,
     ):
+        hidden_states_B_C, dt = torch.split(
+            projected_states[..., self.tped_intermediate_size :],
+            [self.tped_conv_size, self.tped_dt_size],
+            dim=-1,
+        )
+
         forward_context = get_forward_context()
         # attn_metadata contains metadata necessary for the mamba2 triton
         # kernels to operate in continuous batching and in chunked prefill
@@ -524,46 +586,13 @@ def forward_cuda(
             cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
             last_chunk_indices_p = attn_metadata.last_chunk_indices_p
 
-        # 1. Gated MLP's linear projection
-        projected_states, _ = self.in_proj(hidden_states)
-
-        if mup_vector is not None:
-            projected_states = projected_states * mup_vector
-
-        gate, hidden_states_B_C, dt = torch.split(
-            projected_states,
-            [
-                self.intermediate_size // self.tp_size,
-                self.conv_dim // self.tp_size,
-                self.num_heads // self.tp_size,
-            ],
-            dim=-1,
-        )
-
-        conv_weights = self.conv1d.weight.view(
-            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
-        )
-
-        # - get hidden_states, B and C after depthwise convolution.
-        split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
-            hidden_states_B_C,
-            [
-                self.intermediate_size // self.tp_size,
-                self.groups_ssm_state_size // self.tp_size,
-                self.groups_ssm_state_size // self.tp_size,
-            ],
-            dim=-1,
-        )
-
         if attn_metadata is None:
             # profile run
             hidden_states_B_C = (
                 hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)
             ).contiguous()
-            hidden_states, _B, _C = split_hidden_states_B_C_fn(hidden_states_B_C)
-            hidden_states = self.norm(hidden_states, gate)
-            out, _ = self.out_proj(hidden_states)
-            return out
+            hidden_states, _B, _C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
+            return hidden_states
 
         # NOTE: V0 put prefill before decode, v1 puts decode before prefill
         num_prefills = attn_metadata.num_prefills  # request count
@@ -622,18 +651,8 @@ def forward_cuda(
             block_idx_first_scheduled_token_p = None
             num_computed_tokens_p = None
 
-        # Preallocate output tensor to avoid memcpy cost for merging prefill
-        # and decode outputs
-        preallocated_ssm_out = torch.empty(
-            [
-                num_prefill_tokens + num_decodes,
-                (self.num_heads // self.tp_size) * self.head_dim,
-            ],
-            dtype=hidden_states.dtype,
-            device=hidden_states.device,
-        )
         preallocated_ssm_out_d, preallocated_ssm_out_p = torch.split(
-            preallocated_ssm_out,
+            output[:num_actual_tokens],
             [num_decodes, num_prefill_tokens],
             dim=0,
         )
@@ -658,7 +677,7 @@ def forward_cuda(
             )  # this is the form that causal-conv see
             hidden_states_B_C_p = causal_conv1d_fn(
                 x,
-                conv_weights,
+                self.conv_weights,
                 self.conv1d.bias,
                 activation=self.activation,
                 conv_states=conv_state,
@@ -673,7 +692,9 @@ def forward_cuda(
                 query_start_loc=query_start_loc_p,
             ).transpose(0, 1)[:num_prefill_tokens]
 
-            hidden_states_p, B_p, C_p = split_hidden_states_B_C_fn(hidden_states_B_C_p)
+            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(
+                hidden_states_B_C_p
+            )
 
             # 3. State Space Model sequence transformation
             initial_states = None
@@ -815,7 +836,7 @@ def forward_cuda(
             hidden_states_B_C_d = causal_conv1d_update(
                 hidden_states_B_C_d,
                 conv_state,
-                conv_weights,
+                self.conv_weights,
                 self.conv1d.bias,
                 self.activation,
                 conv_state_indices=state_indices_tensor_d,
@@ -823,7 +844,9 @@ def forward_cuda(
                 initial_state_idx=block_idx_last_computed_token_d,
             )
 
-            hidden_states_d, B_d, C_d = split_hidden_states_B_C_fn(hidden_states_B_C_d)
+            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(
+                hidden_states_B_C_d
+            )
 
             # 3. State Space Model sequence transformation
             n_groups = self.n_groups // self.tp_size
@@ -861,15 +884,6 @@ def forward_cuda(
                 out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
             )
 
-        # 4. gated MLP
-        # GatedRMSNorm internally applying SiLU to the gate
-        # SiLU is applied internally before normalization, unlike standard
-        # norm usage
-        hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])
-
-        # 5. Final linear projection
-        output[:num_actual_tokens], _ = self.out_proj(hidden_states)
-
     def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
         assert self.model_config is not None
         assert self.cache_config is not None
@@ -901,21 +915,19 @@ def get_attn_backend(self) -> type["AttentionBackend"]:
 
 
 def mamba_mixer2(
-    hidden_states: torch.Tensor,
+    projected_states: torch.Tensor,
     output: torch.Tensor,
     layer_name: str,
-    mup_vector: torch.Tensor | None = None,
 ) -> None:
     forward_context: ForwardContext = get_forward_context()
     self = forward_context.no_compile_layers[layer_name]
-    self.forward_cuda(hidden_states=hidden_states, output=output, mup_vector=mup_vector)
+    self.conv_ssm_forward(projected_states=projected_states, output=output)
 
 
 def mamba_mixer2_fake(
-    hidden_states: torch.Tensor,
+    projected_states: torch.Tensor,
     output: torch.Tensor,
     layer_name: str,
-    mup_vector: torch.Tensor | None = None,
 ) -> None:
     return
 
diff --git a/vllm/model_executor/models/bamba.py b/vllm/model_executor/models/bamba.py
index e0a2defd5127..c6cc83487fec 100644
--- a/vllm/model_executor/models/bamba.py
+++ b/vllm/model_executor/models/bamba.py
@@ -138,8 +138,7 @@ def forward(
         else:
             hidden_states, residual = self.input_layernorm(hidden_states, residual)
 
-        output = torch.empty_like(hidden_states)
-        self.mamba(hidden_states, output)
+        output = self.mamba(hidden_states)
         # Fully Connected
         hidden_states, residual = self.pre_ff_layernorm(output, residual)
         hidden_states = self.feed_forward(hidden_states)
diff --git a/vllm/model_executor/models/falcon_h1.py b/vllm/model_executor/models/falcon_h1.py
index 3653425b8e1c..b985847af5da 100644
--- a/vllm/model_executor/models/falcon_h1.py
+++ b/vllm/model_executor/models/falcon_h1.py
@@ -198,10 +198,8 @@ def forward(
         residual: torch.Tensor | None,
         **kwargs,
     ):
-        output = torch.empty_like(hidden_states)
-        self.mamba(
+        output = self.mamba(
             hidden_states,
-            output,
             mup_vector=self.mup_vector,
         )
         return output, residual
diff --git a/vllm/model_executor/models/granitemoehybrid.py b/vllm/model_executor/models/granitemoehybrid.py
index 05177f1d1ac2..a340112ec62a 100644
--- a/vllm/model_executor/models/granitemoehybrid.py
+++ b/vllm/model_executor/models/granitemoehybrid.py
@@ -115,8 +115,7 @@ def forward(
     ):
         residual = hidden_states
         hidden_states = self.input_layernorm(hidden_states)
-        output = torch.empty_like(hidden_states)
-        self.mamba(hidden_states, output)
+        output = self.mamba(hidden_states)
         hidden_states = residual + output * self.residual_multiplier
 
         residual = hidden_states
diff --git a/vllm/model_executor/models/mamba2.py b/vllm/model_executor/models/mamba2.py
index fc17f98be198..5fcfa9431230 100644
--- a/vllm/model_executor/models/mamba2.py
+++ b/vllm/model_executor/models/mamba2.py
@@ -87,8 +87,7 @@ def forward(
         else:
             hidden_states, residual = self.norm(hidden_states, residual)
 
-        output = torch.empty_like(hidden_states)
-        self.mixer(hidden_states, output)
+        output = self.mixer(hidden_states)
         return output, residual
 
 
diff --git a/vllm/model_executor/models/nemotron_h.py b/vllm/model_executor/models/nemotron_h.py
index f7e0caf410e1..8675eff59222 100644
--- a/vllm/model_executor/models/nemotron_h.py
+++ b/vllm/model_executor/models/nemotron_h.py
@@ -376,8 +376,7 @@ def forward(
         else:
             hidden_states, residual = self.norm(hidden_states, residual)
 
-        output = torch.empty_like(hidden_states)
-        self.mixer(hidden_states, output)
+        output = self.mixer(hidden_states)
         return output, residual
 
 
diff --git a/vllm/model_executor/models/zamba2.py b/vllm/model_executor/models/zamba2.py
index 64e6979c8fcf..729a9655d087 100644
--- a/vllm/model_executor/models/zamba2.py
+++ b/vllm/model_executor/models/zamba2.py
@@ -567,11 +567,7 @@ def forward(
         hidden_states = self.input_layernorm(hidden_states)
 
         # Process through Mamba mixer
-        output = torch.empty_like(hidden_states)
-        self.mamba(
-            hidden_states,
-            output,
-        )
+        output = self.mamba(hidden_states)
 
         # residual connection after mamba
         hidden_states = residual + output

```
