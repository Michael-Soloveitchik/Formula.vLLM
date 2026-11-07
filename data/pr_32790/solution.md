# PR #32790: [MoE] Enable Shared/Routed Overlap For Latent MoE (Nemotron-H)

- field: parallelism  | area: model_executor_layers
- author: danielafrimi
- state: MERGED  created: 2026-01-21T17:07:39Z  merged: 2026-02-02T14:18:51Z
- url: https://github.com/vllm-project/vllm/pull/32790
- labels: ready

# Issue

## Problem

Latent MoE compresses hidden_states before expert routing, so routed experts consume a small `[S, moe_latent_size]` tensor while shared experts need the original `[S, hidden_size]` tensor. The existing shared/routed CUDA-stream overlap path cloned and fed the SAME tensor to both branches, so it could not serve the two different dimensions. This forced shared and routed experts to run sequentially on the main stream, leaving the auxiliary CUDA stream idle.

## Example

Nemotron-H: fc1_latent_proj maps `[S, 4096]` → `[S, 1024]`. Routed experts need `[S, 1024]`, shared experts need `[S, 4096]`. Single shared clone could not supply both, so streaming stayed off.

## Use case

Serving latent-MoE decoder models (Nemotron-H) where each layer runs both shared and routed experts per token during prefill and decode.

## Resources

GPU compute / SM occupancy and kernel-launch scheduling — one CUDA stream sits idle while the other works, wasting parallel execution capacity.

## Type of users

Users serving latent/compressed MoE models with shared experts, typically multi-GPU TP (H100 TP=4, B200 TP=8 in benchmarks).

## Kind of inputs

Any batch/sequence that passes through the MoE layer; small-batch decode (batch 4) showed the clearest gain (+3.4%), large batches neutral.

## Mechanism with the issue

`FusedMoE.forward_native` reduced input dim before routing and `_maybe_setup_shared_experts_stream` cloned `hidden_states` for the aux stream. Both read the single post-transform tensor, so shared experts could not get the original dimension — blocking the parallel-stream path for latent MoE.

# Solution

## Fix in words

Diff adds an `apply_routed_input_transform()` hook (no-op in base `FusedMoE`, overridden by `SharedFusedMoE` with `routed_input_transform`) and saves the pre-transform tensor. `forward_native` now records `original_hidden_states` before calling the transform, tracks separate `original_hidden_dim` and `transformed_hidden_dim`, and slices shared output to original dim and routed output to transformed dim. A `_set_shared_experts_input` context manager plus `_get_shared_experts_input` carry the original tensor through `forward_impl`; the stream-setup clone and the sequential shared-expert call now pull from `_get_shared_experts_input` instead of the transformed `hidden_states`. The `moe_forward_shared` custom op gains a `shared_experts_input` arg so torch.compile paths also set the var. Effect: shared experts get `[S, 4096]`, routed get `[S, 1024]`, enabling parallel CUDA-stream overlap.

## Author's solution

Save the original `hidden_states` before applying the latent transform so shared experts use the original `[S, hidden_size]` while routed experts use the transformed `[S, latent_size]`. Implemented via an `apply_routed_input_transform()` hook in the `FusedMoE` base class, a `routed_input_transform` parameter overriding that hook in `SharedFusedMoE`, and wiring `routed_input_transform=fc1_latent_proj` from `NemotronHMoE` into `SharedFusedMoE`. This lets shared and routed experts run in parallel on separate CUDA streams with a sync before combine.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/kernels/moe/test_shared_fused_moe_routed_transform.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`
- `vllm/model_executor/models/nemotron_h.py`

### Diff
```diff
diff --git a/tests/kernels/moe/test_shared_fused_moe_routed_transform.py b/tests/kernels/moe/test_shared_fused_moe_routed_transform.py
new file mode 100644
index 000000000000..3be1d9974795
--- /dev/null
+++ b/tests/kernels/moe/test_shared_fused_moe_routed_transform.py
@@ -0,0 +1,162 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Tests for SharedFusedMoE with routed_input_transform.
+
+Verifies that applying routed_input_transform inside SharedFusedMoE
+produces the same results as applying the transform manually outside.
+"""
+
+import pytest
+import torch
+import torch.nn as nn
+
+from vllm.config import VllmConfig, set_current_vllm_config
+from vllm.forward_context import set_forward_context
+from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
+
+
+class SimpleLinear(nn.Module):
+    """A simple linear transform mimicking latent projection in latent MoE."""
+
+    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype):
+        super().__init__()
+        self.weight = nn.Parameter(
+            torch.randn(out_features, in_features, device="cuda", dtype=dtype) / 10
+        )
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        return nn.functional.linear(x, self.weight)
+
+
+class SimpleSharedExperts(nn.Module):
+    """A simple 2-layer MLP mimicking shared experts."""
+
+    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype):
+        super().__init__()
+        self.up = nn.Linear(
+            hidden_size, intermediate_size * 2, bias=False, device="cuda", dtype=dtype
+        )
+        self.down = nn.Linear(
+            intermediate_size, hidden_size, bias=False, device="cuda", dtype=dtype
+        )
+        with torch.no_grad():
+            self.up.weight.div_(10)
+            self.down.weight.div_(10)
+
+    def forward(self, x: torch.Tensor) -> torch.Tensor:
+        gate_up = self.up(x)
+        gate, up = gate_up.chunk(2, dim=-1)
+        return self.down(nn.functional.silu(gate) * up)
+
+
+@pytest.fixture(autouse=True)
+def setup_cuda():
+    if not torch.cuda.is_available():
+        pytest.skip("CUDA not available")
+    torch.set_default_device("cuda")
+
+
+@pytest.mark.parametrize("num_tokens", [1, 32])
+@pytest.mark.parametrize("hidden_size,latent_size", [(256, 128), (128, 64)])
+@pytest.mark.parametrize("dtype", [torch.bfloat16])
+def test_routed_input_transform_inside_vs_outside(
+    num_tokens: int,
+    hidden_size: int,
+    latent_size: int,
+    dtype: torch.dtype,
+    dist_init,
+    workspace_init,
+):
+    """Compare SharedFusedMoE with transform inside vs manually applying outside.
+    Method A (inside): SharedFusedMoE with routed_input_transform
+    Method B (outside): Manually transform, then SharedFusedMoE without transform
+    """
+    torch.manual_seed(42)
+
+    num_experts = 8
+    top_k = 2
+    intermediate_size = hidden_size * 2
+
+    vllm_config = VllmConfig()
+    vllm_config.compilation_config.static_forward_context = dict()
+
+    shared_experts = SimpleSharedExperts(hidden_size, intermediate_size, dtype)
+    routed_transform = SimpleLinear(hidden_size, latent_size, dtype)
+
+    with set_current_vllm_config(vllm_config):
+        # Method A: SharedFusedMoE WITH routed_input_transform
+        moe_with_transform = SharedFusedMoE(
+            shared_experts=shared_experts,
+            routed_input_transform=routed_transform,
+            num_experts=num_experts,
+            top_k=top_k,
+            hidden_size=latent_size,
+            intermediate_size=intermediate_size,
+            reduce_results=False,
+            renormalize=True,
+            params_dtype=dtype,
+            tp_size=1,
+            dp_size=1,
+            pcp_size=1,
+            prefix="moe_with_transform",
+        )
+
+        # Method B: SharedFusedMoE WITHOUT routed_input_transform
+        # Note: shared_experts=None because when transform is done outside,
+        moe_without_transform = SharedFusedMoE(
+            shared_experts=None,
+            routed_input_transform=None,
+            num_experts=num_experts,
+            top_k=top_k,
+            hidden_size=latent_size,
+            intermediate_size=intermediate_size,
+            reduce_results=False,
+            renormalize=True,
+            params_dtype=dtype,
+            tp_size=1,
+            dp_size=1,
+            pcp_size=1,
+            prefix="moe_without_transform",
+        )
+
+        with torch.no_grad():
+            moe_without_transform.w13_weight.copy_(moe_with_transform.w13_weight)
+            moe_without_transform.w2_weight.copy_(moe_with_transform.w2_weight)
+
+        moe_with_transform.quant_method.process_weights_after_loading(
+            moe_with_transform
+        )
+        moe_without_transform.quant_method.process_weights_after_loading(
+            moe_without_transform
+        )
+
+        hidden_states = torch.randn(num_tokens, hidden_size, device="cuda", dtype=dtype)
+        router_logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype)
+
+        with set_forward_context(None, vllm_config, num_tokens=num_tokens):
+            shared_out_A, routed_out_A = moe_with_transform(
+                hidden_states, router_logits
+            )
+
+            transformed_hidden = routed_transform(hidden_states)
+            shared_out_B, routed_out_B = moe_without_transform(
+                transformed_hidden, router_logits
+            )
+
+        torch.testing.assert_close(
+            routed_out_A,
+            routed_out_B,
+            atol=1e-3,
+            rtol=1e-3,
+            msg="Routed output should match: transform inside vs outside",
+        )
+
+        expected_shared_out = shared_experts(hidden_states)
+
+        torch.testing.assert_close(
+            shared_out_A,
+            expected_shared_out,
+            atol=1e-3,
+            rtol=1e-3,
+        )
diff --git a/vllm/model_executor/layers/fused_moe/layer.py b/vllm/model_executor/layers/fused_moe/layer.py
index 5fe4bce7a4fc..b092cf6cf0e8 100755
--- a/vllm/model_executor/layers/fused_moe/layer.py
+++ b/vllm/model_executor/layers/fused_moe/layer.py
@@ -1,8 +1,8 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
-from collections.abc import Callable, Iterable
-from contextlib import nullcontext
+from collections.abc import Callable, Generator, Iterable
+from contextlib import contextmanager, nullcontext
 from enum import Enum
 from typing import Literal, cast, get_args, overload
 
@@ -351,6 +351,10 @@ def __init__(
                     "Enabled separate cuda stream for MoE shared_experts", scope="local"
                 )
 
+        # For latent MoE: stores original hidden_states before routed_input_transform
+        # so shared_experts can use it for cloning (they need original dimension)
+        self._shared_experts_input: torch.Tensor | None = None
+
         if params_dtype is None:
             params_dtype = torch.get_default_dtype()
         self.params_dtype = params_dtype
@@ -664,6 +668,39 @@ def layer_id(self):
     def gate(self) -> torch.nn.Module | None:
         return None
 
+    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Hook to transform hidden_states before passing to routed experts.
+        For latent MoE: transforms [S, hidden_size] → [S, moe_latent_size].
+        The original hidden_states is saved in _shared_experts_input so
+        shared_experts still receive the original [S, hidden_size].
+
+        Override in subclasses (e.g., SharedFusedMoE) for latent MoE.
+        """
+        return hidden_states
+
+    @contextmanager
+    def _set_shared_experts_input(
+        self, value: torch.Tensor | None
+    ) -> Generator[None, None, None]:
+        """Context manager to safely set/clear _shared_experts_input."""
+        self._shared_experts_input = value
+        try:
+            yield
+        finally:
+            self._shared_experts_input = None
+
+    def _get_shared_experts_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Get input for shared experts.
+
+        For latent MoE: shared_experts need original [S, hidden_size],
+        not the transformed [S, latent_size] used by routed experts.
+        """
+        return (
+            self._shared_experts_input
+            if self._shared_experts_input is not None
+            else hidden_states
+        )
+
     @property
     def tp_size(self):
         return self.moe_parallel_config.tp_size
@@ -855,9 +892,11 @@ def _maybe_setup_shared_experts_stream(
         if use_shared_experts_stream:
             assert self.shared_experts_stream is not None
 
+            shared_experts_input = self._get_shared_experts_input(hidden_states)
+
             # Clone BEFORE switching streams to avoid race condition
             # where routed_expert kernel may mutate hidden_states.
-            hidden_states_clone = hidden_states.clone()
+            hidden_states_clone = shared_experts_input.clone()
 
             # Record that the clone will be used by shared_experts_stream
             # to avoid gc issue from deallocation of hidden_states_clone
@@ -1537,11 +1576,20 @@ def forward_native(
         hidden_states: torch.Tensor,
         router_logits: torch.Tensor,
     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
-        og_hidden_states = hidden_states.shape[-1]
-        if self.hidden_size != og_hidden_states:
+        # For latent MoE: save ORIGINAL hidden_states before transform
+        # (shared_experts need original dimension, routed experts use transformed)
+        original_hidden_states = hidden_states
+        original_hidden_dim = hidden_states.shape[-1]
+
+        # Apply transform for routed experts (e.g., latent projection for latent MoE)
+        hidden_states = self.apply_routed_input_transform(hidden_states)
+
+        # This is the dimension after transform (for routed expert output slicing)
+        transformed_hidden_dim = hidden_states.shape[-1]
+        if self.hidden_size != transformed_hidden_dim:
             hidden_states = F.pad(
                 hidden_states,
-                (0, self.hidden_size - og_hidden_states),
+                (0, self.hidden_size - transformed_hidden_dim),
                 mode="constant",
                 value=0.0,
             )
@@ -1576,22 +1624,31 @@ def encode_layer_name() -> str:
                 fused_output = torch.ops.vllm.moe_forward(
                     hidden_states, router_logits, encode_layer_name()
                 )
-            return reduce_output(fused_output)[..., :og_hidden_states]
+            return reduce_output(fused_output)[..., :transformed_hidden_dim]
         else:
             if current_platform.is_tpu() or current_platform.is_cpu():
                 # TODO: Once the OOM issue for the TPU backend is resolved, we
                 # will switch to using the moe_forward custom op.
                 # Note: CPU doesn't require wrapped forward_impl.
-                shared_output, fused_output = self.forward_impl(
-                    hidden_states, router_logits
-                )
+                with self._set_shared_experts_input(original_hidden_states):
+                    shared_output, fused_output = self.forward_impl(
+                        hidden_states, router_logits
+                    )
             else:
+                # Custom op handles setting/clearing _shared_experts_input internally
+                # We pass original tensor for shared experts (not transformed)
                 shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
-                    hidden_states, router_logits, encode_layer_name()
+                    hidden_states,
+                    router_logits,
+                    encode_layer_name(),
+                    original_hidden_states,
                 )
+
+            # shared_output uses original dimension (before transform)
+            # fused_output uses transformed dimension (after transform)
             return (
-                reduce_output(shared_output)[..., :og_hidden_states],
-                reduce_output(fused_output)[..., :og_hidden_states],
+                reduce_output(shared_output)[..., :original_hidden_dim],
+                reduce_output(fused_output)[..., :transformed_hidden_dim],
             )
 
     @property
@@ -1831,7 +1888,8 @@ def forward_impl(
             # because matrix multiply maybe modify the hidden_states.
             if has_separate_shared_experts and not use_shared_experts_stream:
                 assert self.shared_experts is not None
-                shared_output = self.shared_experts(hidden_states)
+                shared_input = self._get_shared_experts_input(hidden_states)
+                shared_output = self.shared_experts(shared_input)
 
             # NOTE: Similar with DP, PCP also needs dispatch and combine. For
             # simplicity, AgRsAll2All was added separately for PCP here. Maybe
@@ -2021,19 +2079,34 @@ def moe_forward_shared(
     hidden_states: torch.Tensor,
     router_logits: torch.Tensor,
     layer_name: str,
+    shared_experts_input: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
     self = get_layer_from_name(layer_name)
     assert self.shared_experts is not None
-    return self.forward_impl(hidden_states, router_logits)
+
+    # Set here because torch.compile skips forward_native() setup code
+    # and calls this op directly. forward_impl() reads from this var.
+    with self._set_shared_experts_input(shared_experts_input):
+        return self.forward_impl(hidden_states, router_logits)
 
 
 def moe_forward_shared_fake(
     hidden_states: torch.Tensor,
     router_logits: torch.Tensor,
     layer_name: str,
+    shared_experts_input: torch.Tensor | None = None,
 ) -> tuple[torch.Tensor, torch.Tensor]:
-    shared_out = torch.empty_like(hidden_states)
+    # Output shapes:
+    # - fused_out: same as hidden_states (routed experts use transformed size)
+    # - shared_out: same as shared_experts_input if provided, else same as hidden_states
+    # (For latent MoE: shared experts use original hidden_size, not latent size)
     fused_out = torch.empty_like(hidden_states)
+
+    if shared_experts_input is not None:
+        shared_out = torch.empty_like(shared_experts_input)
+    else:
+        shared_out = torch.empty_like(hidden_states)
+
     return shared_out, fused_out
 
 
diff --git a/vllm/model_executor/layers/fused_moe/shared_fused_moe.py b/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
index 909a977b84ec..cb601af700ba 100644
--- a/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/shared_fused_moe.py
@@ -23,10 +23,12 @@ def __init__(
         shared_experts: torch.nn.Module | None,
         gate: torch.nn.Module | None = None,
         use_overlapped: bool = True,
+        routed_input_transform: torch.nn.Module | None = None,
         **kwargs,
     ):
         super().__init__(**kwargs)
         self._shared_experts = shared_experts
+        self._routed_input_transform = routed_input_transform
 
         # Disable shared expert overlap if:
         #   - we are using eplb with non-default backend, because of correctness issues
@@ -56,6 +58,26 @@ def gate(self) -> torch.nn.Module | None:
     def is_internal_router(self) -> bool:
         return self.gate is not None
 
+    def apply_routed_input_transform(self, hidden_states: torch.Tensor) -> torch.Tensor:
+        """Apply transform for routed experts (e.g., latent projection).
+
+        This is called by FusedMoE.forward_native. The original hidden_states
+        is saved separately so shared experts get [S, hidden_size] while
+        routed experts get the transformed [S, moe_latent_size].
+
+        TODO: For latent MoE bandwidth optimization, fc2_latent_proj could be
+        moved inside SharedFusedMoE to all-reduce on the smaller latent
+        dimension.
+        """
+        if self._routed_input_transform is not None:
+            result = self._routed_input_transform(hidden_states)
+            # ReplicatedLinear returns (output, extra_bias) tuple.
+            # We only need the output tensor; extra_bias is not used here.
+            if isinstance(result, tuple):
+                return result[0]
+            return result
+        return hidden_states
+
     def forward(
         self,
         hidden_states: torch.Tensor,
diff --git a/vllm/model_executor/models/nemotron_h.py b/vllm/model_executor/models/nemotron_h.py
index 4a1732ab2a59..a935071fc6fe 100644
--- a/vllm/model_executor/models/nemotron_h.py
+++ b/vllm/model_executor/models/nemotron_h.py
@@ -188,10 +188,29 @@ def __init__(
                 prefix=f"{prefix}.shared_experts",
             )
 
+        if self.use_latent_moe:
+            self.fc1_latent_proj = ReplicatedLinear(
+                input_size=config.hidden_size,
+                output_size=self.moe_hidden_size,
+                bias=config.mlp_bias,
+                quant_config=quant_config,
+                disable_tp=self.is_sequence_parallel,
+                prefix=f"{prefix}.fc1_latent_proj",
+            )
+            self.fc2_latent_proj = ReplicatedLinear(
+                input_size=self.moe_hidden_size,
+                output_size=config.hidden_size,
+                bias=config.mlp_bias,
+                quant_config=quant_config,
+                disable_tp=self.is_sequence_parallel,
+                prefix=f"{prefix}.fc2_latent_proj",
+            )
+        else:
+            self.fc1_latent_proj = None
+            self.fc2_latent_proj = None
+
         self.experts = SharedFusedMoE(
-            # TODO: make it possible for shared experts to have
-            # different input in SharedFusedMoE
-            shared_experts=self.shared_experts if not self.use_latent_moe else None,
+            shared_experts=self.shared_experts,
             num_experts=config.n_routed_experts,
             top_k=config.num_experts_per_tok,
             hidden_size=self.moe_hidden_size,
@@ -211,30 +230,9 @@ def __init__(
             num_redundant_experts=self.n_redundant_experts,
             is_sequence_parallel=self.is_sequence_parallel,
             router_logits_dtype=router_logits_dtype,
+            routed_input_transform=self.fc1_latent_proj,
         )
 
-        if self.use_latent_moe:
-            self.fc1_latent_proj = ReplicatedLinear(
-                input_size=config.hidden_size,
-                output_size=self.moe_hidden_size,
-                bias=config.mlp_bias,
-                quant_config=quant_config,
-                disable_tp=self.is_sequence_parallel,
-                prefix=f"{prefix}.fc1_latent_proj",
-            )
-            self.fc2_latent_proj = ReplicatedLinear(
-                input_size=self.moe_hidden_size,
-                output_size=config.hidden_size,
-                bias=config.mlp_bias,
-                quant_config=quant_config,
-                disable_tp=self.is_sequence_parallel,
-                prefix=f"{prefix}.fc2_latent_proj",
-            )
-
-        else:
-            self.fc1_latent_proj = None
-            self.fc2_latent_proj = None
-
     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
         num_tokens, hidden_dim = hidden_states.shape
         hidden_states = hidden_states.view(-1, hidden_dim)
@@ -244,38 +242,28 @@ def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
 
         # router_logits: (num_tokens, n_experts)
         router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))
-        shared_output = None
-        if self.use_latent_moe:
-            if self.shared_experts is not None:
-                shared_output = self.shared_experts(hidden_states)
-            hidden_states, _ = self.fc1_latent_proj(hidden_states)
 
-        fused_moe_out = self.experts(
+        # SharedFusedMoE handles:
+        #   - shared experts (with original hidden_states)
+        #   - routed_input_transform (fc1_latent_proj) for latent MoE
+        #   - multistream parallelism between shared and routed experts
+        shared_output, final_hidden_states = self.experts(
             hidden_states=hidden_states, router_logits=router_logits
         )
 
-        if self.use_latent_moe:
-            _, final_hidden_states = fused_moe_out
-        else:
-            shared_output, final_hidden_states = fused_moe_out
-
         # Fix FP16 overflow
         # See DeepseekV2DecoderLayer for more details.
         if hidden_states.dtype != torch.float16:
             final_hidden_states *= self.routed_scaling_factor
         elif self.shared_experts is not None:
-            assert shared_output is not None
             shared_output *= 1.0 / self.routed_scaling_factor
 
-        # TODO: currently latent up_proj is done before all-reduce for simplicity.
-        #  if and when shared experts will be part of SharedFusedMoE,
-        #  we should do the up_proj after all-reduce,
-        #  to have the all-reduce in the smaller latent dimension.
+        # TODO: See SharedFusedMoE.apply_routed_input_transform
+        # for bandwidth optimization
         if self.use_latent_moe:
             final_hidden_states, _ = self.fc2_latent_proj(final_hidden_states)
 
         if self.shared_experts is not None:
-            assert shared_output is not None
             final_hidden_states += shared_output
 
         if self.is_sequence_parallel:

```
