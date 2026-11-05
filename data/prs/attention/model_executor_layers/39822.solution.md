# PR #39822: [Hybrid] Warmup Mamba2 SSD kernel

- field: attention  | area: model_executor_layers
- author: tdoublep
- state: MERGED  created: 2026-04-14T19:12:41Z  merged: 2026-05-12T12:46:23Z
- url: https://github.com/vllm-project/vllm/pull/39822
- labels: ready

# Issue

## Problem

Triton's auto-tuner for the Mamba2 SSD kernels (`mamba_chunk_scan_combined_varlen`) runs lazily, deferred until the very first real inference request. Autotuning plus JIT compilation of these kernels happens on-demand, so the first request absorbs the full cost of tuning 58 kernel configurations. This produces a large one-time latency spike the first user request pays, even though the model already finished loading.

## Example

On a single H100 80GB running `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (Mamba2 hybrid) with `max_model_len=512` and a cold Triton cache, the first request took 31.343s while subsequent requests averaged 0.083s -- a 378x ratio, with 58 autotuning events firing after model load.

## Use case

Serving a Mamba2 / hybrid SSM model where the first inference request after server startup must be responsive (SLA, health checks, latency-sensitive clients).

## Resources

Kernel-launch/compile time: Triton autotuner CPU-side search plus GPU JIT compilation. Also GPU HBM, since autotuning fires after SSM cache allocation when free memory is tighter, risking OOM.

## Type of users

Users serving Mamba2-based or hybrid (attention+SSM) models on GPU, especially anyone cold-starting a server and routing live traffic to it immediately.

## Kind of inputs

The very first forward pass carrying real chunked-prefill shapes; autotune cache keys depend on tensor dtypes and head dimensions (not sequence length or batch), so any first request triggers the full tuning cost once.

## Mechanism with the issue

`MambaMixer2.conv_ssm_forward` has a profile-run branch (`attn_metadata is None`) during vLLM's V1 profiling, but previously it only did a tensor contiguity touch and never exercised the SSD kernels. Because the real kernel call was never made during profiling, Triton's autotuner for `mamba_chunk_scan_combined_varlen` stayed cold and fired on the first inference request instead. The two `HAS_INITSTATES` constexpr variants compile to separate binaries, so both the no-`initial_states` and with-`initial_states` paths needed tuning.

# Solution

## Fix in words

Adds a `_warmup_ssd_kernels()` method to `MambaMixer2` that runs a minimal `mamba_chunk_scan_combined_varlen` forward with dummy tensors, invoked from the `attn_metadata is None` profile-run branch of `conv_ssm_forward` -- before SSM cache allocation while GPU memory is plentiful. It loops over `use_initial_states in (False, True)` to cover both `HAS_INITSTATES` constexpr code paths, and uses `get_state_dtype()`'s `ssm_state_dtype` plus a single `chunk_size`-length shape so the dummy call matches Triton's dtype/head-dim cache keys. A per-instance `_ssd_kernels_warmed_up` flag guards one warmup per layer (subsequent layers hit the Triton cache), warmup failures are caught and logged rather than fatal, and `torch.accelerator.empty_cache()` frees the dummy tensors. Supporting changes make `get_mamba_chunk_size()` return `int` (non-optional) with a 2048 default and convert `gdn_linear_attn`'s `hasattr`-based warmup guard to an explicit boolean flag; the effect shifts ~47s of autotuning into model-load time and cuts first-request latency from 31.3s to 2.9s (91%).

## Author's solution

The author adds a `_warmup_ssd_kernels()` method to `MambaMixer2` that triggers Triton auto-tuning during vLLM's V1 profile phase (before SSM cache allocation), shifting the cost into server startup. It runs a minimal `mamba_chunk_scan_combined_varlen` forward with dummy tensors, covers both `HAS_INITSTATES` constexpr paths (with and without `initial_states`), and uses correct dtypes including `ssm_state_dtype` to match Triton's cache keys. Each layer warms up once and later layers hit the Triton cache; a residual ~2.9s first-request cost is suspected to be Triton JIT compilation, left as follow-up.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/config/model.py`
- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`

### Diff
```diff
diff --git a/vllm/config/model.py b/vllm/config/model.py
index 22a1f6a42981..193f48a545cc 100644
--- a/vllm/config/model.py
+++ b/vllm/config/model.py
@@ -1345,7 +1345,7 @@ def get_num_layers_by_block_type(
                 )
             raise AssertionError(f"Unsupported block type: {block_type}")
 
-    def get_mamba_chunk_size(self) -> int | None:
+    def get_mamba_chunk_size(self) -> int:
         """
         Returns the mamba chunk size if it exists
         """
@@ -1356,7 +1356,7 @@ def get_mamba_chunk_size(self) -> int | None:
             chunk_size = getattr(self.hf_text_config, "chunk_size", None)
 
         # Since Mamba1 does not have a chunk notion
-        # we use a default chunk size of 1024.
+        # we use a default chunk size of 2048.
         if chunk_size is None:
             chunk_size = 2048
 
diff --git a/vllm/model_executor/layers/mamba/gdn_linear_attn.py b/vllm/model_executor/layers/mamba/gdn_linear_attn.py
index 4493ee783630..011a18ec7b87 100644
--- a/vllm/model_executor/layers/mamba/gdn_linear_attn.py
+++ b/vllm/model_executor/layers/mamba/gdn_linear_attn.py
@@ -397,6 +397,7 @@ def __init__(
         )
 
         self.chunk_gated_delta_rule = ChunkGatedDeltaRule()
+        self._prefill_kernels_warmed_up = False
         self.enable_packed_recurrent_decode = (
             envs.VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE
         )
@@ -905,7 +906,7 @@ def _warmup_prefill_kernels(self, qkv_or_qkvz: torch.Tensor, v_dim: int) -> None
         which has fixed kernel parameters (no autotuning), so only the
         prefill (chunked) path needs warming up.
         """
-        if hasattr(self, "_prefill_kernels_warmed_up"):
+        if self._prefill_kernels_warmed_up:
             return
         self._prefill_kernels_warmed_up = True
 
diff --git a/vllm/model_executor/layers/mamba/mamba_mixer2.py b/vllm/model_executor/layers/mamba/mamba_mixer2.py
index 05cb9aadfe79..674134f373ea 100644
--- a/vllm/model_executor/layers/mamba/mamba_mixer2.py
+++ b/vllm/model_executor/layers/mamba/mamba_mixer2.py
@@ -14,6 +14,7 @@
     tensor_model_parallel_all_reduce,
 )
 from vllm.forward_context import ForwardContext, get_forward_context
+from vllm.logger import init_logger
 from vllm.model_executor.custom_op import CustomOp, PluggableLayer
 from vllm.model_executor.layers.linear import (
     ColumnParallelLinear,
@@ -54,6 +55,8 @@
 from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
 from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
 
+logger = init_logger(__name__)
+
 # Added by the IBM Team, 2024
 
 
@@ -473,6 +476,8 @@ def __init__(
             intermediate_size, n_groups, self.use_rms_norm, eps=rms_norm_eps
         )
 
+        self._ssd_kernels_warmed_up = False
+
         # - get hidden_states, B and C after depthwise convolution.
         self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
             hidden_states_B_C,
@@ -557,6 +562,104 @@ def forward(
 
         return output
 
+    def _warmup_ssd_kernels(self, projected_states: torch.Tensor) -> None:
+        """Run a minimal SSD forward pass to trigger Triton autotuning
+        while GPU memory is still plentiful (before SSM cache allocation).
+        """
+        if self._ssd_kernels_warmed_up:
+            return
+        self._ssd_kernels_warmed_up = True
+        logger.info_once("Warming up Mamba2 SSD Triton kernels...")
+
+        device = projected_states.device
+        dtype = projected_states.dtype
+
+        nheads = self.num_heads // self.tp_size
+        ngroups = self.n_groups // self.tp_size
+        headdim = self.head_dim
+        dstate = self.ssm_state_size
+
+        if self.model_config is None:
+            return
+        chunk_size = self.model_config.get_mamba_chunk_size()
+
+        # Triton's autotuner includes tensor dtypes in its cache key,
+        # so state_dtype must match what real inference uses.
+        _, ssm_state_dtype = self.get_state_dtype()
+
+        # SSD kernel autotune keys depend on dtype and head dimensions,
+        # not on sequence length or batch size, so a single shape suffices.
+        seqlen = chunk_size
+        batch = 1
+        nchunks = seqlen // chunk_size  # = 1
+
+        x = torch.randn(seqlen, nheads, headdim, device=device, dtype=dtype)
+        dt = torch.randn(seqlen, nheads, device=device, dtype=dtype)
+        B = torch.randn(seqlen, ngroups, dstate, device=device, dtype=dtype)
+        C = torch.randn(seqlen, ngroups, dstate, device=device, dtype=dtype)
+        cu_seqlens = torch.tensor([0, seqlen], device=device, dtype=torch.int32)
+        cu_chunk_seqlens = torch.tensor(
+            [i * chunk_size for i in range(nchunks + 1)],
+            device=device,
+            dtype=torch.int32,
+        )
+        last_chunk_indices = torch.tensor(
+            [nchunks - 1], device=device, dtype=torch.int32
+        )
+        seq_idx = torch.zeros(nchunks, device=device, dtype=torch.int32)
+        out = torch.empty(seqlen, nheads, headdim, device=device, dtype=dtype)
+
+        # Two kernels (_state_passing_fwd, _chunk_scan_fwd) use
+        # HAS_INITSTATES as a constexpr, producing separate compiled
+        # binaries. Warm up both code paths so neither triggers
+        # JIT compilation during inference.
+        for use_initial_states in (False, True):
+            initial_states = (
+                torch.randn(
+                    batch,
+                    nheads,
+                    headdim,
+                    dstate,
+                    device=device,
+                    dtype=ssm_state_dtype,
+                )
+                if use_initial_states
+                else None
+            )
+            try:
+                mamba_chunk_scan_combined_varlen(
+                    x=x,
+                    dt=dt,
+                    A=self.A,
+                    B=B,
+                    C=C,
+                    chunk_size=chunk_size,
+                    cu_seqlens=cu_seqlens,
+                    cu_chunk_seqlens=cu_chunk_seqlens,
+                    last_chunk_indices=last_chunk_indices,
+                    seq_idx=seq_idx,
+                    out=out,
+                    D=self.D,
+                    z=None,
+                    dt_bias=self.dt_bias,
+                    initial_states=initial_states,
+                    dt_softplus=True,
+                    dt_limit=(0.0, float("inf")),
+                    state_dtype=ssm_state_dtype,
+                )
+            except Exception:
+                logger.warning(
+                    "Mamba2 SSD kernel warmup failed for layer %s "
+                    "(initial_states=%s). First inference may experience "
+                    "latency spike or OOM due to autotuner.",
+                    self.prefix,
+                    use_initial_states,
+                    exc_info=True,
+                )
+
+        logger.debug("Mamba2 SSD kernel warmup completed for layer %s", self.prefix)
+        torch.accelerator.empty_cache()
+
     def conv_ssm_forward(
         self,
         projected_states: torch.Tensor,
@@ -608,7 +711,9 @@ def conv_ssm_forward(
             num_decode_tokens = attn_metadata.num_decode_tokens
 
         if attn_metadata is None:
-            # profile run
+            # V1 profile run -- warm up SSD kernels so that autotuning
+            # completes before SSM cache allocation.
+            self._warmup_ssd_kernels(projected_states)
             hidden_states_B_C = (
                 hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)
             ).contiguous()

```
