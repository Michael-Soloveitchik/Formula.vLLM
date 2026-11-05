# PR #20839: [Model] Replace Mamba2 RMSNorm Gated with Fused Triton Kernel

- field: gpu_kernels  | area: model_executor_layers
- author: cyang49
- state: MERGED  created: 2025-07-11T19:38:28Z  merged: 2025-07-25T13:49:37Z
- url: https://github.com/vllm-project/vllm/pull/20839
- labels: ready

# Issue

## Problem

The RMSNormGated layer in the Mamba2 mixer used unfused operations: a separate SiLU gating multiply, a dtype cast, and a standalone `ops.rms_norm` call. Each step launched its own kernel and round-tripped intermediate tensors through HBM. On long prompts this layer consumed as much wall-clock time as the entire Mamba2 SSD scan, far more than expected for a normalization op.

## Example

With Bamba-9B-v2 on H100-80GB at 16k input length (batch=1, output-len=1), avg latency was 0.632s before; at 32k it was 1.280s. The author notes RMSNormGated took as long as the mamba2 SSD computation in 16k/32k experiments.

## Use case

Serving long-context Mamba2/SSM hybrid models (e.g. Bamba) where prefill dominates and the gated RMSNorm runs over large token sequences each forward pass.

## Resources

GPU compute plus HBM/memory bandwidth and kernel-launch overhead — multiple separate kernels each read/write full activation tensors instead of one fused pass.

## Type of users

Users running Mamba2-based hybrid models at long sequence lengths on single-GPU (TP=1) configs with ngroups=1.

## Kind of inputs

Long prompts (16k–32k tokens), small batch, float32 intermediate casts for the gate branch.

## Mechanism with the issue

`RMSNormGated.forward_cuda` in `mamba_mixer2.py` computed `y = x * silu(gate.float())`, cast back to x.dtype, then called `ops.rms_norm` into a fresh `torch.empty_like` buffer. Gating, cast, and normalization were three unfused ops, each a separate kernel with full-tensor HBM traffic — no fusion of the gate-multiply into the norm reduction.

# Solution

## Fix in words

Adds new file `ops/layernorm_gated.py` with a Triton fused kernel (`_layer_norm_fwd_1pass_kernel` + `_layer_norm_fwd` + `rms_norm_gated` wrapper), adapted from mamba_ssm. The kernel does the SiLU gate multiply (`x *= z*sigmoid(z)`), RMS variance reduction, normalization, and weight scaling in a single pass per row, keeping intermediates in registers instead of HBM. `forward_cuda` now calls `rms_norm_gated(x, weight, z=gate, norm_before_gate=False)` instead of the three-op silu+cast+`ops.rms_norm` sequence. The TP/ngroups guard changes from `tp_size > 1 or n_groups != 1` to `(n_groups % tp_size) != 0 or n_groups != 1`, still falling back to `forward_native` outside the TP=1/ngroups=1 case. Yields ~8–9% end-to-end latency improvement with unchanged gsm8k quality.

## Author's solution

Replace the current unfused RMSNormGated implementation with a fused Triton kernel taken from the mamba_ssm repo's `layernorm_gated.py`. The author scopes it to TP=1 and ngroups=1 for now, noting broader use cases likely need similar fixes, and validates output quality e2e via lm_eval gsm8k and performance via benchmark_latency.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`
- `vllm/model_executor/layers/mamba/ops/layernorm_gated.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/mamba/mamba_mixer2.py b/vllm/model_executor/layers/mamba/mamba_mixer2.py
index e32b2be4d40e..2c95099e53ad 100644
--- a/vllm/model_executor/layers/mamba/mamba_mixer2.py
+++ b/vllm/model_executor/layers/mamba/mamba_mixer2.py
@@ -24,6 +24,7 @@
     extra_groups_for_head_shards, get_mamba_state_shape)
 from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
     causal_conv1d_fn, causal_conv1d_update)
+from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
 from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
     selective_state_update)
 from vllm.model_executor.layers.mamba.ops.ssd_combined import (
@@ -133,21 +134,15 @@ def forward_cuda(
             return x * nn.functional.silu(gate.to(
                 torch.float32)).to(input_dtype)
 
-        if self.tp_size > 1 or self.n_groups != 1:
+        if (((self.n_groups % self.tp_size) != 0) or self.n_groups != 1):
             return self.forward_native(x, gate)
 
-        from vllm import _custom_ops as ops
-
-        # cast x and gate to float32 before silu
-        out = torch.empty_like(x)
-        y = x * nn.functional.silu(gate.to(torch.float32))
-        ops.rms_norm(
-            out,
-            y.to(x.dtype),
-            self.weight.data,
-            self.variance_epsilon,
-        )
-        return out
+        return rms_norm_gated(x,
+                              self.weight.data,
+                              bias=None,
+                              z=gate,
+                              eps=self.variance_epsilon,
+                              norm_before_gate=False)
 
 
 def mamba_v2_sharded_weight_loader(
diff --git a/vllm/model_executor/layers/mamba/ops/layernorm_gated.py b/vllm/model_executor/layers/mamba/ops/layernorm_gated.py
new file mode 100644
index 000000000000..f3a45ab097c3
--- /dev/null
+++ b/vllm/model_executor/layers/mamba/ops/layernorm_gated.py
@@ -0,0 +1,168 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+# Copyright (c) 2024, Tri Dao.
+# Adapted from https://github.com/state-spaces/mamba/blob/60dadf2e0ee730ac337035d5533de10bc26e4847/mamba_ssm/ops/triton/layernorm_gated.py
+
+import torch
+
+from vllm.triton_utils import tl, triton
+
+
+@triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
+@triton.heuristics({"HAS_Z": lambda args: args["Z"] is not None})
+@triton.jit
+def _layer_norm_fwd_1pass_kernel(
+    X,  # pointer to the input
+    Y,  # pointer to the output
+    W,  # pointer to the weights
+    B,  # pointer to the biases
+    Z,  # pointer to the other branch
+    Mean,  # pointer to the mean
+    Rstd,  # pointer to the 1/std
+    stride_x_row: tl.int64,
+    stride_y_row: tl.int64,
+    stride_z_row: tl.int64,
+    M: tl.int64,  # number of rows in X
+    N: tl.int64,  # number of columns in X
+    eps,  # epsilon to avoid division by zero
+    BLOCK_N: tl.constexpr,
+    HAS_BIAS: tl.constexpr,
+    HAS_Z: tl.constexpr,
+    NORM_BEFORE_GATE: tl.constexpr,
+    IS_RMS_NORM: tl.constexpr,
+):
+    # Map the program id to the row of X and Y it should compute.
+    row = tl.program_id(0)
+    group = tl.program_id(1)
+    X += row * stride_x_row + group * N
+    Y += row * stride_y_row + group * N
+    if HAS_Z:
+        Z += row * stride_z_row + group * N
+    if not IS_RMS_NORM:
+        Mean += group * M
+    Rstd += group * M
+    W += group * N
+    if HAS_BIAS:
+        B += group * N
+    # Compute mean and variance
+    cols = tl.arange(0, BLOCK_N)
+    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
+    if HAS_Z and not NORM_BEFORE_GATE:
+        z = tl.load(Z + cols, mask=cols < N).to(tl.float32)
+        x *= z * tl.sigmoid(z)
+    if not IS_RMS_NORM:
+        mean = tl.sum(x, axis=0) / N
+        tl.store(Mean + row, mean)
+        xbar = tl.where(cols < N, x - mean, 0.)
+        var = tl.sum(xbar * xbar, axis=0) / N
+    else:
+        xbar = tl.where(cols < N, x, 0.)
+        var = tl.sum(xbar * xbar, axis=0) / N
+    rstd = 1 / tl.sqrt(var + eps)
+    tl.store(Rstd + row, rstd)
+    # Normalize and apply linear transformation
+    mask = cols < N
+    w = tl.load(W + cols, mask=mask).to(tl.float32)
+    if HAS_BIAS:
+        b = tl.load(B + cols, mask=mask).to(tl.float32)
+    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
+    y = x_hat * w + b if HAS_BIAS else x_hat * w
+    if HAS_Z and NORM_BEFORE_GATE:
+        z = tl.load(Z + cols, mask=mask).to(tl.float32)
+        y *= z * tl.sigmoid(z)
+    # Write output
+    tl.store(Y + cols, y, mask=mask)
+
+
+def _layer_norm_fwd(x,
+                    weight,
+                    bias,
+                    eps,
+                    z=None,
+                    out=None,
+                    group_size=None,
+                    norm_before_gate=True,
+                    is_rms_norm=False):
+    M, N = x.shape
+    if group_size is None:
+        group_size = N
+    assert N % group_size == 0
+    ngroups = N // group_size
+    assert x.stride(-1) == 1
+    if z is not None:
+        assert z.stride(-1) == 1
+        assert z.shape == (M, N)
+    assert weight.shape == (N, )
+    assert weight.stride(-1) == 1
+    if bias is not None:
+        assert bias.stride(-1) == 1
+        assert bias.shape == (N, )
+    # allocate output
+    if out is not None:
+        assert out.shape == x.shape
+    else:
+        out = torch.empty_like(x)
+    assert out.stride(-1) == 1
+    mean = torch.empty((ngroups * M, ), dtype=torch.float32,
+                       device=x.device) if not is_rms_norm else None
+    rstd = torch.empty((ngroups * M, ), dtype=torch.float32, device=x.device)
+    # Less than 64KB per feature: enqueue fused kernel
+    MAX_FUSED_SIZE = 65536 // x.element_size()
+    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
+    if group_size > BLOCK_N:
+        raise RuntimeError(
+            "This layer norm doesn't support feature dim >= 64KB.")
+    # heuristics for number of warps
+    num_warps = min(max(BLOCK_N // 256, 1), 8)
+    grid = (M, ngroups)
+    with torch.cuda.device(x.device.index):
+        _layer_norm_fwd_1pass_kernel[grid](x,
+                                           out,
+                                           weight,
+                                           bias,
+                                           z,
+                                           mean,
+                                           rstd,
+                                           x.stride(0),
+                                           out.stride(0),
+                                           z.stride(0) if z is not None else 0,
+                                           M,
+                                           group_size,
+                                           eps,
+                                           BLOCK_N=BLOCK_N,
+                                           NORM_BEFORE_GATE=norm_before_gate,
+                                           IS_RMS_NORM=is_rms_norm,
+                                           num_warps=num_warps)
+    return out, mean, rstd
+
+
+def rms_norm_gated(x,
+                   weight,
+                   bias,
+                   z=None,
+                   eps=1e-6,
+                   group_size=None,
+                   norm_before_gate=True):
+    x_shape_og = x.shape
+    # reshape input data into 2D tensor
+    x = x.reshape(-1, x.shape[-1])
+    if x.stride(-1) != 1:
+        x = x.contiguous()
+    if z is not None:
+        assert z.shape == x_shape_og
+        z = z.reshape(-1, z.shape[-1])
+        if z.stride(-1) != 1:
+            z = z.contiguous()
+    weight = weight.contiguous()
+    if bias is not None:
+        bias = bias.contiguous()
+    y, _, _ = _layer_norm_fwd(x,
+                              weight,
+                              bias,
+                              eps,
+                              z=z,
+                              group_size=group_size,
+                              norm_before_gate=norm_before_gate,
+                              is_rms_norm=True)
+
+    return y.reshape(x_shape_og)

```
