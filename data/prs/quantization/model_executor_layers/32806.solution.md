# PR #32806: [torch.compile] Compile `CustomOp.forward_native` for `SiluAndMul` and `QuantFP8` to avoid raw torch ops inside opaque custom ops

- field: quantization  | area: model_executor_layers
- author: ProExpertProg
- state: MERGED  created: 2026-01-21T20:11:26Z  merged: 2026-01-23T03:52:26Z
- url: https://github.com/vllm-project/vllm/pull/32806
- labels: ready, torch.compile, ready-run-all-tests

# Issue

## Problem

When a `CustomOp.forward_native` (e.g. `SiluAndMul`, `QuantFP8`) is called from inside another opaque torch custom op like `fused_moe` or `unified_attention`, the inner op is hidden from model-level `torch.compile`. The native implementation then runs as eager PyTorch ops instead of a fused/compiled graph, incurring per-op kernel-launch overhead and no fusion. This measurably hurts latency for the affected layers.

## Example

The shared expert in Llama4 and DeepSeek runs `SiluAndMul` activation and `QuantFP8` quantization inside the `fused_moe` custom op. Running these as eager native ops gave ~15.8% worse latency on nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 (1.51s vs 1.30s avg) and ~3.7% worse on DeepSeek-R1-NVFP4 (tp=4, expert-parallel).

## Use case

Low-latency / high-throughput serving of MoE models where activation + input quantization execute inside a fused expert custom op.

## Resources

GPU compute and kernel-launch overhead — eager ops launch many small kernels with no fusion; also compile time at startup.

## Type of users

Users serving MoE models (Llama4, DeepSeek) with FP8/NVFP4 quantization, often on multi-GPU TP + expert-parallel setups.

## Kind of inputs

Forward passes routed through `fused_moe` shared experts with FP8/fp4 activations; batch of decode/prefill tokens (benchmarks: max_num_seqs=256, max_model_len=1024).

## Mechanism with the issue

`CustomOp.dispatch_forward` returned `self.forward_native` raw when the op was disabled. Because `fused_moe`/`unified_attention` are opaque torch custom ops, torch.compile cannot see into them, so `forward_native`'s native torch ops execute eagerly op-by-op with no graph capture or fusion.

# Solution

## Fix in words

`CustomOp.__init__` gains a keyword `compile_native` flag threaded into `dispatch_forward`. Before, the disabled-op branch returned raw `self.forward_native`; after, it returns `self.maybe_compile(self.forward_native, enable=compile_native)`. New `maybe_compile` wraps the fn in `torch.compile(dynamic=True, backend=simple_compile_backend)` (skipping when compilation is NONE or backend is `eager`, and nesting inside an already-compiled region is a no-op). `SiluAndMul` and `QuantFP8` default `compile_native=True`; call sites that must stay native (matcher_utils, activation test, rotary embedding) pass `compile_native=False` or switch to keyword args.

## Author's solution

Manually compile `forward_native` for the specific ops that run inside opaque custom ops, so they get a compiled graph instead of eager torch ops. Applied only to `SiluAndMul` and `QuantFP8` to avoid an avalanche of compilation issues, as a temporary measure until these ops move to vLLM IR / `PluggableLayer`. Existing model-level-visible CustomOp invocations are unaffected since nested `torch.compile` is ignored, and the author notes CUDA kernels alone give little benefit (slower than compiled native).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/compile/test_silu_mul_quant_fusion.py`
- `tests/kernels/core/test_activation.py`
- `vllm/compilation/matcher_utils.py`
- `vllm/model_executor/custom_op.py`
- `vllm/model_executor/layers/activation.py`
- `vllm/model_executor/layers/quantization/input_quant_fp8.py`
- `vllm/model_executor/layers/rotary_embedding/common.py`

### Diff
```diff
diff --git a/tests/compile/test_silu_mul_quant_fusion.py b/tests/compile/test_silu_mul_quant_fusion.py
index f8ae83ca350a..dec5ca8de153 100644
--- a/tests/compile/test_silu_mul_quant_fusion.py
+++ b/tests/compile/test_silu_mul_quant_fusion.py
@@ -222,7 +222,7 @@ def test_fusion_silu_and_mul_quant(
     x = torch.rand(num_tokens, hidden_size * 2)
 
     # Reshape pass is needed for the fusion pass to work
-    custom_ops = []
+    custom_ops = ["none"]
     if enable_silu_mul_custom_op:
         custom_ops.append("+silu_and_mul")
     if enable_quant_fp8_custom_op:
@@ -231,6 +231,7 @@ def test_fusion_silu_and_mul_quant(
         compilation_config=CompilationConfig(
             mode=CompilationMode.VLLM_COMPILE,
             custom_ops=custom_ops,
+            backend="eager",  # avoid compilation for SiluAndMul and QuantFP8
             pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
         ),
     )
diff --git a/tests/kernels/core/test_activation.py b/tests/kernels/core/test_activation.py
index 2fa4fd627204..8f28e967aec3 100644
--- a/tests/kernels/core/test_activation.py
+++ b/tests/kernels/core/test_activation.py
@@ -57,7 +57,7 @@ def test_act_and_mul(
     torch.set_default_device(device)
     x = torch.randn(num_tokens, 2 * d, dtype=dtype)
     if activation == "silu_and_mul":
-        layer = SiluAndMul()
+        layer = SiluAndMul(compile_native=False)
         fn = torch.ops._C.silu_and_mul
     if activation == "mul_and_silu":
         layer = MulAndSilu()
diff --git a/vllm/compilation/matcher_utils.py b/vllm/compilation/matcher_utils.py
index 8fd7a46173e9..5e6baf393f39 100644
--- a/vllm/compilation/matcher_utils.py
+++ b/vllm/compilation/matcher_utils.py
@@ -336,6 +336,7 @@ def __init__(
             quant_key.scale.group_shape,
             column_major_scales=has_col_major_scales,
             use_ue8m0=is_e8m0,
+            compile_native=False,
         )
 
     def forward_rocm_aiter(
diff --git a/vllm/model_executor/custom_op.py b/vllm/model_executor/custom_op.py
index 6fe252fa27ee..ee75d627d55d 100644
--- a/vllm/model_executor/custom_op.py
+++ b/vllm/model_executor/custom_op.py
@@ -1,16 +1,15 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-
-
+import torch
 import torch.nn as nn
 
 from vllm.config import get_cached_compilation_config
 from vllm.logger import init_logger
+from vllm.model_executor.utils import maybe_disable_graph_partition
 from vllm.platforms import current_platform
 
 logger = init_logger(__name__)
 
-
 # Dictionary of all custom ops (classes, indexed by registered name).
 # To check if an op with a name is enabled, call .enabled() on the class.
 # Examples:
@@ -118,10 +117,10 @@ def __new__(cls, *args, **kwargs):
             )
         return super().__new__(op_cls_to_instantiate)
 
-    def __init__(self, enforce_enable: bool = False):
+    def __init__(self, *, enforce_enable: bool = False, compile_native: bool = False):
         super().__init__()
         self._enforce_enable = enforce_enable
-        self._forward_method = self.dispatch_forward()
+        self._forward_method = self.dispatch_forward(compile_native=compile_native)
 
     def forward(self, *args, **kwargs):
         return self._forward_method(*args, **kwargs)
@@ -162,7 +161,7 @@ def forward_oot(self, *args, **kwargs):
         # PyTorch-native implementation.
         return self.forward_native(*args, **kwargs)
 
-    def dispatch_forward(self):
+    def dispatch_forward(self, compile_native: bool):
         # NOTE(woosuk): Here we assume that vLLM was built for only one
         # specific backend. Currently, we do not support dynamic dispatching.
         compilation_config = get_cached_compilation_config()
@@ -180,7 +179,9 @@ def dispatch_forward(self):
             compilation_config.disabled_custom_ops.update([self.__class__.name])
 
         if not enabled:
-            return self.forward_native
+            # Compile forward_native to avoid eager torch ops if inside
+            # opaque torch custom op (e.g. fused_moe, unified_attention, etc.)
+            return self.maybe_compile(self.forward_native, enable=compile_native)
 
         if current_platform.is_rocm():
             return self.forward_hip
@@ -195,6 +196,40 @@ def dispatch_forward(self):
         else:
             return self.forward_cuda
 
+    def maybe_compile(self, fn, *, enable: bool = True):
+        """
+        Compile fn if compilation enabled.
+        Useful for CustomOp instances called from within a torch custom op,
+        meaning the forward call is hidden from the model-level torch.compile.
+
+        NOTE: this does not enable fusion across ops, so opaque custom ops
+        should still be unwrapped wherever possible.
+        """
+        # Do not compile if compilation disabled
+        from vllm.config.compilation import CompilationMode
+
+        if not enable:
+            return fn
+
+        # Do not compile if global compilation disabled
+        compilation_config = get_cached_compilation_config()
+        if compilation_config.mode == CompilationMode.NONE:
+            return fn
+
+        # If eager backend is used, do not compile either
+        if compilation_config.backend == "eager":
+            return fn
+
+        # dynamic=True to avoid recompilations
+        return torch.compile(
+            fn,
+            dynamic=True,
+            backend=current_platform.simple_compile_backend,
+            options=maybe_disable_graph_partition(
+                current_platform.simple_compile_backend
+            ),
+        )
+
     @classmethod
     def enabled(cls) -> bool:
         # if no name, then it was not registered
diff --git a/vllm/model_executor/layers/activation.py b/vllm/model_executor/layers/activation.py
index 5e904b9070cc..c8822aed2e3f 100644
--- a/vllm/model_executor/layers/activation.py
+++ b/vllm/model_executor/layers/activation.py
@@ -75,8 +75,8 @@ class SiluAndMul(CustomOp):
 
     # --8<-- [end:silu_and_mul]
 
-    def __init__(self):
-        super().__init__()
+    def __init__(self, *, compile_native: bool = True):
+        super().__init__(compile_native=compile_native)
         if current_platform.is_cuda_alike():
             self.op = torch.ops._C.silu_and_mul
         elif current_platform.is_xpu():
diff --git a/vllm/model_executor/layers/quantization/input_quant_fp8.py b/vllm/model_executor/layers/quantization/input_quant_fp8.py
index c1a901c37a0b..36c61ce7aeeb 100644
--- a/vllm/model_executor/layers/quantization/input_quant_fp8.py
+++ b/vllm/model_executor/layers/quantization/input_quant_fp8.py
@@ -36,6 +36,7 @@ def __init__(
         num_token_padding: int | None = None,
         column_major_scales: bool = False,
         use_ue8m0: bool | None = None,  # for Torch compile
+        compile_native: bool = True,
     ):
         """
         :param static: static or dynamic quantization
@@ -45,8 +46,9 @@ def __init__(
             size
         :param column_major_scales: For group quantization, output scales in
             column major format
+        :param compile_native: Manually compile forward_native if compile mode > None
         """
-        super().__init__()
+        super().__init__(compile_native=compile_native)
         self.static = static
         self.group_shape = group_shape
         self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
diff --git a/vllm/model_executor/layers/rotary_embedding/common.py b/vllm/model_executor/layers/rotary_embedding/common.py
index 34de1da561f5..2cca86b05b35 100644
--- a/vllm/model_executor/layers/rotary_embedding/common.py
+++ b/vllm/model_executor/layers/rotary_embedding/common.py
@@ -129,7 +129,7 @@ def __init__(
         is_neox_style: bool = True,
         enable_fp32_compute: bool = False,
     ) -> None:
-        super().__init__(enforce_enable)
+        super().__init__(enforce_enable=enforce_enable)
         self.is_neox_style = is_neox_style
         self.enable_fp32_compute = enable_fp32_compute
 

```
