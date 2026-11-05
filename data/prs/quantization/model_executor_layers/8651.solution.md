# PR #8651: [Core] Default to using per_token quantization for fp8 when cutlass is supported.

- field: quantization  | area: model_executor_layers
- author: elfiegg
- state: MERGED  created: 2024-09-20T04:56:02Z  merged: 2025-01-16T04:31:28Z
- url: https://github.com/vllm-project/vllm/pull/8651
- labels: ready

# Issue

## Problem

FP8 dynamic quantization path in the apply method hardcoded `use_per_token_if_dynamic=False`, forcing per-tensor dynamic quantization even when CUTLASS FP8 GEMM kernels were available. Per-tensor quantization requires a separate amax reduction and quantization step that cannot be fused into the CUTLASS GEMM, leaving performance on the table. The dequantization/scaling could have been fused directly with the CUTLASS GEMM kernel but was not.

## Example

Llama2-7B on H100 ran at 152.60 requests/s, 10681.75 tokens/s before the change, bounded by the unfused amax + quantization overhead on the dynamic FP8 path.

## Use case

Serving FP8-quantized LLMs for high-throughput inference on hardware with CUTLASS FP8 GEMM support (e.g. H100).

## Resources

GPU compute and kernel-launch overhead — extra unfused amax reduction + quantization kernels consume cycles and launches that fused CUTLASS GEMM would avoid.

## Type of users

Users serving FP8 dynamically-quantized models on CUTLASS-capable GPUs (Hopper/H100 and similar).

## Kind of inputs

Standard dynamic-scale FP8 inference inputs where activation scales are computed at runtime per forward pass; any batch/sequence config using the FP8 dynamic path.

## Mechanism with the issue

In `vllm/model_executor/layers/quantization/fp8.py`, `Fp8LinearMethod.apply` called the FP8 linear helper with `use_per_token_if_dynamic=False` unconditionally. This selects per-tensor dynamic quantization, which runs a standalone amax + quant step instead of per-token quantization that CUTLASS can fuse into the GEMM epilogue, adding redundant compute and kernel launches.

# Solution

## Fix in words

Diff changes the hardcoded `use_per_token_if_dynamic=False` to `use_per_token_if_dynamic=self.cutlass_fp8_supported` in `Fp8LinearMethod.apply`. Now when CUTLASS FP8 is supported, the path uses per-token dynamic quantization, letting the amax + quantization step fuse with the CUTLASS GEMM kernel. When CUTLASS is unsupported the behavior stays per-tensor (False), preserving old fallback. Effect: fused dequant + GEMM yields ~4.68% throughput speedup with similar or slightly better accuracy.

## Author's solution

Default to per-token quantization for FP8 when CUTLASS is supported so dequantization fuses with the CUTLASS GEMM kernel, enabling fusion of amax + quantization. Author reports ~4.68% speedup on Llama2-7B H100 (152.60→159.73 req/s) with accuracy similar or slightly higher.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/quantization/fp8.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/quantization/fp8.py b/vllm/model_executor/layers/quantization/fp8.py
index a1be45a49e94..4969ee559522 100644
--- a/vllm/model_executor/layers/quantization/fp8.py
+++ b/vllm/model_executor/layers/quantization/fp8.py
@@ -355,7 +355,8 @@ def apply(self,
             input_scale=layer.input_scale,
             bias=bias,
             cutlass_fp8_supported=self.cutlass_fp8_supported,
-            use_per_token_if_dynamic=False)
+            # Default to using per_token quantization if cutlass is supported
+            use_per_token_if_dynamic=self.cutlass_fp8_supported)
 
 
 class Fp8MoEMethod(FusedMoEMethodBase):

```
