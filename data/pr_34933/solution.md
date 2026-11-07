# PR #34933: [FIX] fused moe with lora shared expert dual stream (1.07x otps)

- field: parallelism  | area: lora_layers
- author: jhaotingc
- state: MERGED  created: 2026-02-20T01:53:25Z  merged: 2026-02-25T04:40:46Z
- url: https://github.com/vllm-project/vllm/pull/34933
- labels: performance, ready

# Issue

## Problem

MoE models with a shared expert get dual-stream overlap in tensor parallelism: the shared expert runs on a separate CUDA stream concurrently with the routed experts. But when a LoRA module is enabled, the LoRA fused-MoE wrapper hands the shared experts to the modular MoE kernel, which runs them inside the kernel instead of leaving them for the runner. This serializes shared-expert and routed-expert compute, eliminating the overlap and dropping throughput.

## Example

Serving nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 on H200 with TP and a LoRA adapter (Nemotron_3_Nano_BF16_r16_zeros, rank 16), the shared-expert/routed-expert overlap visible in nsys disappears once LoRA is enabled, losing ~1.02x-1.08x OTPS/GPU across concurrency 1-64.

## Use case

Serving a shared-expert MoE model with LoRA adapters enabled under tensor parallelism, the common multi-tenant adapter-serving setup.

## Resources

GPU compute utilization. Overlap keeps the SMs busy by running shared and routed experts concurrently on two CUDA streams; without it compute idles during serialization.

## Type of users

Users serving shared-expert MoE models (e.g. Nemotron MoE) with LoRA adapters under multi-GPU tensor parallelism.

## Kind of inputs

Any inference requests routed through the LoRA path, even requests with no LoRA attached, since enabling LoRA on the engine forces the LoRA fused-MoE code path. Tested at input len 1600, output len 600, concurrency 1-64.

## Mechanism with the issue

In vllm/lora/layers/fused_moe.py, _inject_lora_into_fused_moe builds/reuses a FusedMoEModularKernel. Both branches attached self.base_layer.shared_experts to the kernel (passed as a constructor arg, or already set on moe_mk). When the kernel owns shared_experts it computes them inline, so the runner's separate-stream overlap path never fires.

# Solution

## Fix in words

The diff stops the LoRA fused-MoE kernel from owning the shared experts. In the supports_internal_mk branch, it sets m_fused_moe_fn.shared_experts = None on the reused moe_mk kernel. In the select_gemm_impl branch, it drops the self.base_layer.shared_experts argument from the FusedMoEModularKernel constructor. With shared_experts unset on the kernel, the runner retains ownership and overlaps them with routed experts on a separate CUDA stream, restoring the dual-stream overlap and yielding ~1.02x-1.08x OTPS/GPU.

## Author's solution

Author found shared-expert dual-stream overlap works in TP without LoRA but vanishes when a LoRA module is added. Fix keeps shared experts out of the LoRA MoE modular kernel so the runner can overlap them with routed experts via a separate CUDA stream. Verified via nsys that enabling LoRA routes through the LoRA path, accuracy-tested the LoRA server with no-LoRA requests on gsm8k, and benchmarked ~1.07x OTPS/GPU.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/lora/layers/fused_moe.py`

### Diff
```diff
diff --git a/vllm/lora/layers/fused_moe.py b/vllm/lora/layers/fused_moe.py
index ed33452bf55d..e08dcc87e90e 100644
--- a/vllm/lora/layers/fused_moe.py
+++ b/vllm/lora/layers/fused_moe.py
@@ -133,15 +133,19 @@ def _inject_lora_into_fused_moe(self):
         if getattr(self.base_layer.quant_method, "supports_internal_mk", False):
             # Use the existing modular kernel from the quant method
             m_fused_moe_fn = self.base_layer.quant_method.moe_mk
+            # Don't let the kernel own shared experts so the runner can
+            # overlap them with routed experts via a separate CUDA stream.
+            m_fused_moe_fn.shared_experts = None
         else:
-            # Create a new modular kernel via select_gemm_impl
+            # Create a new modular kernel via select_gemm_impl.
+            # Don't pass shared_experts to the kernel so the runner can
+            # overlap them with routed experts via a separate CUDA stream.
             prepare_finalize = MoEPrepareAndFinalizeNoEP()
             m_fused_moe_fn = FusedMoEModularKernel(
                 prepare_finalize,
                 self.base_layer.quant_method.select_gemm_impl(
                     prepare_finalize, self.base_layer
                 ),
-                self.base_layer.shared_experts,
             )
 
         if quant_config.use_mxfp4_w4a16:

```
