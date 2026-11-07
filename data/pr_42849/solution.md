# PR #42849: [Perf] Add do_not_specialize in fused FP8 RoPE kernel

- field: gpu_kernels  | area: v1_attention
- author: xyang16
- state: MERGED  created: 2026-05-16T20:29:03Z  merged: 2026-05-18T08:32:47Z
- url: https://github.com/vllm-project/vllm/pull/42849
- labels: performance, ready, v1

# Issue

## Problem

The Triton kernel `_fused_inv_rope_fp8_quant_per_head` was decorated with a plain `@triton.jit`, so Triton treated the `num_tokens` runtime argument as a specialization key. Triton specializes integer arguments by their value (and divisibility properties), meaning each previously-unseen `num_tokens` value triggered a fresh JIT compilation of the kernel. During inference, prefill/decode batches arrive with varying token counts, so the kernel kept recompiling instead of reusing a cached compiled artifact.

## Example

Serving DeepSeek-V4-Flash on 8x H200 (TP=8, max-num-seqs=8, fp8 KV cache, block-size 256), the first time a given batch's `num_tokens` is encountered, the kernel recompiles, inflating time-to-first-token. Mean TTFT dropped from 370.14 ms to 338.37 ms (~9%) once recompilation was eliminated.

## Use case

Online LLM serving where request batches have continuously varying token counts, hitting the fused inverse-RoPE + FP8-quant path per-head on each forward pass.

## Resources

Kernel-launch/JIT compile time (Triton recompilation on the host CPU), which stalls the GPU and adds latency rather than steady-state compute.

## Type of users

Users serving DeepSeek-V4 / MLA-style models with FP8 KV cache on multi-GPU tensor-parallel setups (H200-class hardware) using the Triton fused RoPE+quant kernel.

## Kind of inputs

Requests with diverse `num_tokens` per batch — variable prompt lengths and dynamic batch sizes during prefill/decode — so new token-count values keep appearing.

## Mechanism with the issue

`_fused_inv_rope_fp8_quant_per_head` in `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py` takes `num_tokens` as a kernel arg. Under default `@triton.jit`, Triton auto-specializes scalar int args, so each novel `num_tokens` value is a distinct specialization and forces a new compile. Repeated compiles serialize on the CPU and delay kernel execution, raising TTFT.

# Solution

## Fix in words

The diff changes the decorator from `@triton.jit` to `@triton.jit(do_not_specialize=["num_tokens"])`. This tells Triton to stop treating `num_tokens` as a specialization key, so it is passed as a generic runtime value instead of baking its value into a compiled variant. As a result, varying token counts reuse one cached compiled kernel instead of triggering fresh JIT compiles. Effect: eliminates per-batch recompilation overhead, improving e2e TTFT ~9% on H200.

## Author's solution

Add `do_not_specialize=["num_tokens"]` to the `@triton.jit` decorator of `_fused_inv_rope_fp8_quant_per_head` to prevent unnecessary recompilation when a `num_tokens` value is seen for the first time during inference. Author reports e2e TTFT improves 9% on H200.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py`

### Diff
```diff
diff --git a/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py b/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
index 68d33f1aa105..c650eabd00d2 100644
--- a/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
+++ b/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py
@@ -14,7 +14,7 @@
 from vllm.utils.torch_utils import direct_register_custom_op
 
 
-@triton.jit
+@triton.jit(do_not_specialize=["num_tokens"])
 def _fused_inv_rope_fp8_quant_per_head(
     o_ptr,
     positions_ptr,

```
