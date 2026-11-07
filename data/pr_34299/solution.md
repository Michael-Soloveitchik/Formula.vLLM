# PR #34299: [torch.compile] Enable AR+rms fusion by default available for `-O2`

- field: gpu_kernels  | area: vllm_config
- author: ProExpertProg
- state: MERGED  created: 2026-02-11T01:59:12Z  merged: 2026-02-11T08:30:00Z
- url: https://github.com/vllm-project/vllm/pull/34299
- labels: performance, ready, torch.compile

# Issue

## Problem

The AllReduce+RMSNorm (AR+RMS) fusion pass was disabled by default at every optimization level, including `-O2`, with `fuse_allreduce_rms` hardcoded to `False`. This left a known 5-22% latency improvement on the table for multi-GPU tensor-parallel setups. The fusion was gated behind a manual flag that users had to set explicitly, so almost nobody benefited.

## Example

Serving `RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8` with `-tp=4` on B200: without the fusion TTFT/TPOT ran 5-22% slower than achievable. Manually enabling via `-cc.pass_config.fuse_allreduce_rms` was the only way to get it.

## Use case

Multi-GPU tensor-parallel inference of large dense and MoE models (Llama-3.1-70B, Llama-3.3-70B, Qwen3-235B, DeepSeek-V3.1) where every decode layer issues an all-reduce immediately followed by an RMSNorm.

## Resources

NVLink/inter-GPU collective bandwidth and kernel-launch overhead: the unfused path launches a separate all-reduce kernel and a separate RMSNorm kernel per layer, adding latency on the critical decode path.

## Type of users

Users running TP>1 on Hopper+ (sm90+) CUDA GPUs with FlashInfer installed, serving FP8/NVFP4 quantized or dense large models at `-O2` and above.

## Kind of inputs

Decode-heavy and prefill workloads on tensor-parallel models; the per-layer all-reduce→RMSNorm pattern triggers regardless of batch/sequence specifics, so any TP>1 request benefits.

## Mechanism with the issue

In `vllm/config/vllm.py` the `OPTIMIZATION_LEVEL_02` and `_03` pass configs set `fuse_allreduce_rms: False`, so the torch.compile fusion pass that merges the collective with the following norm never ran. Because the fusion marginally raises compile time, it was conservatively left off rather than auto-enabled when hardware supports it.

# Solution

## Fix in words

In `vllm/config/vllm.py`, adds `enable_allreduce_rms_fusion(cfg)` which returns True when `tensor_parallel_size > 1`, platform is CUDA with device capability >= 90 (Hopper+), and FlashInfer is installed. `OPTIMIZATION_LEVEL_02` and `_03` now set `fuse_allreduce_rms` to this callable instead of hardcoded `False`, so the AR+RMS fusion turns on automatically on supported setups. Separately, `eliminate_noops` default changes from `None` to `True` in `PassConfig` and is dropped from both the per-level configs and the `compute_hash` field list, making no-op elimination unconditional. Net effect: `-O2`+ gets AR+RMS fusion by default, yielding the measured 5-22% latency gains at minor compile-time cost.

## Author's solution

Author found AR+RMS fusion plus compiling for a second range only marginally increases warm/cold compile time while delivering 5-22% benefits (per #24252), so this PR enables the fusion by default. Validated via startup timing, latency benchmarks on llama3-70B-fp8 and DeepSeek-V3.1-NVFP4 (B200), gsm8k eval across several FP8/NVFP4 models, and CI.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/config/compilation.py`
- `vllm/config/vllm.py`

### Diff
```diff
diff --git a/vllm/config/compilation.py b/vllm/config/compilation.py
index fb7a1466b2eb..f1909ace6dcb 100644
--- a/vllm/config/compilation.py
+++ b/vllm/config/compilation.py
@@ -115,7 +115,7 @@ class PassConfig:
     """Fuse the custom SiluMul + quant ops."""
     fuse_attn_quant: bool = Field(default=None)
     """Fuse the custom attention + quant ops."""
-    eliminate_noops: bool = Field(default=None)
+    eliminate_noops: bool = Field(default=True)
     """Eliminate no-op ops."""
     enable_sp: bool = Field(default=None)
     """Enable sequence parallelism."""
@@ -194,7 +194,6 @@ def compute_hash(self) -> str:
         "fuse_norm_quant",
         "fuse_act_quant",
         "fuse_attn_quant",
-        "eliminate_noops",
         "enable_sp",
         "fuse_gemm_comms",
         "fuse_allreduce_rms",
diff --git a/vllm/config/vllm.py b/vllm/config/vllm.py
index c1ef8e6aae39..eccaa6ce653b 100644
--- a/vllm/config/vllm.py
+++ b/vllm/config/vllm.py
@@ -102,6 +102,19 @@ def enable_act_fusion(cfg: "VllmConfig") -> bool:
     ) or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
 
 
+def enable_allreduce_rms_fusion(cfg: "VllmConfig") -> bool:
+    """Enable if TP > 1 and Hopper+ and flashinfer installed."""
+    from vllm.platforms import current_platform
+    from vllm.utils.flashinfer import has_flashinfer
+
+    return (
+        cfg.parallel_config.tensor_parallel_size > 1
+        and current_platform.is_cuda()
+        and current_platform.has_device_capability(90)
+        and has_flashinfer()
+    )
+
+
 def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
     """Enable if using AITER RMSNorm and AITER Triton GEMMs
     and hidden size is 2880 i.e. gpt-oss; otherwise Inductor handles fusion."""
@@ -118,7 +131,6 @@ def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
 OPTIMIZATION_LEVEL_00 = {
     "compilation_config": {
         "pass_config": {
-            "eliminate_noops": False,
             "fuse_norm_quant": False,
             "fuse_act_quant": False,
             "fuse_allreduce_rms": False,
@@ -137,7 +149,6 @@ def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
 OPTIMIZATION_LEVEL_01 = {
     "compilation_config": {
         "pass_config": {
-            "eliminate_noops": True,
             "fuse_norm_quant": enable_norm_fusion,
             "fuse_act_quant": enable_act_fusion,
             "fuse_allreduce_rms": False,
@@ -156,10 +167,9 @@ def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
 OPTIMIZATION_LEVEL_02 = {
     "compilation_config": {
         "pass_config": {
-            "eliminate_noops": True,
             "fuse_norm_quant": enable_norm_fusion,
             "fuse_act_quant": enable_act_fusion,
-            "fuse_allreduce_rms": False,
+            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
             "fuse_attn_quant": IS_QUANTIZED,
             "enable_sp": IS_DENSE,
             "fuse_gemm_comms": IS_DENSE,
@@ -175,10 +185,9 @@ def enable_norm_pad_fusion(cfg: "VllmConfig") -> bool:
 OPTIMIZATION_LEVEL_03 = {
     "compilation_config": {
         "pass_config": {
-            "eliminate_noops": True,
             "fuse_norm_quant": enable_norm_fusion,
             "fuse_act_quant": enable_act_fusion,
-            "fuse_allreduce_rms": False,
+            "fuse_allreduce_rms": enable_allreduce_rms_fusion,
             "fuse_attn_quant": IS_QUANTIZED,
             "enable_sp": IS_DENSE,
             "fuse_gemm_comms": IS_DENSE,

```
