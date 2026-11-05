# PR #23351: [Feature] Enable DeepGEMM Linear on B200; 1.5% E2E throughput improvement (#23351)

# Issue

## Original PR description (captured source)

## Purpose

Enable DeepGEMM Linear on B200

This should also fix some cutlass linear acc error since the weight is quantized to e8m0

## Test

```bash
VLLM_USE_DEEP_GEMM=1 lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8529|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8855|±  |0.0088|

# main
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8590|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8969|±  |0.0084|
```

```bash
VLLM_USE_DEEP_GEMM=1 vllm bench throughput --model Qwen/Qwen3-30B-A3B-FP8 --load-format dummy --input-len 1000 --output-len 100 --trust_remote_code --enable-expert-parallel
Throughput: 40.59 requests/s, 44557.64 total tokens/s, 4059.10 output tokens/s
# main
Throughput: 39.97 requests/s, 43880.62 total tokens/s, 3997.42 output tokens/s
```

## Full discussion

See [23351.md](./23351.md) and [23351.json](./23351.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/23351
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/23351.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `7cf8f37e4f433e23745faa0f8ce2c5756e08d8f1669cf2173cc69f7b9d3a1cd7`

Existing semantic notes in [23351.sem.json](./23351.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/layers/quantization/utils/fp8_utils.py`
- `vllm/utils/deep_gemm.py`

### Diff

```diff
diff --git a/vllm/model_executor/layers/quantization/utils/fp8_utils.py b/vllm/model_executor/layers/quantization/utils/fp8_utils.py
index 2fb7ef29e468..ab1d5383f465 100644
--- a/vllm/model_executor/layers/quantization/utils/fp8_utils.py
+++ b/vllm/model_executor/layers/quantization/utils/fp8_utils.py
@@ -19,8 +19,9 @@
     CUTLASS_BLOCK_FP8_SUPPORTED)
 from vllm.platforms import current_platform
 from vllm.triton_utils import tl, triton
-from vllm.utils import cdiv, direct_register_custom_op, has_deep_gemm
-from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used
+from vllm.utils import cdiv, direct_register_custom_op
+from vllm.utils.deep_gemm import (is_blackwell_deep_gemm_e8m0_used,
+                                  should_use_deepgemm_for_fp8_linear)
 
 logger = init_logger(__name__)
 
@@ -108,19 +109,6 @@ def dispatch_w8a8_blockscale_func(
     return w8a8_block_fp8_matmul
 
 
-def should_use_deepgemm(output_dtype: torch.dtype, weight: torch.Tensor):
-    """
-    Check if DeepGEMM should be used based on the output dtype and weight shape.
-    DeepGEMM is only supported for bfloat16 output dtype and weights with shape
-    divisible by 128.
-    """
-
-    return (current_platform.is_cuda()
-            and current_platform.is_device_capability(90) and has_deep_gemm()
-            and envs.VLLM_USE_DEEP_GEMM and output_dtype == torch.bfloat16
-            and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)
-
-
 # TODO fix ROCm->Triton custom path:
 #  https://github.com/vllm-project/vllm/issues/14397
 def apply_w8a8_block_fp8_linear(
@@ -139,7 +127,7 @@ def apply_w8a8_block_fp8_linear(
     output_shape = [*input.shape[:-1], weight.shape[0]]
     output_dtype = input.dtype
 
-    if should_use_deepgemm(output_dtype, weight):
+    if should_use_deepgemm_for_fp8_linear(output_dtype, weight):
 
         input_2d = input.view(-1, input.shape[-1])
         output_shape = [*input.shape[:-1], weight.shape[0]]
@@ -150,7 +138,9 @@ def apply_w8a8_block_fp8_linear(
             column_major_scales=True,
         )
 
+        # ensure DeepGEMM-backed custom op is registered before use
         import vllm.model_executor.layers.quantization.deepgemm  # noqa: F401
+
         output = torch.ops.vllm.w8a8_block_fp8_matmul_deepgemm(
             q_input,
             weight,
diff --git a/vllm/utils/deep_gemm.py b/vllm/utils/deep_gemm.py
index 861d9c0c0005..c0a4ed077e66 100644
--- a/vllm/utils/deep_gemm.py
+++ b/vllm/utils/deep_gemm.py
@@ -202,6 +202,12 @@ def calc_diff(x: torch.Tensor, y: torch.Tensor):
     return 1 - sim
 
 
+def should_use_deepgemm_for_fp8_linear(output_dtype: torch.dtype,
+                                       weight: torch.Tensor):
+    return (is_deep_gemm_supported() and output_dtype == torch.bfloat16
+            and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0)
+
+
 __all__ = [
     "calc_diff",
     "fp8_gemm_nt",
@@ -210,4 +216,5 @@ def calc_diff(x: torch.Tensor, y: torch.Tensor):
     "per_block_cast_to_fp8",
     "is_blackwell_deep_gemm_e8m0_used",
     "is_deep_gemm_supported",
+    "should_use_deepgemm_for_fp8_linear",
 ]
```
