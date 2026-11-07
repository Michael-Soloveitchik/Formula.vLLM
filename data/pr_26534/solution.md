# PR #26534: Move query quantization to attention layer for Flashinfer & Triton. (#26534)

# Issue

## Original PR description (captured source)

### Purpose
Implements refactor of quantization to the attention layer for triton and flashinfer, resolves feature request [#25584](https://github.com/vllm-project/vllm/issues/25584)

### Test Plan
Spin up server:

Flashinfer:
```
VLLM_ATTENTION_BACKEND=FLASHINFER  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --no-enable-prefix-caching
```
  
Triton:
```
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8 \
  --compilation-config '{"compile_sizes": [1,2,4,8], "cudagraph_capture_sizes": [1,2,4,8], "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --no-enable-prefix-caching
```

Benchmark:
```
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct  \
    --dataset-name sonnet \
    --dataset-path vllm/benchmarks/sonnet.txt \
    --sonnet-input-len 1000 \
    --sonnet-output-len 200 \
    --port 8000 \
    --num-prompts 20 \
    --max-concurrency 1
```
### Accuracy
To ensure there is no accidental accuracy degradation we also run the following for Flashinfer & Triton with kv_cache_dtype in {auto,fp8} both on this PR and on mainline. We also run without enforce_eager=True for the FP8 variants
```
lm_eval \
  --model vllm \
  --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,kv_cache_dtype=auto,tensor_parallel_size=1,enforce_eager=True \
  --tasks gsm8k \
  --batch_size 
```

### Test Results
```
PR + FlashInfer
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  16.81     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.19      
Output token throughput (tok/s):         237.97    
Peak output token throughput (tok/s):    240.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1323.60   
---------------Time to First Token----------------
Mean TTFT (ms):                          20.13     
Median TTFT (ms):                        19.97     
P99 TTFT (ms):                           23.20     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.12      
Median TPOT (ms):                        4.13      
P99 TPOT (ms):                           4.15      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.12      
Median ITL (ms):                         4.12      
P99 ITL (ms):                            4.61      
==================================================

PR + Triton
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  17.23     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.16      
Output token throughput (tok/s):         232.19    
Peak output token throughput (tok/s):    235.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1291.45   
---------------Time to First Token----------------
Mean TTFT (ms):                          24.86     
Median TTFT (ms):                        24.77     
P99 TTFT (ms):                           31.37     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.20      
Median TPOT (ms):                        4.20      
P99 TPOT (ms):                           4.24      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.20      
Median ITL (ms):                         4.20      
P99 ITL (ms):                            4.58      
==================================================

Main + FlashInfer
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  18.15     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.10      
Output token throughput (tok/s):         220.42    
Peak output token throughput (tok/s):    223.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1225.98   
---------------Time to First Token----------------
Mean TTFT (ms):                          24.18     
Median TTFT (ms):                        24.08     
P99 TTFT (ms):                           29.89     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.44      
Median TPOT (ms):                        4.44      
P99 TPOT (ms):                           4.46      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.44      
Median ITL (ms):                         4.43      
P99 ITL (ms):                            5.01      
==================================================

Main + Triton
============ Serving Benchmark Result ============
Successful requests:                     20        
Maximum request concurrency:             1         
Benchmark duration (s):                  17.42     
Total input tokens:                      18248     
Total generated tokens:                  4000      
Request throughput (req/s):              1.15      
Output token throughput (tok/s):         229.59    
Peak output token throughput (tok/s):    234.00    
Peak concurrent requests:                3.00      
Total Token throughput (tok/s):          1276.95   
---------------Time to First Token----------------
Mean TTFT (ms):                          27.61     
Median TTFT (ms):                        26.97     
P99 TTFT (ms):                           37.66     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4.24      
Median TPOT (ms):                        4.24      
P99 TPOT (ms):                           4.32      
---------------Inter-token Latency----------------
Mean ITL (ms):                           4.24      
Median ITL (ms):                         4.23      
P99 ITL (ms):                            4.94      
==================================================
```

### Accuracy on GSM8k
This PR shows no accuracy variation compared to mainline vllm.
```
PR + Triton + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7817|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|

PR + FlashInfer + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7801|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7551|±  |0.0118|


Main + FlashInfer + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7854|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7635|±  |0.0118|


Main + Triton + auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7817|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|


PR + Triton + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7771|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7521|±  |0.0119|

PR + FlashInfer + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7665|±  |0.0117|
|     |       |strict-match    |     5|exact_match|↑  |0.7483|±  |0.0120|

Main + FlashInfer + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7665|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7483|±  |0.0118|


Main + Triton + fp8 (enforce-eager)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7771|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7521|±  |0.0118|


PR + Triton + fp8 (w/ compile)
Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7695|±  |0.0116|
|     |       |strict-match    |     5|exact_match|↑  |0.7460|±  |0.0120|


PR + FlashInfer + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7756|±  |0.0115|
|     |       |strict-match    |     5|exact_match|↑  |0.7566|±  |0.0118|

Main + FlashInfer + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7680|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7437|±  |0.0118|


Main + Triton + fp8 (w/ compile)
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7672|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7468|±  |0.0118| 
```

## Full discussion

See [26534.md](./26534.md) and [26534.json](./26534.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/26534
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/26534.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `b4d3c0f5def96d1d4b36681f093bcd6222d685f607eebfd6866f7edf2f9546bc`

Existing semantic notes in [26534.sem.json](./26534.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/compile/test_fusion_attn.py`
- `vllm/attention/backends/abstract.py`
- `vllm/attention/layer.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/backends/triton_attn.py`

### Diff

```diff
diff --git a/tests/compile/test_fusion_attn.py b/tests/compile/test_fusion_attn.py
index a8d78daa32a1..4d6f4b471a3a 100644
--- a/tests/compile/test_fusion_attn.py
+++ b/tests/compile/test_fusion_attn.py
@@ -421,7 +421,9 @@ def test_attention_quant_pattern(
     ]
     if any(attn_fusion_supported):
         # Check quantization ops in the graph before and after fusion
-        test_backend.check_before_ops([QUANT_OPS[quant_key]], fully_replaced=True)
+        # Note: fully_replaced=False because query quant ops remain in graph.
+        # Only output quant ops are fused into attention.
+        test_backend.check_before_ops([QUANT_OPS[quant_key]], fully_replaced=False)
 
     # access the underlying `AttnFusionPass` on the `LazyInitPass`
     assert attn_pass.pass_.matched_count == sum(attn_fusion_supported)
diff --git a/vllm/attention/backends/abstract.py b/vllm/attention/backends/abstract.py
index 421b0c4beb37..fb2db4d0b0ec 100644
--- a/vllm/attention/backends/abstract.py
+++ b/vllm/attention/backends/abstract.py
@@ -41,14 +41,6 @@ class AttentionBackend(ABC):
     # makes sure the output tensor is allocated inside the cudagraph.
     accept_output_buffer: bool = False
 
-    # Whether this backend supports receiving pre-quantized query input.
-    # If True, the attention layer will handle query quantization instead
-    # of the backend, allowing torch.compile to fuse quantization with
-    # previous operations.
-    # Needs to be worked through for all backends
-    # https://github.com/vllm-project/vllm/issues/25584
-    supports_quant_query_input: bool = False
-
     @staticmethod
     @abstractmethod
     def get_name() -> str:
@@ -199,6 +191,22 @@ def fused_output_quant_supported(self, quant_key: QuantKey):
         """
         return False
 
+    def supports_quant_query_input(self) -> bool:
+        """
+        Check if this attention implementation supports pre-quantized query input.
+
+        When True, the attention layer will quantize queries before passing them
+        to this backend, allowing torch.compile to fuse the quantization with
+        previous operations. This is typically supported when using FP8 KV cache
+        with compatible attention kernels (e.g., TRT-LLM).
+        TODO add support to more backends:
+        https://github.com/vllm-project/vllm/issues/25584
+
+        Returns:
+            bool: True if the implementation can accept pre-quantized queries.
+        """
+        return False
+
 
 class MLAAttentionImpl(AttentionImpl[T], Generic[T]):
     @abstractmethod
diff --git a/vllm/attention/layer.py b/vllm/attention/layer.py
index 16c5799f7d0b..9f879f7272e2 100644
--- a/vllm/attention/layer.py
+++ b/vllm/attention/layer.py
@@ -36,6 +36,7 @@
 from vllm.platforms import current_platform
 from vllm.utils import GiB_bytes, direct_register_custom_op
 
+FP8_DTYPE = current_platform.fp8_dtype()
 logger = init_logger(__name__)
 USE_XFORMERS_OPS = None
 
@@ -304,7 +305,7 @@ def __init__(
         self.query_quant = None
         if (
             self.kv_cache_dtype.startswith("fp8")
-            and self.attn_backend.supports_quant_query_input
+            and self.impl.supports_quant_query_input()
         ):
             self.query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
 
@@ -329,7 +330,6 @@ def forward(
         """
         if self.calculate_kv_scales:
             torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
-
         output_dtype = query.dtype
         if self.query_quant is not None:
             # quantizing with a simple torch operation enables
@@ -338,7 +338,10 @@ def forward(
             # Otherwise queries are quantized using custom ops
             # which causes decoding overheads
             assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}
-            query, _ = self.query_quant(query, self._q_scale)
+
+            # check if query quantization is supported
+            if self.impl.supports_quant_query_input():
+                query, _ = self.query_quant(query, self._q_scale)
 
         if self.use_output:
             output_shape = output_shape if output_shape is not None else query.shape
diff --git a/vllm/v1/attention/backends/flash_attn.py b/vllm/v1/attention/backends/flash_attn.py
index 9e0c125d9edb..087f995e0528 100755
--- a/vllm/v1/attention/backends/flash_attn.py
+++ b/vllm/v1/attention/backends/flash_attn.py
@@ -49,7 +49,6 @@
 
 class FlashAttentionBackend(AttentionBackend):
     accept_output_buffer: bool = True
-    supports_quant_query_input: bool = True
 
     @classmethod
     def get_supported_dtypes(cls) -> list[torch.dtype]:
@@ -494,6 +493,9 @@ def __init__(
                 "heads in the layer"
             )
 
+    def supports_quant_query_input(self) -> bool:
+        return True
+
     def forward(
         self,
         layer: torch.nn.Module,
diff --git a/vllm/v1/attention/backends/flashinfer.py b/vllm/v1/attention/backends/flashinfer.py
index ee32f7e2904f..eb9f6a280d8f 100755
--- a/vllm/v1/attention/backends/flashinfer.py
+++ b/vllm/v1/attention/backends/flashinfer.py
@@ -16,7 +16,6 @@
 from flashinfer.prefill import trtllm_batch_context_with_kv_cache
 from flashinfer.utils import FP4Tensor
 
-from vllm import _custom_ops as ops
 from vllm.attention.backends.abstract import (
     AttentionBackend,
     AttentionImpl,
@@ -828,6 +827,12 @@ def fused_output_quant_supported(self, quant_key: QuantKey):
             and quant_key in (kFp8StaticTensorSym, kNvfp4Quant)
         )
 
+    def supports_quant_query_input(self) -> bool:
+        if flashinfer_disable_q_quantization():
+            return False
+
+        return self.support_trtllm_attn
+
     def forward(
         self,
         layer: torch.nn.Module,
@@ -859,6 +864,12 @@ def forward(
             # Profiling run.
             return output.fill_(0)
 
+        # Ensure query dtype matches the expected dtype from attention metadata
+        assert attn_metadata.q_data_type == query.dtype, (
+            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
+            f"got {query.dtype}"
+        )
+
         if self.bmm1_scale is None:
             self.bmm1_scale = layer._q_scale_float * layer._k_scale_float * self.scale
 
@@ -899,15 +910,6 @@ def forward(
                 elif output.dtype == FP4_DTYPE:
                     self.o_sf_scale = layer._o_scale_float
 
-        # Insert FP8 quant for query
-        if attn_metadata.q_data_type == FP8_DTYPE:
-            num_tokens, num_heads, head_size = query.shape
-            query, _ = ops.scaled_fp8_quant(
-                query.reshape((num_tokens, num_heads * head_size)).contiguous(),
-                layer._q_scale,
-            )
-            query = query.reshape((num_tokens, num_heads, head_size))
-
         # IMPORTANT!
         # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
         # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
diff --git a/vllm/v1/attention/backends/triton_attn.py b/vllm/v1/attention/backends/triton_attn.py
index 9746a0eb58bd..b1d34dbfd172 100644
--- a/vllm/v1/attention/backends/triton_attn.py
+++ b/vllm/v1/attention/backends/triton_attn.py
@@ -32,11 +32,6 @@
 )
 from vllm.v1.kv_cache_interface import AttentionSpec
 
-if current_platform.is_cuda_alike():
-    from vllm import _custom_ops as ops
-elif current_platform.is_xpu():
-    from vllm._ipex_ops import ipex_ops as ops
-
 logger = init_logger(__name__)
 
 
@@ -210,6 +205,9 @@ class TritonAttentionImpl(AttentionImpl):
     def fused_output_quant_supported(self, quant_key: QuantKey):
         return quant_key == kFp8StaticTensorSym
 
+    def supports_quant_query_input(self) -> bool:
+        return current_platform.is_cuda()
+
     def __init__(
         self,
         num_heads: int,
@@ -338,19 +336,9 @@ def forward(
             if key_cache.dtype != self.fp8_dtype:
                 key_cache = key_cache.view(self.fp8_dtype)
                 value_cache = value_cache.view(self.fp8_dtype)
-            num_tokens, num_heads, head_size = query.shape
             assert layer._q_scale_float == 1.0, (
                 "A non 1.0 q_scale is not currently supported."
             )
-            if current_platform.is_cuda():
-                # Skip Q quantization on ROCm and XPU, enable this on cuda
-                # only, since dequantizing back to f32 in the attention kernel
-                # is not supported.
-                query, _ = ops.scaled_fp8_quant(
-                    query.reshape((num_tokens, num_heads * head_size)).contiguous(),
-                    layer._q_scale,
-                )
-                query = query.reshape((num_tokens, num_heads, head_size))
 
         cu_seqlens_q = attn_metadata.query_start_loc
         seqused_k = attn_metadata.seq_lens
```
