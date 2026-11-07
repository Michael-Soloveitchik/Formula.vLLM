# PR #36795: [Perf] Enable dual stream execution of input projection for Qwen3 (#36795)

# Issue

## Original PR description (captured source)


## Purpose

This PR Enable dual stream execution of input projection for Qwen3 Next.
* Parallelize the execution of `in_proj_qkvz` and `in_proj_ba` in 2 streams, because their outputs are independent.
* Wrap the implementation in custom op for torch.compile.

## Profiling

Main:

<img width="1722" height="180" alt="Screenshot 2026-03-10 at 6 08 16 PM" src="https://github.com/user-attachments/assets/58f0249e-c9de-47a6-8f00-218781344132" />

PR:

<img width="1723" height="139" alt="Screenshot 2026-03-10 at 6 08 31 PM" src="https://github.com/user-attachments/assets/10cd4f6b-df85-42b2-b164-92aab942c98b" />

Main: nvjet_tst_64x8_64x16_4x2_h_bz_TNT (in_proj_qkvz) and nvjet_tst_64x8_64x16_1x2_h_bz_TNT (in_proj_ba) kernels launched sequentially.

PR: kernels launched in parallel.

## Benchmarking

Benchmarked on H200.

* Qwen3

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  121.46    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              7.90      
Output token throughput (tok/s):         2371.20   
Peak output token throughput (tok/s):    2640.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4175.46   
---------------Time to First Token----------------
Mean TTFT (ms):                          176.72    
Median TTFT (ms):                        193.47    
P99 TTFT (ms):                           222.77    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.18      
Median TPOT (ms):                        6.15      
P99 TPOT (ms):                           6.48      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.18      
Median ITL (ms):                         6.13      
P99 ITL (ms):                            6.94      
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  120.20    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              7.99      
Output token throughput (tok/s):         2396.09   
Peak output token throughput (tok/s):    2672.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4219.28   
---------------Time to First Token----------------
Mean TTFT (ms):                          191.90    
Median TTFT (ms):                        214.37    
P99 TTFT (ms):                           249.63    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.06      
Median TPOT (ms):                        6.04      
P99 TPOT (ms):                           6.40      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.06      
Median ITL (ms):                         6.02      
P99 ITL (ms):                            6.72      
==================================================
```

* Qwen3 fp8

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  195.14    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              4.92      
Output token throughput (tok/s):         1475.89   
Peak output token throughput (tok/s):    1648.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2641.98   
---------------Time to First Token----------------
Mean TTFT (ms):                          162.16    
Median TTFT (ms):                        156.37    
P99 TTFT (ms):                           234.19    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.33     
Median TPOT (ms):                        10.31     
P99 TPOT (ms):                           10.77     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.33     
Median ITL (ms):                         10.24     
P99 ITL (ms):                            11.43     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  191.31    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              5.02      
Output token throughput (tok/s):         1505.38   
Peak output token throughput (tok/s):    1712.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2650.83   
---------------Time to First Token----------------
Mean TTFT (ms):                          236.09    
Median TTFT (ms):                        236.95    
P99 TTFT (ms):                           380.30    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.87      
Median TPOT (ms):                        9.83      
P99 TPOT (ms):                           10.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.87      
Median ITL (ms):                         9.81      
P99 ITL (ms):                            10.99     
==================================================
```

* Qwen3.5

```
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3.5-35B-A3B \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:
```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  197.96    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              4.85      
Output token throughput (tok/s):         1454.81   
Peak output token throughput (tok/s):    1648.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2604.25   
---------------Time to First Token----------------
Mean TTFT (ms):                          142.74    
Median TTFT (ms):                        152.15    
P99 TTFT (ms):                           199.72    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.55     
Median TPOT (ms):                        10.59     
P99 TPOT (ms):                           11.10     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.55     
Median ITL (ms):                         10.24     
P99 ITL (ms):                            12.05     
==================================================
```


PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  190.61    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              5.04      
Output token throughput (tok/s):         1510.93   
Peak output token throughput (tok/s):    1715.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2704.71   
---------------Time to First Token----------------
Mean TTFT (ms):                          169.85    
Median TTFT (ms):                        173.71    
P99 TTFT (ms):                           254.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.05     
Median TPOT (ms):                        10.06     
P99 TPOT (ms):                           10.28     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.05     
Median ITL (ms):                         10.06     
P99 ITL (ms):                            11.37     
==================================================
```

## Accuracy Testing

* Qwen3

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8575|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8150|±  |0.0107|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8552|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8082|±  |0.0108|
```

* Qwen3 fp8

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8575|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

* Qwen3.5

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-35B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8476|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8370|±  |0.0102|
```

PR:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8499|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8332|±  |0.0103|
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [36795.md](./36795.md) and [36795.json](./36795.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/36795
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/36795
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `3d822760d83b3bced84fa34320cbafefa65b6e19dcc6f7a7783ce8dfdaa64fa2`

Existing semantic notes in [36795.sem.json](./36795.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/models/qwen3_5.py`
- `vllm/model_executor/models/qwen3_next.py`
- `vllm/utils/multi_stream_utils.py`

### Diff

```diff
diff --git a/vllm/model_executor/models/qwen3_5.py b/vllm/model_executor/models/qwen3_5.py
index 9b1dc7468fb6..e5967c122945 100644
--- a/vllm/model_executor/models/qwen3_5.py
+++ b/vllm/model_executor/models/qwen3_5.py
@@ -180,12 +180,16 @@ def forward(
         # ============================================================
         # Part 1: Input Projection
         # ============================================================
-        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
+        mixed_qkvz, ba = torch.ops.vllm.gdn_in_proj(
+            hidden_states,
+            self.in_proj_qkvz.weight.shape[0],
+            self.in_proj_ba.weight.shape[0],
+            self.prefix,
+        )
         qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
         z_size = self.value_dim // self.tp_size
         mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
         z = z.reshape(z.size(0), -1, self.head_v_dim)
-        ba, _ = self.in_proj_ba(hidden_states)
         b, a = ba.chunk(2, dim=-1)
 
         b = b.contiguous()
diff --git a/vllm/model_executor/models/qwen3_next.py b/vllm/model_executor/models/qwen3_next.py
index bbe30c71903f..b94bcd2768f8 100644
--- a/vllm/model_executor/models/qwen3_next.py
+++ b/vllm/model_executor/models/qwen3_next.py
@@ -82,7 +82,11 @@
 from vllm.sequence import IntermediateTensors
 from vllm.transformers_utils.configs import Qwen3NextConfig
 from vllm.triton_utils import tl, triton
-from vllm.utils.torch_utils import direct_register_custom_op
+from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
+from vllm.utils.torch_utils import (
+    aux_stream,
+    direct_register_custom_op,
+)
 from vllm.v1.attention.backend import AttentionMetadata
 from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
 
@@ -419,6 +423,12 @@ def __init__(
         self.act = ACT2FN[config.hidden_act]
         self.layer_norm_epsilon = config.rms_norm_eps
         self.prefix = prefix
+        self.aux_stream = aux_stream()
+        self.events = (
+            [torch.cuda.Event(), torch.cuda.Event()]
+            if current_platform.is_cuda()
+            else [None, None]
+        )
 
         self.config = config
         self.model_config = model_config
@@ -647,8 +657,12 @@ def forward(
         # ============================================================
         # Part 1: Input Projection
         # ============================================================
-        projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
-        projected_states_ba, _ = self.in_proj_ba(hidden_states)
+        projected_states_qkvz, projected_states_ba = torch.ops.vllm.gdn_in_proj(
+            hidden_states,
+            self.in_proj_qkvz.weight.shape[0],
+            self.in_proj_ba.weight.shape[0],
+            self.prefix,
+        )
         query, key, value, z, b, a = self.fix_query_key_value_ordering(
             projected_states_qkvz, projected_states_ba
         )
@@ -783,6 +797,18 @@ def _warmup_prefill_kernels(self, mixed_qkv: torch.Tensor) -> None:
 
         torch.accelerator.empty_cache()
 
+    def _forward_in_proj(
+        self, hidden_states: torch.Tensor
+    ) -> tuple[torch.Tensor, torch.Tensor]:
+        projected_states_qkvz, projected_states_ba = maybe_execute_in_parallel(
+            lambda: self.in_proj_qkvz(hidden_states)[0],
+            lambda: self.in_proj_ba(hidden_states)[0],
+            self.events[0],
+            self.events[1],
+            self.aux_stream,
+        )
+        return projected_states_qkvz, projected_states_ba
+
     def _forward_core(
         self,
         mixed_qkv: torch.Tensor,
@@ -1670,6 +1696,32 @@ def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
         return self.model.get_expert_mapping()
 
 
+def gdn_in_proj(
+    hidden_states: torch.Tensor,
+    qkvz_output_size: int,
+    ba_output_size: int,
+    layer_name: str,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """
+    Custom op for the input projection.
+    """
+    forward_context: ForwardContext = get_forward_context()
+    self = forward_context.no_compile_layers[layer_name]
+    return self._forward_in_proj(hidden_states)
+
+
+def gdn_in_proj_fake(
+    hidden_states: torch.Tensor,
+    qkvz_output_size: int,
+    ba_output_size: int,
+    layer_name: str,
+) -> tuple[torch.Tensor, torch.Tensor]:
+    """Fake implementation for torch.compile."""
+    return hidden_states.new_empty(
+        hidden_states.shape[0], qkvz_output_size
+    ), hidden_states.new_empty(hidden_states.shape[0], ba_output_size)
+
+
 def gdn_attention_core(
     mixed_qkv: torch.Tensor,
     b: torch.Tensor,
@@ -1703,6 +1755,12 @@ def gdn_attention_core_fake(
     return
 
 
+direct_register_custom_op(
+    op_name="gdn_in_proj",
+    op_func=gdn_in_proj,
+    fake_impl=gdn_in_proj_fake,
+)
+
 direct_register_custom_op(
     op_name="gdn_attention_core",
     op_func=gdn_attention_core,
diff --git a/vllm/utils/multi_stream_utils.py b/vllm/utils/multi_stream_utils.py
new file mode 100644
index 000000000000..3ade910bf99c
--- /dev/null
+++ b/vllm/utils/multi_stream_utils.py
@@ -0,0 +1,48 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+
+from collections.abc import Callable
+from typing import Any
+
+import torch
+
+
+def maybe_execute_in_parallel(
+    fn0: Callable[[], Any],
+    fn1: Callable[[], Any],
+    event0: torch.cuda.Event,
+    event1: torch.cuda.Event,
+    aux_stream: torch.cuda.Stream | None = None,
+) -> tuple[Any, Any]:
+    """Run two functions potentially in parallel on separate CUDA streams.
+
+    When aux_stream is provided, fn0 runs on the current (default) stream and
+    fn1 runs on aux_stream, synchronized via CUDA events.  When aux_stream is
+    None, both functions execute sequentially on the current stream.
+
+    This design follows TensorRT-LLM's maybe_execute_in_parallel pattern
+    (tensorrt_llm/_torch/modules/multi_stream_utils.py).
+
+    Args:
+        fn0: Callable for the default stream.
+        fn1: Callable for the auxiliary stream.
+        event0: CUDA event recorded before fn0 so aux_stream can wait.
+        event1: CUDA event recorded after fn1 so default stream can wait.
+        aux_stream: The second CUDA stream for fn1.
+            Multi-stream is disabled when aux_stream is None.
+
+    Returns:
+        Tuple of (fn0_result, fn1_result).
+    """
+    if aux_stream is not None:
+        event0.record()
+        result0 = fn0()
+        with torch.cuda.stream(aux_stream):
+            event0.wait()
+            result1 = fn1()
+            event1.record()
+        event1.wait()
+    else:
+        result0 = fn0()
+        result1 = fn1()
+    return (result0, result1)
```
