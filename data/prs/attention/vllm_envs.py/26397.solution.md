# PR #26397: [Attention] Add MLA prefill backend: trtllm_ragged_attention_deepseek (#26397)

# Issue

## Original PR description (captured source)

## Purpose

Add MLA prefill backend: trtllm_ragged_attention_deepseek
* controlled by `VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1`

* [x] Need to fix accuracy issue
  * 0.54 -> 0.59 fixed by zeroing output tensor for context chunk prefill
  * 0.59 -> 0.61 (0.22 -> 0.64 w/ prefix caching) fixed by transposing lse
  * 0.61 -> 0.65 fixed by zeroing workspace
* [x] performance benchmark

## Test Plan

* [ ] microbenchmark
  * in follow-up work after refactoring
* [x] vllm bench serve
* [x] lm_eval gsm8k

## Test Result

### bench serve

12.7% qps improvement
-10.2% TTFT improvement

#### Baseline
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/fi.log
```
```
============ Serving Benchmark Result ============
Successful requests:                     2048
Maximum request concurrency:             2048
Benchmark duration (s):                  200.42
Total input tokens:                      4192256
Total generated tokens:                  2097152
Request throughput (req/s):              10.22
Output token throughput (tok/s):         10463.54
Peak output token throughput (tok/s):    16094.00
Peak concurrent requests:                2048.00
Total Token throughput (tok/s):          31380.41
---------------Time to First Token----------------
Mean TTFT (ms):                          64511.82
Median TTFT (ms):                        55968.36
P99 TTFT (ms):                           132592.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          86.54
Median TPOT (ms):                        89.57
P99 TPOT (ms):                           93.98
---------------Inter-token Latency----------------
Mean ITL (ms):                           86.54
Median ITL (ms):                         70.26
P99 ITL (ms):                            208.33
==================================================
```
#### This PR
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" 2>&1 | tee /tmp/ragged.log
```

```
============ Serving Benchmark Result ============
Successful requests:                     2048
Maximum request concurrency:             2048
Benchmark duration (s):                  177.74
Total input tokens:                      4192256
Total generated tokens:                  2097152
Request throughput (req/s):              11.52
Output token throughput (tok/s):         11799.21
Peak output token throughput (tok/s):    18098.00
Peak concurrent requests:                2048.00
Total Token throughput (tok/s):          35386.10
---------------Time to First Token----------------
Mean TTFT (ms):                          57924.67
Median TTFT (ms):                        49533.53
P99 TTFT (ms):                           118626.18
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          76.15
Median TPOT (ms):                        78.87
P99 TPOT (ms):                           83.38
---------------Inter-token Latency----------------
Mean ITL (ms):                           76.16
Median ITL (ms):                         60.18
P99 ITL (ms):                            181.91
==================================================
```

### gsm8k

#### Baseline
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" --enforce-eager 2>&1 | tee /tmp/fi.log

local-completions (model=deepseek-ai/DeepSeek-V2-Lite-Chat,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=256), gen_kwargs: (None), limit: None, num_fewshot: 8, batch_size: 1
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.6520|±  |0.0131|
|     |       |strict-match    |     8|exact_match|↑  |0.6482|±  |0.0132|

#### This PR
```
CUDA_VISIBLE_DEVICES=0 VLLM_LOGGING_LEVEL=DEBUG VLLM_ATTENTION_BACKEND=FLASHINFER_MLA VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL=1 vllm serve --model="deepseek-ai/DeepSeek-V2-Lite-Chat" --enforce-eager 2>&1 | tee /tmp/ragged.log

local-completions (model=deepseek-ai/DeepSeek-V2-Lite-Chat,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=256), gen_kwargs: (None), limit: None, num_fewshot: 8, batch_size: 1
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     8|exact_match|↑  |0.655|±  |0.0131|
|     |       |strict-match    |     8|exact_match|↑  |0.649|±  |0.0131|

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

See [26397.md](./26397.md) and [26397.json](./26397.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/26397
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/26397.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `0588fbbfcd7d27235b47c0cf72daf19ffb4f31f540db88db3a4d067aa017e4fd`

Existing semantic notes in [26397.sem.json](./26397.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/envs.py`
- `vllm/v1/attention/backends/mla/common.py`

### Diff

```diff
diff --git a/vllm/envs.py b/vllm/envs.py
index e91d8d033211..b5c4889c8ce7 100755
--- a/vllm/envs.py
+++ b/vllm/envs.py
@@ -183,6 +183,7 @@
     VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB: int | None = None
     VLLM_NIXL_ABORT_REQUEST_TIMEOUT: int = 480
     VLLM_USE_CUDNN_PREFILL: bool = False
+    VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL: bool = False
     VLLM_ENABLE_CUDAGRAPH_GC: bool = False
     VLLM_LOOPBACK_IP: str = ""
     VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE: bool = False
@@ -1250,6 +1251,10 @@ def get_vllm_port() -> int | None:
     "VLLM_USE_CUDNN_PREFILL": lambda: bool(
         int(os.getenv("VLLM_USE_CUDNN_PREFILL", "0"))
     ),
+    # Controls whether to use TRT-LLM ragged DeepSeek prefill
+    "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL": lambda: bool(
+        int(os.getenv("VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL", "0"))
+    ),
     # If set to 1/True, use the TRTLLM attention backend in flashinfer.
     # If set to 0/False, use the default attention backend in flashinfer.
     # If not set, auto-detect the attention backend in flashinfer.
@@ -1481,6 +1486,7 @@ def compute_hash() -> str:
         "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS",
         "VLLM_USE_FLASHINFER_MOE_MXFP4_BF16",
         "VLLM_USE_CUDNN_PREFILL",
+        "VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL",
         "VLLM_USE_TRTLLM_ATTENTION",
         "VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION",
         "VLLM_ROCM_USE_AITER",
diff --git a/vllm/v1/attention/backends/mla/common.py b/vllm/v1/attention/backends/mla/common.py
index 51a9032f4269..b920fd929e85 100755
--- a/vllm/v1/attention/backends/mla/common.py
+++ b/vllm/v1/attention/backends/mla/common.py
@@ -371,6 +371,7 @@ class ChunkedContextMetadata:
     query_start_loc: torch.Tensor
     max_query_len: int
     chunked_context: ChunkedContextMetadata | None = None
+    query_seq_lens: torch.Tensor | None = None
 
 
 @dataclass
@@ -386,7 +387,6 @@ class CudnnPrefillMetadata(MLACommonPrefillMetadata):
     class ChunkedContextMetadata(MLACommonPrefillMetadata.ChunkedContextMetadata):
         seq_lens: torch.Tensor
 
-    query_seq_lens: torch.Tensor | None = None
     cudnn_workspace: torch.Tensor | None = None
 
 
@@ -457,6 +457,7 @@ def use_flashinfer_prefill() -> bool:
         not envs.VLLM_DISABLE_FLASHINFER_PREFILL
         and flashinfer_available
         and not envs.VLLM_USE_CUDNN_PREFILL
+        and not envs.VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL
         and current_platform.is_device_capability(100)
     )
 
@@ -470,6 +471,15 @@ def use_cudnn_prefill() -> bool:
     )
 
 
+def use_trtllm_ragged_deepseek_prefill() -> bool:
+    """Check if TRT-LLM ragged DeepSeek prefill should be used."""
+    return (
+        flashinfer_available
+        and envs.VLLM_USE_TRTLLM_RAGGED_DEEPSEEK_PREFILL
+        and current_platform.is_device_capability(100)
+    )
+
+
 # Currently 394MB, this can be tuned based on GEMM sizes used.
 # Chosen to be the same as sglang:
 #  https://github.com/sgl-project/sglang/blob/766392c6bda2558b61ce6d1c1bfd8081a549e1f1/python/sglang/global_config.py#L37
@@ -593,6 +603,7 @@ def __init__(
 
         self._use_cudnn_prefill = use_cudnn_prefill()
         self._use_fi_prefill = use_flashinfer_prefill()
+        self._use_trtllm_ragged_prefill = use_trtllm_ragged_deepseek_prefill()
         self.prefill_metadata_cls = (
             FlashInferPrefillMetadata
             if self._use_fi_prefill
@@ -613,6 +624,11 @@ def __init__(
                 get_per_layer_parameters(vllm_config, layer_names, MLACommonImpl)
             )
 
+        if self._use_trtllm_ragged_prefill:
+            self._workspace_buffer = torch.empty(
+                FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=device
+            )
+
         if self._use_cudnn_prefill:
             self.cudnn_workspace = torch.empty(
                 CUDNN_WORKSPACE_SIZE * scheduler_config.max_num_seqs,
@@ -934,6 +950,11 @@ def build(
                 )
                 prefill_metadata.cudnn_workspace = self.cudnn_workspace
 
+            if self._use_trtllm_ragged_prefill:
+                prefill_metadata.query_seq_lens = (
+                    prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
+                )
+
         decode_metadata = None
         if num_decodes > 0:
             decode_metadata = self._build_decode(
@@ -1230,6 +1251,13 @@ def __init__(self, *args, **kwargs) -> None:
             self._run_prefill_context_chunk = self._run_prefill_context_chunk_fi
             self._run_prefill_new_tokens = self._run_prefill_new_tokens_fi
             self._pad_v = False
+        elif use_trtllm_ragged_deepseek_prefill():
+            logger.debug_once("Using TRT-LLM ragged DeepSeek prefill for MLA")
+            self._run_prefill_context_chunk = (
+                self._run_prefill_context_chunk_trtllm_ragged
+            )
+            self._run_prefill_new_tokens = self._run_prefill_new_tokens_trtllm_ragged
+            self._pad_v = False
         elif use_cudnn_prefill():
             logger.debug_once("Using CUDNN prefill for MLA")
             self._run_prefill_context_chunk = self._run_prefill_context_chunk_cudnn
@@ -1326,6 +1354,7 @@ def _run_prefill_new_tokens_fi(
     ):
         assert isinstance(prefill, FlashInferPrefillMetadata)
         assert prefill.prefill_main is not None
+
         ret = prefill.prefill_main.run(
             q=q,
             k=k,
@@ -1334,7 +1363,6 @@ def _run_prefill_new_tokens_fi(
         )
 
         if isinstance(ret, tuple):
-            # Convert from (q_len, num_heads) to (num_heads, q_len)
             return ret[0], ret[1].transpose(0, 1).contiguous()
         return ret
 
@@ -1384,12 +1412,14 @@ def _run_prefill_context_chunk_fi(
         self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v
     ):
         assert isinstance(prefill, FlashInferPrefillMetadata)
+
         attn_out, lse = prefill.prefill_chunks[chunk_idx].run(
             q=q,
             k=k,
             v=v,
             return_lse=True,
         )
+
         # Convert from (q_len, num_heads) to (num_heads, q_len)
         return attn_out, lse.transpose(0, 1).contiguous()
 
@@ -1418,6 +1448,81 @@ def _run_prefill_context_chunk_cudnn(
             is_cuda_graph_compatible=True,
         )
 
+    def _run_prefill_new_tokens_trtllm_ragged(
+        self, prefill: MLACommonPrefillMetadata, q, k, v, return_softmax_lse
+    ):
+        """TRT-LLM ragged attention for new tokens (causal)."""
+        from flashinfer.prefill import trtllm_ragged_attention_deepseek
+
+        assert prefill.query_seq_lens is not None
+
+        ret = trtllm_ragged_attention_deepseek(
+            query=q,
+            key=k,
+            value=v,
+            workspace_buffer=self._workspace_buffer,
+            seq_lens=prefill.query_seq_lens,
+            max_q_len=prefill.max_query_len,
+            max_kv_len=prefill.max_query_len,
+            bmm1_scale=self.scale,
+            bmm2_scale=1.0,
+            o_sf_scale=1.0,
+            batch_size=prefill.query_seq_lens.shape[0],
+            window_left=-1,
+            cum_seq_lens_q=prefill.query_start_loc,
+            cum_seq_lens_kv=prefill.query_start_loc,
+            enable_pdl=False,
+            is_causal=True,
+            return_lse=return_softmax_lse,
+        )
+
+        if isinstance(ret, tuple):
+            # Convert from (q_len, num_heads) to (num_heads, q_len)
+            return ret[0], ret[1].transpose(0, 1).contiguous()
+        return ret
+
+    def _run_prefill_context_chunk_trtllm_ragged(
+        self, prefill: MLACommonPrefillMetadata, chunk_idx: int, q, k, v
+    ):
+        """TRT-LLM ragged attention for context chunks (non-causal)."""
+        from flashinfer.prefill import trtllm_ragged_attention_deepseek
+
+        assert prefill.chunked_context is not None
+        assert prefill.chunked_context.seq_lens[chunk_idx] is not None
+
+        out = torch.zeros(
+            q.shape[0],
+            q.shape[1],
+            v.shape[2],
+            device=q.device,
+            dtype=q.dtype,
+        )
+        self._workspace_buffer.fill_(0)
+
+        attn_out, lse = trtllm_ragged_attention_deepseek(
+            query=q,
+            key=k,
+            value=v,
+            workspace_buffer=self._workspace_buffer,
+            seq_lens=prefill.chunked_context.seq_lens[chunk_idx],
+            max_q_len=prefill.max_query_len,
+            max_kv_len=prefill.chunked_context.max_seq_lens[chunk_idx],
+            bmm1_scale=self.scale,
+            bmm2_scale=1.0,
+            o_sf_scale=1.0,
+            batch_size=prefill.chunked_context.seq_lens[chunk_idx].shape[0],
+            window_left=-1,
+            cum_seq_lens_q=prefill.query_start_loc,
+            cum_seq_lens_kv=prefill.chunked_context.cu_seq_lens[chunk_idx],
+            enable_pdl=False,
+            is_causal=False,
+            return_lse=True,
+            out=out,
+        )
+
+        # Convert from (q_len, num_heads) to (num_heads, q_len)
+        return attn_out, lse.transpose(0, 1).contiguous()
+
     def process_weights_after_loading(self, act_dtype: torch.dtype):
         def get_layer_weight(layer):
             WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
```
