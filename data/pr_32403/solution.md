# PR #32403: [Performance] Improve Triton prefill attention kernel's performance  (#32403)

# Issue

## Original PR description (captured source)

<!-- markdownlint-disable -->

## Purpose
Refine prefill attention kernel with FA2 implementation based on https://github.com/ModelTC/LightLLM/blob/2fbd2b8a7b180570129bd539bb34cbe0a5dbb22a/lightllm/common/basemodel/triton_kernel/att/prefill_att/context_flashattention_nopad.py#L287-L288
- In previous implementation, we use `float("-inf")` in attention mask, which needs additional invalid values mask for SWA. This PR replaces it with extremely large values $-10^8$ to overflow.
- Pre-scale `sm_scale` with $\frac{1}{ln(2)}$, then we can fuse `triton.math.exp` with faster `triton.math.exp2`

## Test Plan
Benchmark with https://github.com/noooop/snippet/blob/main/benchmarks/triton_attention/v1_offline_triton.py

## Test Result
**Main branch**
```
Batchsize 8, Input_len 2048 Throughput: 33.9996 requests/s, 69631.2040 tokens/s, Latency 113.12 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 34.4612 requests/s, 70576.6036 tokens/s, Latency 207.27 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 34.8346 requests/s, 71341.2651 tokens/s, Latency 358.84 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 34.8565 requests/s, 71386.2010 tokens/s, Latency 478.15 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 34.7642 requests/s, 71197.0056 tokens/s, Latency 719.13 ms, n_step 4
```

**Main branch + exp2 fusion**
```
Batchsize 8, Input_len 2048 Throughput: 34.4454 requests/s, 70544.1212 tokens/s, Latency 111.66 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 34.7209 requests/s, 71108.4363 tokens/s, Latency 205.72 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 35.1655 requests/s, 72018.9614 tokens/s, Latency 355.46 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 35.2775 requests/s, 72248.3425 tokens/s, Latency 472.44 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 34.9167 requests/s, 71509.3940 tokens/s, Latency 715.99 ms, n_step 4
```

**PR**
```
Batchsize 8, Input_len 2048 Throughput: 34.9690 requests/s, 71616.4340 tokens/s, Latency 109.99 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 35.0962 requests/s, 71876.9206 tokens/s, Latency 203.52 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 35.5592 requests/s, 72825.2077 tokens/s, Latency 351.53 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 35.5579 requests/s, 72822.6684 tokens/s, Latency 468.72 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 35.2918 requests/s, 72277.6372 tokens/s, Latency 708.38 ms, n_step 4
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [32403.md](./32403.md) and [32403.json](./32403.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/32403
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/32403.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `673f0aebb1b7e7e9ecff7c034fad5563754369c3fd943ae66aa9be2cc7b1b2d1`

Existing semantic notes in [32403.sem.json](./32403.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `tests/models/language/pooling/test_token_classification.py`
- `vllm/utils/math_utils.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`

### Diff

```diff
diff --git a/tests/models/language/pooling/test_token_classification.py b/tests/models/language/pooling/test_token_classification.py
index 7bc8daaea650..91a9dd6a75af 100644
--- a/tests/models/language/pooling/test_token_classification.py
+++ b/tests/models/language/pooling/test_token_classification.py
@@ -46,7 +46,7 @@ def test_bert_models(
     for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
         hf_output = hf_output.detach().clone().cpu().float()
         vllm_output = vllm_output.detach().clone().cpu().float()
-        torch.testing.assert_close(hf_output, vllm_output, atol=1.2e-2, rtol=1e-3)
+        torch.testing.assert_close(hf_output, vllm_output, atol=3.2e-2, rtol=1e-3)
 
 
 @pytest.mark.parametrize("model", ["disham993/electrical-ner-ModernBERT-base"])
@@ -86,7 +86,7 @@ def test_modernbert_models(
     for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
         hf_output = hf_output.detach().clone().cpu().float()
         vllm_output = vllm_output.detach().clone().cpu().float()
-        torch.testing.assert_close(hf_output, vllm_output, atol=1.2e-2, rtol=1e-3)
+        torch.testing.assert_close(hf_output, vllm_output, atol=3.2e-2, rtol=1e-3)
 
 
 @pytest.mark.parametrize("model", ["bd2lcco/Qwen3-0.6B-finetuned"])
diff --git a/vllm/utils/math_utils.py b/vllm/utils/math_utils.py
index bdfa5fd2cbcb..5fc6c3d664f0 100644
--- a/vllm/utils/math_utils.py
+++ b/vllm/utils/math_utils.py
@@ -2,6 +2,10 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Math utility functions for vLLM."""
 
+# Approximate value of 1/ln(2), used for log/exp base conversion
+# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
+RCP_LN2 = 1.4426950216
+
 
 def cdiv(a: int, b: int) -> int:
     """Ceiling division."""
diff --git a/vllm/v1/attention/ops/triton_prefill_attention.py b/vllm/v1/attention/ops/triton_prefill_attention.py
index 046d0c1707a9..e9b123fa0d8a 100644
--- a/vllm/v1/attention/ops/triton_prefill_attention.py
+++ b/vllm/v1/attention/ops/triton_prefill_attention.py
@@ -30,6 +30,7 @@
 
 from vllm.platforms import current_platform
 from vllm.triton_utils import tl, triton
+from vllm.utils.math_utils import RCP_LN2
 
 
 @triton.jit
@@ -110,15 +111,7 @@ def _fwd_kernel(
     end_n_limit = block_mask * end_n
 
     for start_n in range(start_n_limit, end_n_limit, BLOCK_N):
-        start_n = tl.multiple_of(start_n, BLOCK_N)
-        # -- compute qk ----
-        k = tl.load(
-            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
-            mask=((start_n + offs_n[None, :]) < cur_batch_seq_len) & (mask_d[:, None]),
-            other=0.0,
-        )
-
-        # Apply attention mask (causal + bidirectional sliding window)
+        # -- prepare attention mask ----
         # Position indices in the sequence
         pos_q = offs_m[:, None]  # Query positions [BLOCK_M, 1]
         pos_k = start_n + offs_n[None, :]  # Key positions [1, BLOCK_N]
@@ -141,53 +134,38 @@ def _fwd_kernel(
         if sliding_mask_k is not None:
             mask &= sliding_mask_k
 
-        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
-        qk += tl.where(mask, 0, float("-inf"))
-        qk += tl.dot(q, k)
-        qk *= sm_scale
-
-        # -- compute m_ij, p, l_ij
-        m_ij = tl.max(qk, 1)
-        # For sliding window there's a chance the max is -inf due to masking of
-        # the entire row. In this case we need to set m_j 0 to avoid NaN
-        m_ij_valid_mask = m_ij > float("-inf")
-        m_ij_masked = tl.where(m_ij_valid_mask, m_ij, 0.0)
-        # -- compute p and l_ij --
-        p = tl.exp(qk - m_ij_masked[:, None])
+        start_n = tl.multiple_of(start_n, BLOCK_N)
+        # -- compute qk ----
+        k = tl.load(
+            k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
+            mask=(pos_k < cur_batch_seq_len) & (mask_d[:, None]),
+            other=0.0,
+        )
+
+        qk = tl.dot(q, k)
+        qk = tl.where(mask, qk * sm_scale, -1.0e8)
+        m_ij = tl.maximum(m_i, tl.max(qk, 1))
+        qk -= m_ij[:, None]
+        p = tl.math.exp2(qk)
         l_ij = tl.sum(p, 1)
+
         # -- update m_i and l_i
-        m_i_new = tl.maximum(m_i, m_ij)
-        m_i_new_mask = m_i_new > float("-inf")
-        alpha = tl.exp(m_i - m_i_new)
-        beta = tl.exp(m_ij - m_i_new)
-        # mask alpha and beta for sliding window
-        alpha = tl.where(m_i_new_mask, alpha, 1.0)
-        beta = tl.where(m_i_new_mask, beta, 0.0)
-        l_i_new = alpha * l_i + beta * l_ij
+        alpha = tl.math.exp2(m_i - m_ij)
+        l_i = l_i * alpha + l_ij
         # -- update output accumulator --
-        # scale p
-        # For sliding window there's a chance the l_i_new is 0 due to masking
-        # the entire row. We need to set l_i_new 1 to avoid zero division
-        l_i_new_mask = (l_i_new != 0.0) & (m_i_new_mask > float("-inf"))
-        l_i_new_safe = tl.where(l_i_new_mask, l_i_new, 1.0)
-        p_scale = beta / l_i_new_safe
-        p = p * p_scale[:, None]
-        # scale acc
-        acc_scale = l_i / l_i_new_safe * alpha
-        acc = acc * acc_scale[:, None]
+        acc = acc * alpha[:, None]
         # update acc
         v = tl.load(
             v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
             mask=((start_n + offs_n[:, None]) < cur_batch_seq_len) & (mask_d[None, :]),
             other=0.0,
         )
-
         p = p.to(v.dtype)
-        acc += tl.dot(p, v)
-        # update m_i and l_i
-        l_i = l_i_new
-        m_i = m_i_new
-    # initialize pointers to output
+        acc = tl.dot(p, v, acc)
+        # update m_i
+        m_i = m_ij
+
+    acc = acc / l_i[:, None]
     off_o = (
         (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
         + cur_head * stride_oh
@@ -234,6 +212,9 @@ def context_attention_fwd(
     Lq, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
 
     sm_scale = 1.0 / (Lq**0.5) if softmax_scale is None else softmax_scale
+    # rescale with 1/ln(2) for triton exp2
+    sm_scale *= RCP_LN2
+
     batch, head = b_seq_len.shape[0], q.shape[1]
     kv_group_num = q.shape[1] // k.shape[1]
 
```
