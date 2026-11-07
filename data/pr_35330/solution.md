# PR #35330: [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement

- field: scheduling  | area: entrypoints_pooling
- author: yewentao256
- state: MERGED  created: 2026-02-25T20:41:55Z  merged: 2026-02-26T17:14:54Z
- url: https://github.com/vllm-project/vllm/pull/35330
- labels: frontend, ready

# Issue

## Problem

ColBERT MaxSim score computation for pooling/rerank requests was done one pair at a time on the CPU. For each (query, document) embedding pair the server called compute_maxsim_score, running a separate matmul and amax/sum per pair on CPU tensors. This serialized the work and used slow CPU compute inside the API server's post-processing stage, bottlenecking end-to-end throughput.

## Example

Rerank benchmark with jinaai/jina-colbert-v2, 2000 prompts, max-concurrency 64, random input len 2048: main runs 14.15s (141.38 req/s, mean E2EL 446ms); optimized runs 12.41s (161.14 req/s, mean E2EL 391ms) — 13.9% E2E throughput gain.

## Use case

Online scoring/reranking endpoints (/v1/rerank, /score) served by pooling models producing per-token late-interaction (ColBERT) embeddings.

## Resources

CPU compute in the API server post-processing path; the per-pair loop left the GPU idle and serialized small matmuls on CPU, making it the latency bottleneck.

## Type of users

Users serving ColBERT / late-interaction pooling models for reranking at high request concurrency.

## Kind of inputs

Batches of many (query, document) embedding pairs with variable query_len and doc_len, each shaped [tokens, dim]; large request counts (e.g. 2000 prompts, concurrency 64, input len ~2048).

## Mechanism with the issue

In serving.py _late_interaction_score, a Python for-loop iterated emb pairs calling compute_maxsim_score(q_emb, d_emb) in utils.py, each computing token_scores = q @ d.T then amax(dim=-1).sum() on CPU tensors. Per-pair invocation meant no batching, repeated tiny-tensor ops, and no GPU use.

# Solution

## Fix in words

Adds compute_maxsim_scores in utils.py that processes pairs in padded mini-batches on GPU. It groups up to max_batch_size (16) pairs, pads queries/docs to max_q/max_d with zero tensors plus boolean masks, and shrinks the batch when (batch*max_q*max_d) exceeds max_score_matrix_elements (16M) to bound allocation. One torch.bmm computes all token-score matrices; padded doc columns are masked with -inf before amax, padded query rows zeroed before sum, then results moved to CPU. serving.py replaces the per-pair compute_maxsim_score loop with a single compute_maxsim_scores call over all embeddings, then zips the precomputed scores into outputs.

## Author's solution

Scores were originally computed on CPU per pair; the author moves the computation to GPU and uses a batched version, giving large performance improvement. Memory is bounded via max_score_matrix_elements=16M and batch_size=16, yielding ~40MB-200MB GPU usage, so OOM is not a concern.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/entrypoints/pooling/score/test_utils.py`
- `vllm/entrypoints/pooling/score/serving.py`
- `vllm/entrypoints/pooling/score/utils.py`

### Diff
```diff
diff --git a/tests/entrypoints/pooling/score/test_utils.py b/tests/entrypoints/pooling/score/test_utils.py
index d69da822dd06..e5e1fd606845 100644
--- a/tests/entrypoints/pooling/score/test_utils.py
+++ b/tests/entrypoints/pooling/score/test_utils.py
@@ -4,10 +4,15 @@
 from unittest.mock import patch
 
 import pytest
+import torch
 
 from vllm.config import ModelConfig
 from vllm.entrypoints.chat_utils import ChatTemplateResolutionError
-from vllm.entrypoints.pooling.score.utils import get_score_prompt
+from vllm.entrypoints.pooling.score.utils import (
+    compute_maxsim_score,
+    compute_maxsim_scores,
+    get_score_prompt,
+)
 from vllm.inputs import TokensPrompt
 from vllm.tokenizers import get_tokenizer
 
@@ -349,3 +354,36 @@ def test_post_process_tokens_called(
         assert_prompt_tokenization_consistent(
             cross_encoder_tokenizer, full_prompt, engine_prompt
         )
+
+
+def test_compute_maxsim_scores_matches_reference_per_pair() -> None:
+    generator = torch.Generator()
+    generator.manual_seed(7)
+
+    shared_query = torch.randn(5, 8, generator=generator)
+    q_embs = [
+        shared_query,  # 1:N style shared query
+        shared_query,
+        torch.randn(2, 8, generator=generator),
+        torch.randn(4, 8, generator=generator),
+    ]
+    d_embs = [
+        torch.randn(6, 8, generator=generator),
+        torch.randn(3, 8, generator=generator),
+        torch.randn(5, 8, generator=generator),
+        torch.randn(7, 8, generator=generator),
+    ]
+
+    batched_scores = compute_maxsim_scores(
+        q_embs,
+        d_embs,
+        max_batch_size=4,
+        max_score_matrix_elements=40,  # batch shrinking path.
+    )
+    reference_scores = [
+        compute_maxsim_score(q, d).to("cpu") for q, d in zip(q_embs, d_embs)
+    ]
+
+    assert len(batched_scores) == len(reference_scores)
+    for batched, reference in zip(batched_scores, reference_scores):
+        torch.testing.assert_close(batched, reference, rtol=1e-4, atol=1e-4)
diff --git a/vllm/entrypoints/pooling/score/serving.py b/vllm/entrypoints/pooling/score/serving.py
index 3fe18ca8b3a5..aec6e909d161 100644
--- a/vllm/entrypoints/pooling/score/serving.py
+++ b/vllm/entrypoints/pooling/score/serving.py
@@ -31,7 +31,7 @@
     ScoreInputs,
     _cosine_similarity,
     compress_token_type_ids,
-    compute_maxsim_score,
+    compute_maxsim_scores,
     get_score_prompt,
     parse_score_data_single,
     validate_score_input,
@@ -311,19 +311,17 @@ async def _late_interaction_score(
         # Compute MaxSim scores
         from vllm.outputs import PoolingOutput
 
+        maxsim_scores = compute_maxsim_scores(
+            [emb.outputs.data for emb in emb_data_1],
+            [emb.outputs.data for emb in emb_data_2],
+        )
+
         scores: list[PoolingRequestOutput] = []
         padding: list[int] = []
         if (pad_token_id := tokenizer.pad_token_id) is not None:
             padding = [pad_token_id]
 
-        for emb_1, emb_2 in zip(emb_data_1, emb_data_2):
-            # emb_1.outputs.data: [query_len, dim]
-            # emb_2.outputs.data: [doc_len, dim]
-            q_emb = emb_1.outputs.data
-            d_emb = emb_2.outputs.data
-
-            maxsim_score = compute_maxsim_score(q_emb, d_emb)
-
+        for emb_1, emb_2, maxsim_score in zip(emb_data_1, emb_data_2, maxsim_scores):
             tokens = emb_1.prompt_token_ids + padding + emb_2.prompt_token_ids
 
             scores.append(
diff --git a/vllm/entrypoints/pooling/score/utils.py b/vllm/entrypoints/pooling/score/utils.py
index 60e71ff73953..98c24856bf3c 100644
--- a/vllm/entrypoints/pooling/score/utils.py
+++ b/vllm/entrypoints/pooling/score/utils.py
@@ -1,6 +1,6 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from collections.abc import Iterable
+from collections.abc import Iterable, Sequence
 from typing import Any, TypeAlias, cast
 
 import torch
@@ -53,6 +53,82 @@ def compute_maxsim_score(q_emb: torch.Tensor, d_emb: torch.Tensor) -> torch.Tens
     return token_scores.amax(dim=-1).sum()
 
 
+def compute_maxsim_scores(
+    q_embs: Sequence[torch.Tensor],
+    d_embs: Sequence[torch.Tensor],
+    max_batch_size: int = 16,
+    max_score_matrix_elements: int = 16_000_000,
+) -> list[torch.Tensor]:
+    """Compute ColBERT MaxSim scores in padded mini-batches."""
+    if len(q_embs) != len(d_embs):
+        raise ValueError("q_embs and d_embs must have the same length")
+
+    num_pairs = len(q_embs)
+    if num_pairs == 0:
+        return []
+
+    for q_emb, d_emb in zip(q_embs, d_embs):
+        if q_emb.ndim != 2 or d_emb.ndim != 2:
+            raise ValueError("Each embedding tensor must be 2-D")
+        if q_emb.shape[1] != d_emb.shape[1]:
+            raise ValueError("Query and document embeddings must have same dim")
+
+    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+    scores: list[torch.Tensor] = []
+    start = 0
+    while start < num_pairs:
+        end = min(start + max_batch_size, num_pairs)
+        max_q = max(int(x.shape[0]) for x in q_embs[start:end])
+        max_d = max(int(x.shape[0]) for x in d_embs[start:end])
+
+        # keep score matrix bounded to avoid oversized allocations.
+        while (
+            end - start > 1
+            and (end - start) * max_q * max_d > max_score_matrix_elements
+        ):
+            end -= 1
+            max_q = max(int(x.shape[0]) for x in q_embs[start:end])
+            max_d = max(int(x.shape[0]) for x in d_embs[start:end])
+
+        batch_q = q_embs[start:end]
+        batch_d = d_embs[start:end]
+        batch_size = end - start
+        dim = int(batch_q[0].shape[1])
+        dtype = batch_q[0].dtype
+
+        q_batch = torch.zeros(
+            (batch_size, max_q, dim), dtype=dtype, device=compute_device
+        )
+        d_batch = torch.zeros(
+            (batch_size, max_d, dim), dtype=dtype, device=compute_device
+        )
+        q_mask = torch.zeros(
+            (batch_size, max_q), dtype=torch.bool, device=compute_device
+        )
+        d_mask = torch.zeros(
+            (batch_size, max_d), dtype=torch.bool, device=compute_device
+        )
+
+        # copy to padded tensors
+        for i, (q_emb, d_emb) in enumerate(zip(batch_q, batch_d)):
+            q_len = int(q_emb.shape[0])
+            d_len = int(d_emb.shape[0])
+            q_batch[i, :q_len] = q_emb.to(device=compute_device, dtype=dtype)
+            d_batch[i, :d_len] = d_emb.to(device=compute_device, dtype=dtype)
+            q_mask[i, :q_len] = True
+            d_mask[i, :d_len] = True
+
+        token_scores = torch.bmm(q_batch, d_batch.transpose(1, 2))
+        token_scores.masked_fill_(~d_mask.unsqueeze(1), float("-inf"))
+        max_per_query = token_scores.amax(dim=-1)
+        max_per_query.masked_fill_(~q_mask, 0)
+        batch_scores = max_per_query.sum(dim=-1).to("cpu")
+        scores.extend(batch_scores.unbind(0))
+        start = end
+
+    return [cast(torch.Tensor, score) for score in scores]
+
+
 class ScoreMultiModalParam(TypedDict, total=False):
     """
     A specialized parameter type for scoring multimodal content

```
