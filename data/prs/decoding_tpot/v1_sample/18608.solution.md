# PR #18608: [V1][Sampler] Improve performance of FlashInfer sampling by sampling logits instead of probs

- field: decoding_tpot  | area: v1_sample
- author: lgeiger
- state: MERGED  created: 2025-05-23T11:41:47Z  merged: 2025-05-26T15:49:37Z
- url: https://github.com/vllm-project/vllm/pull/18608
- labels: ready, v1

# Issue

## Problem

The FlashInfer top-k/top-p sampling path always computed a full softmax over logits to produce a probability tensor, then passed those probs into `top_k_top_p_sampling_from_probs`. That FlashInfer entry point internally does `softmax -> top_k_renorm_probs -> top_p_sampling_from_probs`, so vLLM paid for an extra explicit softmax plus a probability-space top-k renormalization step. The combined-filter path is slower than necessary because renormalizing probabilities is more expensive than masking logits directly.

## Example

On an L40s GPU micro-benchmark with batch_size=40 and num_classes=262208 (float), the probs-based path is ~9.3% slower than the logits-based path for combined top-k=64 / top-p=0.95 sampling.

## Use case

Autoregressive decode in vLLM V1 where every generated token runs the sampler with top-k and top-p filtering enabled. This executes once per decode step per request, directly on the TPOT critical path.

## Resources

GPU compute and kernel-launch overhead on the sampling kernels; the redundant softmax and prob-space renorm add extra GPU work each decode step.

## Type of users

Users serving models with FlashInfer-backed sampling on CUDA GPUs who apply top-k and top-p simultaneously, e.g. serving `google/gemma-3-12b-it` with large vocabularies.

## Kind of inputs

Decode requests where both `k` and `p` are set (combined top-k + top-p), large vocab size (~262k classes), typical serving batch sizes.

## Mechanism with the issue



# Solution

## Fix in words

The diff stops pre-computing `probs = logits.softmax(...)` at the top of `forward_cuda` and instead passes raw `logits` into `flashinfer_sample`. In the combined top-k + top-p branch, it swaps `top_k_top_p_sampling_from_probs(probs, ...)` for `top_k_top_p_sampling_from_logits(logits, ...)`, which internally does `top_k_mask_logits -> softmax -> top_p_sampling_from_probs`, avoiding the slower prob-space top-k renormalization. The top-p-only and top-k-only branches still softmax locally (only when actually taken), and the `k is None and p is None` random_sample path computes its softmax inline. Net effect: masking in logit space replaces renorm in probability space, removing redundant work.

## Author's solution

Replace `flashinfer.sampling.top_k_top_p_sampling_from_probs` with `top_k_top_p_sampling_from_logits`. The probs path runs `softmax -> top_k_renorm_probs -> top_p_sampling_from_probs`, while the logits path runs the faster `top_k_mask_logits -> softmax -> top_p_sampling_from_probs`. Author measured 9.3% speedup on an L40s micro-benchmark and 1.75% end-to-end throughput improvement for `google/gemma-3-12b-it`.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/sample/ops/topk_topp_sampler.py`

### Diff
```diff
diff --git a/vllm/v1/sample/ops/topk_topp_sampler.py b/vllm/v1/sample/ops/topk_topp_sampler.py
index 5d8b3f423b02..4a5fbb10d408 100644
--- a/vllm/v1/sample/ops/topk_topp_sampler.py
+++ b/vllm/v1/sample/ops/topk_topp_sampler.py
@@ -89,18 +89,18 @@ def forward_cuda(
         p: Optional[torch.Tensor],
     ) -> torch.Tensor:
         """More optimized implementation for top-k and top-p sampling."""
-        probs = logits.softmax(dim=-1, dtype=torch.float32)
         if k is None and p is None:
             # We prefer `random_sample` over `flashinfer_sample` when sorting is
             # not needed. This is because `random_sample` does not require
             # CPU-GPU synchronization while `flashinfer_sample` does.
+            probs = logits.softmax(dim=-1, dtype=torch.float32)
             return random_sample(probs, generators)
         if generators:
             logger.warning("FlashInfer 0.2.3+ does not support "
                            "per-request generators. Falling back to "
                            "PyTorch-native implementation.")
             return self.forward_native(logits, generators, k, p)
-        return flashinfer_sample(probs, k, p, generators)
+        return flashinfer_sample(logits, k, p, generators)
 
     def forward_tpu(
         self,
@@ -254,17 +254,17 @@ def random_sample(
 
 
 def flashinfer_sample(
-    probs: torch.Tensor,
+    logits: torch.Tensor,
     k: Optional[torch.Tensor],
     p: Optional[torch.Tensor],
     generators: dict[int, torch.Generator],
 ) -> torch.Tensor:
-    """Sample from the probabilities using FlashInfer.
+    """Sample from the logits using FlashInfer.
 
     Statistically, this function is equivalent to the `random_sample` function.
     However, this function is faster because it avoids sorting the logits tensor
     via rejection sampling.
-    
+
     NOTE: The outputs of this function do not necessarily match the outputs of
     the `random_sample` function. It only guarantees that the outputs are
     statistically equivalent.
@@ -274,18 +274,19 @@ def flashinfer_sample(
     the synchronization overhead.
     """
     assert not (k is None and p is None)
-
     if k is None:
         # Top-p only.
+        probs = logits.softmax(dim=-1, dtype=torch.float32)
         next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(
             probs, p, deterministic=True)
     elif p is None:
         # Top-k only.
+        probs = logits.softmax(dim=-1, dtype=torch.float32)
         next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(
             probs, k, deterministic=True)
     else:
         # Both top-k and top-p.
-        next_token_ids = (flashinfer.sampling.top_k_top_p_sampling_from_probs(
-            probs, k, p, deterministic=True))
+        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(
+            logits, k, p, deterministic=True)
 
     return next_token_ids.view(-1)

```
