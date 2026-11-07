# PR #29826: [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor

- field: decoding_tpot  | area: v1_sample
- author: jthomson04
- state: MERGED  created: 2025-12-02T00:38:03Z  merged: 2025-12-02T21:25:53Z
- url: https://github.com/vllm-project/vllm/pull/29826
- labels: ready, v1

# Issue

## Problem

MinTokensLogitsProcessor.apply writes a Python float (-float("inf")) into a device tensor via indexed assignment. This creates a scalar on the host and triggers a pageable host-to-device (HtoD) memory transfer on every decode step. Because pageable HtoD copies synchronize with the device, they block the CPU until the GPU reaches that point, destroying scheduler/execution overlap.

## Example

With --async-scheduling at low concurrency, including min_tokens in a request raises TPOT by up to 30% versus the same request without min_tokens.

## Use case

Online serving with async scheduling enabled, where the scheduler is meant to run ahead and overlap with model execution to hide latency.

## Resources

PCIe/HtoD bandwidth and, critically, CPU-GPU synchronization: the pageable copy forces a device sync that stalls kernel-launch/scheduler overlap.

## Type of users

Users running vLLM v1 with --async-scheduling at low concurrency who submit requests carrying min_tokens (min-length enforcement).

## Kind of inputs

Decode requests that specify min_tokens so the EOS/stop tokens must be inhibited until minimum length is reached; impact is largest at low batch/concurrency.

## Mechanism with the issue

In builtin.py MinTokensLogitsProcessor.apply, `logits[self.logits_slice] = -float("inf")` constructs a host scalar each step and copies it to the device via pageable memory. The pageable HtoD transfer is synchronous wrt the device, so under async scheduling the scheduler can no longer stay overlapped with model execution, inflating per-token decode latency.

# Solution

## Fix in words

Before: apply assigned a host-side `-float("inf")` into the device logits via `logits[self.logits_slice] = -float("inf")`, forcing a synchronizing pageable HtoD copy each step. After: the processor preallocates `self.neg_inf_tensor`, a scalar `-inf` tensor already on `self.device` in __init__, and applies it with `logits.index_put_(self.logits_slice, self.neg_inf_tensor)`. This removes the per-step host-to-device transfer and the implied device sync, restoring scheduler/execution overlap under async scheduling. A related MinPLogitsProcessor line was also switched to `logits.masked_fill_(invalid_token_mask, -float("inf"))` for the in-place masked write.

## Author's solution

Author identified the pageable HtoD transfer in MinTokensLogitsProcessor as a device-synchronizing operation that breaks scheduler overlap under --async-scheduling, causing up to 30% TPOT increase at low concurrency, backed by nsys traces. Fix avoids the transfer by using a preallocated device tensor for the -inf fill; per a review comment, the min-p path was converted to use `.masked_fill_(-float("inf"))`. Existing MinTokensLogitsProcessor tests deemed sufficient.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/sample/logits_processor/builtin.py`

### Diff
```diff
diff --git a/vllm/v1/sample/logits_processor/builtin.py b/vllm/v1/sample/logits_processor/builtin.py
index 4ee7dc2880c8..82743f72b031 100644
--- a/vllm/v1/sample/logits_processor/builtin.py
+++ b/vllm/v1/sample/logits_processor/builtin.py
@@ -110,7 +110,7 @@ def apply(self, logits: torch.Tensor) -> torch.Tensor:
         # Identify valid tokens using threshold comparison
         invalid_token_mask = probability_values < adjusted_min_p
         # Apply mask using boolean indexing
-        logits[invalid_token_mask] = -float("inf")
+        logits.masked_fill_(invalid_token_mask, -float("inf"))
         return logits
 
 
@@ -178,6 +178,10 @@ def __init__(
             self._device_tensor([], torch.int32),
         )
 
+        self.neg_inf_tensor = torch.tensor(
+            -float("inf"), dtype=torch.float32, device=self.device
+        )
+
     def is_argmax_invariant(self) -> bool:
         """By censoring stop tokens, min-tokens can change the outcome
         of the argmax operation in greedy sampling."""
@@ -229,7 +233,7 @@ def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
     def apply(self, logits: torch.Tensor) -> torch.Tensor:
         if self.min_toks:
             # Inhibit EOS token for requests which have not reached min length
-            logits[self.logits_slice] = -float("inf")
+            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
         return logits
 
 

```
