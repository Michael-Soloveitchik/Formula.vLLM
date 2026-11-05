# PR #12563: [Feature] Fix guided decoding blocking bitmask memcpy

- field: decoding_tpot  | area: model_executor_guided_decoding
- author: xpbowler
- state: MERGED  created: 2025-01-29T21:16:01Z  merged: 2025-01-31T23:37:30Z
- url: https://github.com/vllm-project/vllm/pull/12563
- labels: performance, structured-output, ready

# Issue

## Problem

In xgrammar guided decoding, the per-step token bitmask is moved to the GPU with a synchronous (blocking) host-to-device copy via `self.token_bitmask.to(scores.device)`. Because the copy is blocking, the CPU must wait on a `cudaStreamSynchronize` until the current decode work finishes before the transfer completes. This stalls the CPU and prevents it from pre-launching the downstream sampler kernels, serializing work that could overlap.

## Example

Profiling one decode step of Llama 3.1 8B on an H100 (Nsys) shows the CPU blocked on `cudaStreamSynchronize`, only enqueuing the sampling kernels after the bitmask is applied; removing the blocking copy eliminates that stall.

## Use case

Online serving / interactive generation where structured (guided/JSON/grammar) decoding is enabled and every decode step applies a token bitmask before sampling.

## Resources

PCIe host-to-device copy bandwidth plus CPU kernel-launch time; the blocking copy wastes the overlap window and raises per-token latency (TPOT).

## Type of users

Users running guided/structured decoding with the xgrammar backend, especially latency-sensitive low-batch or single-request serving on CUDA GPUs.

## Kind of inputs

Requests with a guided-decoding constraint (grammar/JSON schema) active, so a token bitmask is generated and copied to device every decode step; most impactful at small batch / single-request decode.

## Mechanism with the issue

In `xgrammar_decoding.py` `__call__`, the call `self.token_bitmask.to(scores.device)` issues a synchronous H2D transfer. A blocking `.to()` forces the CPU to sync on the CUDA stream, so it cannot run ahead and pre-launch the sampler kernels, breaking CPU/GPU overlap during decode.

# Solution

## Fix in words

The diff changes the bitmask device transfer from a blocking copy to an async one by passing `non_blocking=True`: `self.token_bitmask.to(scores.device)` becomes `self.token_bitmask.to(scores.device, non_blocking=True)`, still passed into `xgr.apply_token_bitmask_inplace(scores, ...)`. With the async copy, the CPU no longer stalls on `cudaStreamSynchronize` and can pre-launch the sampler kernels while the transfer proceeds. This restores CPU/GPU overlap during the decode step. Net effect: lower per-token latency and ~5% higher tok/s in single-request Llama 3.1 8B benchmarks.

## Author's solution

Author makes the host-to-device bitmask copy asynchronous by setting `non_blocking=True` on the `.to()` call, so the CPU is no longer blocked on `cudaStreamSynchronize` and can pre-launch the sampler kernels. Reported ~5% single-request tok/s gain on Llama 3.1 8B / H100 (87.5 → 92 tok/s), backed by Nsys before/after profiles.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/guided_decoding/xgrammar_decoding.py`

### Diff
```diff
diff --git a/vllm/model_executor/guided_decoding/xgrammar_decoding.py b/vllm/model_executor/guided_decoding/xgrammar_decoding.py
index 2d8594cb8aaf..ee30ce96f0a1 100644
--- a/vllm/model_executor/guided_decoding/xgrammar_decoding.py
+++ b/vllm/model_executor/guided_decoding/xgrammar_decoding.py
@@ -307,8 +307,8 @@ def __call__(self, input_ids: list[int],
         # Note: In this method, if the tensors have different dimensions
         # on CPU device fails, but on GPU it runs without error. Hence the
         # unsqueeze above for scores, to match the token bitmask shape
-        xgr.apply_token_bitmask_inplace(scores,
-                                        self.token_bitmask.to(scores.device))
+        xgr.apply_token_bitmask_inplace(
+            scores, self.token_bitmask.to(scores.device, non_blocking=True))
         if device_type != "cuda":
             scores = scores.to(dtype).to(device_type).squeeze()
 

```
