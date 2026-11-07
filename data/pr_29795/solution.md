# PR #29795: [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum

- field: quantization  | area: distributed_device_communicators
- author: IwakuraRein
- state: MERGED  created: 2025-12-01T17:08:41Z  merged: 2025-12-08T23:02:34Z
- url: https://github.com/vllm-project/vllm/pull/29795
- labels: ready, v1, nvidia

# Issue

## Problem

Two inefficiencies in the MLA decode path. First, the CUDA communicator's `reduce_scatterv` always used the variable-sizes NCCL path (`reduce_scatterv`) whenever `sizes` was non-None, even when all per-rank sizes were equal; the equal-size case should use the faster uniform `reduce_scatter`. Second, when fp8 attention is enabled, the MLA backend ran the `scaled_fp8_quant` kernel twice (once on `decode_ql_nope`, once on `decode_q_pe`), plus a downstream `torch.cat` to join them.

## Example

Serving DeepSeek R1 0528 FP4 on 4 GB200 with DEP4 and FP8 attention: every decode step launched two fp8 quant kernels and a cat, and reduce_scatter took the slow variable-size path even with uniform shards.

## Use case

Decode phase of MLA attention with fp8 KV-cache / fp8 attention under tensor/data-parallel serving of large DeepSeek-style models.

## Resources

GPU kernel-launch overhead and HBM bandwidth (redundant quant kernels + extra cat allocation/copy); NVLink collective efficiency (variable-size vs uniform reduce-scatter).

## Type of users

Users serving FP8-quantized MLA models (DeepSeek R1) on multi-GPU TP/DP setups, Hopper (FLASHMLA) and Blackwell (FLASHINFER_MLA) hardware.

## Kind of inputs

Decode-stage requests with fp8 attention enabled; query tensors `decode_ql_nope` (B,N,L) and `decode_q_pe` (B,N,P) in fp8; collective inputs where all ranks hold equal shard sizes.

## Mechanism with the issue

In `cuda_communicator.reduce_scatterv`, the branch picked `pynccl_comm.reduce_scatterv` whenever `sizes is not None`, missing the uniform-size fast path. In `mla/common.py forward`, the fp8 branch called `ops.scaled_fp8_quant` separately on reshaped `decode_ql_nope` and `decode_q_pe`, each with reshape round-trips, then the flashinfer backend did a `torch.cat` to merge them — doubling quant kernel launches and adding cat overhead.

# Solution

## Fix in words

In `cuda_communicator.py`, the condition guarding the variable-size path becomes `sizes is not None and sizes.count(sizes[0]) != len(sizes)`, so uniform shard sizes fall through to the cheaper `reduce_scatter` and only truly-varying sizes use `reduce_scatterv`. In `mla/common.py`, instead of quantizing `decode_ql_nope` and `decode_q_pe` separately, a single pre-concatenated fp8 staging buffer `decode_q0` of shape (B, N, L+P) is allocated with `torch.empty` and filled via two `copy_` into the L and P slices (empty+copy chosen because `torch.cat` adds significant overhead). A single `scaled_fp8_quant` runs on the flattened buffer, then reshapes back to (B,N,L+P). This halves fp8 quant kernel launches and eliminates the downstream `torch.cat` in the flashinfer MLA backend.

## Author's solution

Author chose `ReduceScatterSum` over `ReduceSum` when sizes are the same. To avoid calling the fp8 quant kernel twice under fp8 attention, author created a stage buffer for `decode_ql_nope` and `decode_q_pe`, which also eliminates the `torch.cat` in the flashinfer MLA backend. Reported ~5% projected decode improvement for flashinfer MLA serving DeepSeek R1 FP4 on 4 GB200 with DEP4, with gsm8k accuracy unchanged.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/distributed/device_communicators/cuda_communicator.py`
- `vllm/v1/attention/backends/mla/common.py`

### Diff
```diff
diff --git a/vllm/distributed/device_communicators/cuda_communicator.py b/vllm/distributed/device_communicators/cuda_communicator.py
index 2e878eef908a..cd9c267beb5b 100644
--- a/vllm/distributed/device_communicators/cuda_communicator.py
+++ b/vllm/distributed/device_communicators/cuda_communicator.py
@@ -225,7 +225,7 @@ def reduce_scatterv(
             output_shape, dtype=input_tensor.dtype, device=input_tensor.device
         )
 
-        if sizes is not None:
+        if sizes is not None and sizes.count(sizes[0]) != len(sizes):
             pynccl_comm.reduce_scatterv(output, input_tensor, sizes=sizes)
         else:
             pynccl_comm.reduce_scatter(output, input_tensor)
diff --git a/vllm/v1/attention/backends/mla/common.py b/vllm/v1/attention/backends/mla/common.py
index 309ddee4fc2f..0a5257a1d87d 100755
--- a/vllm/v1/attention/backends/mla/common.py
+++ b/vllm/v1/attention/backends/mla/common.py
@@ -2037,21 +2037,30 @@ def forward(
 
             if fp8_attention:
                 ql_nope_shape = decode_ql_nope.shape
-                decode_ql_nope, _ = ops.scaled_fp8_quant(
-                    decode_ql_nope.reshape(
-                        [ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]]
-                    ),
-                    layer._q_scale,
-                )
-                decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
                 q_pe_shape = decode_q_pe.shape
-                decode_q_pe, _ = ops.scaled_fp8_quant(
-                    decode_q_pe.reshape([q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
-                    layer._q_scale,
+                assert decode_ql_nope.shape[0] == decode_q_pe.shape[0]
+                assert decode_ql_nope.shape[1] == decode_q_pe.shape[1]
+                decode_q_shape = (
+                    ql_nope_shape[0],
+                    ql_nope_shape[1],
+                    ql_nope_shape[2] + q_pe_shape[2],
+                )
+                # Using empty and copy since torch.cat introduces significant overhead.
+                decode_q0 = torch.empty(
+                    decode_q_shape,
+                    device=decode_ql_nope.device,
+                    dtype=decode_ql_nope.dtype,
                 )
-                decode_q_pe = decode_q_pe.reshape(q_pe_shape)
+                decode_q0[..., : ql_nope_shape[2]].copy_(decode_ql_nope)
+                decode_q0[..., ql_nope_shape[2] :].copy_(decode_q_pe)
 
-            decode_q = (decode_ql_nope, decode_q_pe)
+                decode_q, _ = ops.scaled_fp8_quant(
+                    decode_q0.view(decode_q_shape[0], -1),
+                    layer._q_scale,
+                )
+                decode_q = decode_q.view(decode_q_shape)
+            else:
+                decode_q = (decode_ql_nope, decode_q_pe)
             if self.dcp_world_size > 1:
                 assert not fp8_attention, "DCP not support fp8 kvcache now."
                 # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)

```
