# PR #14540: [Perf] Improve MLA on V1

- field: attention  | area: v1_attention
- author: simon-mo
- state: MERGED  created: 2025-03-10T05:50:14Z  merged: 2025-03-10T19:06:58Z
- url: https://github.com/vllm-project/vllm/pull/14540
- labels: ready, v1

# Issue

## Problem

vLLM V1's MLA (Multi-head Latent Attention) backend was slower than V0, and in some cases produced a 0 GSM8K accuracy score. Two root causes: (1) the `rotary_emb` specialization passed non-contiguous query/key tensors into the CUDA RoPE kernel, which silently produced wrong results because the kernel requires contiguous input; (2) the metadata `build` function performed several GPU->CPU synchronizing operations (`.max()`, `.sum().item()`, `cdiv` on device tensors), each of which blocks on all prior kernels and adds CPU overhead.

## Example

On 8xH200 running DeepSeek-R1 with 6k input / 1k output, V1 throughput was 527 output tok/s vs 865 after the fix (64% boost). The GPU->CPU syncs added up to ~1ms p99 tail latency per build call. The RoPE contiguity bug zeroed out GSM8K entirely.

## Use case

Serving DeepSeek MLA models (R1, V2-Lite) under V1 engine, especially longer-context decode/prefill where per-step metadata overhead and RoPE correctness matter.

## Resources

CPU overhead and kernel-launch/sync latency (GPU->CPU syncs block the stream); also GPU compute correctness for the RoPE kernel path.

## Type of users

Users serving MLA/DeepSeek models on CUDA GPUs via the V1 engine, particularly multi-GPU tensor-parallel long-context inference.

## Kind of inputs

Longer input sequences (3k-6k tokens) with chunked prefill enabled; batches mixing prefill and decode requests where context_lens vary per request.

## Mechanism with the issue

In `MLACommonMetadataBuilder.build`, computing `context_lens.max()`, `num_prefills_with_context`, `num_chunks`, and `max_query_len` from device tensors forced GPU->CPU syncs. In `MLACommonImpl.forward`, `self.rotary_emb` called `_forward_method` with non-contiguous `q_pe`/`k_pe` slices, violating the CUDA RoPE kernel's contiguity requirement.

# Solution

## Fix in words

The `build` function now keeps `seq_lens`/`context_lens` work on CPU tensors (`seq_lens_cpu`, `context_lens_cpu`): `max_query_len`, `max_context_len_cpu`, `num_prefills_with_context_cpu`, and `num_chunks` are all derived via `.max().item()`/`.sum().item()` on CPU, eliminating GPU->CPU syncs. Chunk metadata (`chunk_starts`, cumsum into a pinned `cu_seq_lens_cpu`) is built on CPU then copied to device with `non_blocking=True`, and block_table fetch is reordered before the async H2D copies. For RoPE, `self.rotary_emb` is set to `forward_native` by default and `forward_cuda` on CUDA, and `decode_q_pe`/`prefill_q_pe` are passed as `.contiguous()` so the CUDA kernel gets valid contiguous input, fixing the 0 GSM8K bug.

## Author's solution

Author fixed two things: (1) the `rotary_emb` specialization from #14384/#14471/#14476 — identified the 0 GSM8K score came from the CUDA kernel needing contiguous input, and fixed it by making the input contiguous where possible (noting a better fix would change the kernel itself). (2) Reordered operations in the `build` function to avoid GPU->CPU communication (CPU->GPU is fine), since the syncs cost up to 1ms p99 tail latency. Validated on 8xH200 with no accuracy regression.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/attention/backends/mla/common.py`

### Diff
```diff
diff --git a/vllm/v1/attention/backends/mla/common.py b/vllm/v1/attention/backends/mla/common.py
index 0b0f521672b0..526b792ab1f9 100644
--- a/vllm/v1/attention/backends/mla/common.py
+++ b/vllm/v1/attention/backends/mla/common.py
@@ -223,6 +223,7 @@
 from vllm.model_executor.layers.quantization.utils.quant_utils import (
     scaled_quantize)
 from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
+from vllm.platforms import current_platform
 from vllm.utils import cdiv, round_down
 
 try:
@@ -471,18 +472,23 @@ def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
               common_prefix_len: int) -> M:
         assert self._num_decodes + self._num_prefills == num_reqs
 
+        # Note(simon): be careful about the CPU <> GPU memory movement in this
+        # function. We should avoid GPU -> CPU sync as much as possible because
+        # it blocks on all previous kernels.
         device = self.runner.device
-        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
-            device, non_blocking=True)
-        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(device,
-                                                          non_blocking=True)
         block_table = (
             self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
+        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
+            device, non_blocking=True)
         slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
             device, non_blocking=True).long()
         input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
             device, non_blocking=True).long()
 
+        seq_lens_cpu = self.runner.seq_lens_cpu[:num_reqs]
+        seq_lens = seq_lens_cpu.to(device, non_blocking=True)
+        max_query_len = seq_lens_cpu.max().item()
+
         prefill_metadata = None
         if self._num_prefills > 0:
             reqs_start = self._num_decodes  # prefill_start
@@ -490,24 +496,22 @@ def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
 
             context_lens_cpu = self.runner.input_batch.\
                 num_computed_tokens_cpu_tensor[reqs_start:num_reqs]
-            context_lens = context_lens_cpu.to(device, non_blocking=True)
+            max_context_len_cpu = context_lens_cpu.max().item()
+            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
 
             chunked_context_metadata = None
             if self.chunked_prefill_enabled and self._num_prefills > 0 \
-                and context_lens.max() > 0:
+                and max_context_len_cpu > 0:
                 # NOTE: it is recommend you read the `Chunked Prefill` section
                 # in the comment at the top of the file before trying to
                 # understand the following code
 
-                num_prefills_with_context = (context_lens > 0).sum().item()
-
                 # currently we allocate an equal amount of workspace for each
                 # prefill in the batch, we could probably use a more advanced
                 # algorithm here and allocate more workspace to prefills with
                 # longer context lengths
-                max_context_chunk = \
-                    self.chunked_prefill_workspace_size \
-                        // num_prefills_with_context
+                max_context_chunk = (self.chunked_prefill_workspace_size //
+                                     num_prefills_with_context_cpu)
 
                 # align max_context_chunk to page_size by rounding down,
                 # currently the `gather_cache` kernel cannot handle
@@ -516,30 +520,35 @@ def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
                                                self.page_size)
 
                 assert max_context_chunk > 0
-                num_chunks = cdiv(context_lens.max(), max_context_chunk)
+                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)
 
                 # if `max_context_chunk = 256`, `num_chunks = 3`, and
                 #   `num_prefills_with_context = 4`, create a tensor that looks
                 # like
                 #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
+                # Note(simon): this is done in CPU because of downstream's
+                # of `to_list`.
                 chunk_starts = \
-                    torch.arange(num_chunks, device=device, dtype=torch.int32) \
+                    torch.arange(num_chunks, dtype=torch.int32) \
                     .unsqueeze(1).expand(-1, self._num_prefills) \
                     * max_context_chunk
-                chunk_ends = torch.min(context_lens.unsqueeze(0),
+                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                        chunk_starts + max_context_chunk)
                 chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)
-                _chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(
-                    torch.int32)
-                zero = torch.zeros(num_chunks,
-                                   dtype=torch.int32,
-                                   device=device).unsqueeze(-1)
+
+                cu_seq_lens_cpu = torch.zeros(num_chunks,
+                                              self._num_prefills + 1,
+                                              dtype=torch.int32,
+                                              pin_memory=True)
+                torch.cumsum(chunk_seq_lens,
+                             dim=1,
+                             out=cu_seq_lens_cpu[:, 1:],
+                             dtype=torch.int32)
 
                 chunked_context_metadata = \
                     MLACommonPrefillMetadata.ChunkedContextMetadata(
-                    cu_seq_lens=torch.cat(
-                        [zero, _chunk_cu_seq_lens], dim=1),
-                    starts=chunk_starts,
+                    cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
+                    starts=chunk_starts.to(device, non_blocking=True),
                     seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                     max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                     workspace=self.chunked_prefill_workspace,
@@ -553,7 +562,7 @@ def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
                 block_table=block_table[reqs_start:, ...],
                 query_start_loc=query_start_loc[reqs_start:] -
                 query_start_loc[reqs_start],
-                max_query_len=seq_lens[reqs_start:].max().item(),
+                max_query_len=max_query_len,
                 chunked_context=chunked_context_metadata,
             )
 
@@ -629,7 +638,9 @@ def __init__(
         # already inside an attention custom op), pull out the forward
         # method from the rotary embedding and call it directly
         # TODO(lucas): we should probably find a cleaner way to do this
-        self.rotary_emb = rotary_emb._forward_method
+        self.rotary_emb = rotary_emb.forward_native
+        if current_platform.is_cuda():
+            self.rotary_emb = rotary_emb.forward_cuda
 
         self.q_proj = q_proj
         self.kv_b_proj = kv_b_proj
@@ -1043,17 +1054,20 @@ def forward(
             decode_q_nope = self._q_proj_and_k_up_proj(decode_hs_or_q_c)
             decode_q_pe = torch.matmul(decode_hs_or_q_c, self.W_QR)\
                 .view(-1, self.num_heads, self.qk_rope_head_dim)
+
             decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
-                attn_metadata.decode.input_positions, decode_q_pe, decode_k_pe)
+                attn_metadata.decode.input_positions, decode_q_pe.contiguous(),
+                decode_k_pe)
 
         if has_prefill:
             assert attn_metadata.prefill is not None
             prefill_q = self.q_proj(prefill_hs_or_q_c)[0]\
                 .view(-1, self.num_heads, self.qk_head_dim)
             prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
+
             prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
-                attn_metadata.prefill.input_positions, prefill_q_pe,
-                prefill_k_pe)
+                attn_metadata.prefill.input_positions,
+                prefill_q_pe.contiguous(), prefill_k_pe)
 
         # write the latent and rope to kv cache
         if kv_cache.numel() > 0:

```
