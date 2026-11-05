# PR #39016: [MoE] Triton MoE Perf regression - restore low latency path

- field: parallelism  | area: model_executor_layers
- author: milesial
- state: MERGED  created: 2026-04-05T05:43:29Z  merged: 2026-04-21T06:37:11Z
- url: https://github.com/vllm-project/vllm/pull/39016
- labels: ready

# Issue

## Problem

The unquantized and FP8 MoE paths lost a low-latency optimization during the modular-kernel refactor. In the new path, UnquantizedFusedMoEMethod and FP8 backends call TritonExperts.apply()/invoke_fused_moe_kernel() directly, which unconditionally invokes moe_align_block_size (grouping and padding token ids by expert). The original fused_experts_impl() skipped this alignment when the effective batch (num_tokens * top_k) was far smaller than the number of experts, so that optimization no longer ran.

## Example

Nemotron Nano 3 on B200 at ISL50/OSL1000, batch size 1: baseline 250.42 tok/s/user (BF16) and 392.2 tok/s/user (FP8), each losing ~3-4% throughput versus the restored low-latency path.

## Use case

Low-batch, single-user interactive decoding of sparse MoE models where each forward pass routes very few tokens across many experts.

## Resources

GPU compute and kernel scheduling: wasted work launching one Triton tile per activated expert plus moe_align_block_size sort/pad overhead instead of one tile per token.

## Type of users

Users serving MoE models (BF16, FP8, and WNA16 formats) at low batch size, especially latency-sensitive single-request decode on high-end GPUs like B200.

## Kind of inputs

Small batches where num_tokens * top_k is much less than global_num_experts (e.g. BS1 decode), so each expert receives fewer than one token on average; dtype BF16 or FP8.

## Mechanism with the issue

moe_align_block_size groups and pads token ids by expert, producing sorted_token_ids and one Triton tile per activated expert. When the batch is tiny relative to expert count, this sorting/padding is wasted and launches far more tiles than tokens. The modular refactor routed the unquantized path through TritonExperts.apply() and FP8 through its modular kernel, both calling moe_align_block_size unconditionally and bypassing the skip-path that fused_experts_impl() used.

# Solution

## Fix in words

The diff extracts the low-latency logic into a new helper _prepare_expert_assignment(). The helper reproduces the original heuristic: when expert_map is None and num_tokens * top_k_num * 4 <= global_num_experts (and not a block-quantized int8/int4 case), it skips moe_align_block_size and returns sorted_token_ids=None, expert_ids=topk_ids.view(-1), and num_tokens_post_padded = topk_ids.numel() * BLOCK_SIZE_M, activating the kernel's sorted_token_ids-is-None path (one tile per token). Otherwise it falls back to moe_align_block_size. Both fused_experts_impl() and the modular TritonExperts.apply() now call this helper, so the unquantized and FP8/WNA16 modular paths regain the skip-alignment optimization, restoring ~3-4% throughput and lower latency.

## Author's solution

The author re-introduces the low-latency optimization from PR #29354 into the new modular path, noting the refactor (#36286, with roots in #31052 and FP8 in #30825) made the unquantized/FP8 paths call invoke_fused_moe_kernel directly and bypass the moe_align_block_size skip that fused_experts_impl() had. The optimization skips moe_align_block_size when batch (num_tokens * top_k) is much smaller than the expert count, giving one tile per batch item instead of per activated expert, measured at ~3-4% tok/s gain on B200 for both BF16 and FP8.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe.py b/vllm/model_executor/layers/fused_moe/fused_moe.py
index 6de25da051ad..9218d0aff748 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe.py
@@ -1551,6 +1551,55 @@ def dispatch_fused_experts_func(inplace: bool) -> Callable[..., torch.Tensor]:
     return torch_vllm_outplace_fused_experts
 
 
+def _prepare_expert_assignment(
+    topk_ids: torch.Tensor,
+    config: dict[str, Any],
+    num_tokens: int,
+    top_k_num: int,
+    global_num_experts: int,
+    expert_map: torch.Tensor | None,
+    *,
+    use_int8_w8a16: bool = False,
+    use_int4_w4a16: bool = False,
+    block_shape: list[int] | None = None,
+    ignore_invalid_experts: bool = False,
+) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
+    """Prepare expert assignments for the aligned and low-latency Triton paths."""
+    # SPARSITY_FACTOR is a heuristic margin ensuring tokens_in_chunk * top_k
+    # activates only a small fraction of total experts
+    # Skips moe_align_block_size and activates the `sorted_token_ids is None`
+    # path of the fused_moe_kernel kernel
+    naive_block_assignment = (
+        expert_map is None
+        and num_tokens * top_k_num * 4 <= global_num_experts
+        and not (
+            (use_int8_w8a16 or use_int4_w4a16)
+            and block_shape is not None
+            and block_shape[1] > 0
+        )
+    )
+
+    if naive_block_assignment:
+        return (
+            None,
+            topk_ids.view(-1),
+            torch.full(
+                (1,),
+                topk_ids.numel() * config["BLOCK_SIZE_M"],
+                dtype=torch.int32,
+                device=topk_ids.device,
+            ),
+        )
+
+    return moe_align_block_size(
+        topk_ids,
+        config["BLOCK_SIZE_M"],
+        global_num_experts,
+        expert_map,
+        ignore_invalid_experts=ignore_invalid_experts,
+    )
+
+
 # TODO (bnell): replace this with modular op.  Can get rid of inplace/outplace
 # torch ops.
 def fused_experts(
@@ -1791,37 +1840,19 @@ def fused_experts_impl(
         ocp_mx_scheme=ocp_mx_scheme,
     )
 
-    # SPARSITY_FACTOR is a heuristic margin ensuring num_tokens * top_k
-    # activates only a small fraction of total experts
-    SPARSITY_FACTOR = 4
-    # block quantized code path is not implemented yet.
-    naive_block_assignment = (
-        expert_map is None
-        and num_tokens * top_k_num * SPARSITY_FACTOR <= global_num_experts
-        and not (
-            (use_int8_w8a16 or use_int4_w4a16)
-            and block_shape is not None
-            and block_shape[1] > 0
-        )
+    sorted_token_ids, expert_ids, num_tokens_post_padded = _prepare_expert_assignment(
+        topk_ids,
+        config,
+        num_tokens,
+        top_k_num,
+        global_num_experts,
+        expert_map,
+        use_int8_w8a16=use_int8_w8a16,
+        use_int4_w4a16=use_int4_w4a16,
+        block_shape=block_shape,
+        ignore_invalid_experts=True,
     )
 
-    if not naive_block_assignment:
-        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
-            topk_ids,
-            config["BLOCK_SIZE_M"],
-            global_num_experts,
-            expert_map,
-            ignore_invalid_experts=True,
-        )
-    else:
-        max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
-        expert_ids = topk_ids.view(-1)
-        num_tokens_post_padded = torch.empty(
-            (1), dtype=torch.int32, device=topk_ids.device
-        )
-        num_tokens_post_padded.fill_(max_num_tokens_padded)
-        sorted_token_ids = None
-
     dispatch_fused_moe_kernel(
         qhidden_states,
         w1,
@@ -2073,8 +2104,18 @@ def apply(
         )
         intermediate_cache3 = _resize_cache(workspace2, (num_tokens, top_k_num, K))
 
-        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
-            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
+        sorted_token_ids, expert_ids, num_tokens_post_padded = (
+            _prepare_expert_assignment(
+                topk_ids,
+                config,
+                num_tokens,
+                top_k_num,
+                global_num_experts,
+                expert_map,
+                use_int8_w8a16=self.quant_config.use_int8_w8a16,
+                use_int4_w4a16=self.quant_config.use_int4_w4a16,
+                block_shape=self.block_shape,
+            )
         )
 
         invoke_fused_moe_triton_kernel(

```
