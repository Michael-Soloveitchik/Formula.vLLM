# PR #28879: [Feature] Shared Experts Overlap with FI deepgemm swap kernel, 2.2% throughput improvement and 3.6% TTFT improvement

- field: gpu_kernels  | area: model_executor_layers
- author: yewentao256
- state: MERGED  created: 2025-11-17T20:55:06Z  merged: 2025-11-21T01:41:49Z
- url: https://github.com/vllm-project/vllm/pull/28879
- labels: ready

# Issue

## Problem

The shared-experts-on-separate-CUDA-stream overlap optimization existed only in the dense FusedMoE forward path (layer.py forward_impl), not in the modular-kernel path used by the FlashInfer/DeepGEMM swap kernel. So when MoE ran through FusedMoEModularKernel, the shared experts executed serially on the current stream instead of overlapping with the routed-expert GEMM, leaving the GPU idle between launches. Additionally, the prepare step mutated the input activation in place (a1.mul_), which is unsafe once a cloned copy of hidden_states is handed to a concurrent stream.

## Example

DeepSeek-V3.1 served with -tp 8 --enable-expert-parallel at batch size 32: shared experts and routed experts ran back-to-back rather than concurrently, visible as non-overlapping gaps in the Perfetto trace. Overlapping yields 2.2% throughput and 3.6% TTFT improvement at BS 32.

## Use case

Serving large MoE models (DeepSeek-V3.1) with expert parallelism where the modular fused-MoE kernel (FlashInfer deepgemm) handles routing and the model also has separate shared experts per token.

## Resources

GPU compute / SM occupancy and kernel-launch scheduling — the routed-expert GEMM and shared-expert GEMM could run on parallel streams but were serialized, wasting SM cycles and adding latency.

## Type of users

Users serving MoE models with separate shared experts on multi-GPU tensor-parallel + expert-parallel setups that route through the modular kernel (deepgemm/FlashInfer backend) on CUDA.

## Kind of inputs

Small-to-moderate token counts: batch size at or below VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (e.g. BS 32). Above ~256 tokens the overlap is disabled, so the win targets low-concurrency / decode-heavy batches.

## Mechanism with the issue

FusedMoEModularKernel had no shared_experts_stream member and _finalize always ran self.shared_experts(hidden_states) on the current stream. Overlap plumbing lived only in FusedMoE.forward_impl, so the modular path never cloned hidden_states, never issued wait_stream/record_stream, and never launched shared experts concurrently. The inplace a1.mul_ in prepare_finalize also risked corrupting hidden_states shared across streams.

# Solution

## Fix in words

Threads a shared_experts_stream through the modular kernel: fused_moe_modular_method.make passes getattr(moe_layer, 'shared_experts_stream', None) into the FusedMoEModularKernel constructor, which stores it. New _maybe_setup_shared_experts_stream clones hidden_states (gated by is_cuda and the token threshold), calls record_stream on the clone, and wait_stream to fork the side stream. In _finalize, maybe_run_shared_experts launches self.shared_experts(hidden_states_clone) inside torch.cuda.stream(shared_experts_stream) to overlap with the fused-MoE finalize, and _wait_for_shared_experts_stream joins the streams before consuming shared_output. prepare_finalize.py switches the apply_router_weight_on_input scaling from inplace a1.mul_ to a1 = a1 * ... so the original activation is not mutated while the clone runs concurrently. layer.py extracts the existing inline overlap logic into a helper with no behavior change.

## Author's solution

The author enables overlap of the shared experts with the FlashInfer deepgemm swap kernel so they run concurrently instead of serially, reporting 2.2% throughput and 3.6% TTFT improvement at batch size 32 with smaller but still positive gains at BS 256. Overlap is gated by the VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD env var so it is not used above batch size 256. Accuracy was verified on gsm8k (0.9591 exact_match) and performance via vllm bench serve on DeepSeek-V3.1.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`
- `vllm/model_executor/layers/fused_moe/layer.py`
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`
- `vllm/model_executor/layers/fused_moe/prepare_finalize.py`

### Diff
```diff
diff --git a/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py b/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
index 43974ba917e4..c6dc95acdb63 100644
--- a/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
+++ b/vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py
@@ -50,6 +50,7 @@ def make(
                 prepare_finalize,
                 old_quant_method.select_gemm_impl(prepare_finalize, moe_layer),
                 shared_experts,
+                getattr(moe_layer, "shared_experts_stream", None),
             ),
         )
 
diff --git a/vllm/model_executor/layers/fused_moe/layer.py b/vllm/model_executor/layers/fused_moe/layer.py
index d9525a7439c3..b2f554efd8a6 100644
--- a/vllm/model_executor/layers/fused_moe/layer.py
+++ b/vllm/model_executor/layers/fused_moe/layer.py
@@ -850,6 +850,45 @@ def update_expert_map(self):
                     dp_size=get_dp_group().world_size,
                 )
 
+    def _maybe_setup_shared_experts_stream(
+        self,
+        hidden_states: torch.Tensor,
+        has_separate_shared_experts: bool,
+        use_chunked_impl: bool,
+    ) -> tuple[bool, torch.Tensor | None]:
+        use_shared_experts_stream = (
+            has_separate_shared_experts
+            and not use_chunked_impl
+            and self.shared_experts_stream is not None
+            and (
+                hidden_states.shape[0]
+                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
+            )
+        )
+
+        hidden_states_clone: torch.Tensor | None = None
+        if use_shared_experts_stream:
+            assert self.shared_experts_stream is not None
+
+            # Clone BEFORE switching streams to avoid race condition
+            # where routed_expert kernel may mutate hidden_states.
+            hidden_states_clone = hidden_states.clone()
+
+            # Record that the clone will be used by shared_experts_stream
+            # to avoid gc issue from deallocation of hidden_states_clone
+            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
+            # NOTE: We dont need shared_output.record_stream(current_stream())
+            # because we synch the streams before using shared_output.
+            hidden_states_clone.record_stream(self.shared_experts_stream)
+
+            # Mark sync start point for the separate shared experts
+            # stream here since we want to run in parallel with the
+            # router/gate (next op below)
+            assert self.shared_experts_stream is not None
+            self.shared_experts_stream.wait_stream(current_stream())
+
+        return use_shared_experts_stream, hidden_states_clone
+
     def _load_per_tensor_weight_scale(
         self,
         shard_id: str,
@@ -1819,36 +1858,12 @@ def forward_impl(
 
         use_chunked_impl = self.use_dp_chunking
 
-        use_shared_experts_stream = (
-            has_separate_shared_experts
-            and not use_chunked_impl
-            and self.shared_experts_stream is not None
-            and (
-                hidden_states.shape[0]
-                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
+        use_shared_experts_stream, hidden_states_clone = (
+            self._maybe_setup_shared_experts_stream(
+                hidden_states, has_separate_shared_experts, use_chunked_impl
             )
         )
 
-        if use_shared_experts_stream:
-            assert self.shared_experts_stream is not None
-
-            # Clone BEFORE switching streams to avoid race condition
-            # where routed_expert kernel may mutate hidden_states.
-            hidden_states_clone = hidden_states.clone()
-
-            # Record that the clone will be used by shared_experts_stream
-            # to avoid gc issue from deallocation of hidden_states_clone
-            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
-            # NOTE: We dont need shared_output.record_stream(current_stream())
-            # because we synch the streams before using shared_output.
-            hidden_states_clone.record_stream(self.shared_experts_stream)
-
-            # Mark sync start point for the separate shared experts
-            # stream here since we want to run in parallel with the
-            # router/gate (next op below)
-            assert self.shared_experts_stream is not None
-            self.shared_experts_stream.wait_stream(current_stream())
-
         # If router/gate provided, then apply it here.
         # (Note: This code runs only when "overlapped mode" is on to allow
         #        parallel execution of shared experts with the FusedMoE via
diff --git a/vllm/model_executor/layers/fused_moe/modular_kernel.py b/vllm/model_executor/layers/fused_moe/modular_kernel.py
index 093affe51f50..4af7af9257df 100644
--- a/vllm/model_executor/layers/fused_moe/modular_kernel.py
+++ b/vllm/model_executor/layers/fused_moe/modular_kernel.py
@@ -16,6 +16,7 @@
     count_expert_num_tokens,
     disable_inplace,
 )
+from vllm.platforms import current_platform
 from vllm.utils.math_utils import cdiv
 from vllm.v1.worker.ubatching import (
     dbo_current_ubatch_id,
@@ -709,11 +710,13 @@ def __init__(
         prepare_finalize: FusedMoEPrepareAndFinalize,
         fused_experts: FusedMoEPermuteExpertsUnpermute,
         shared_experts: torch.nn.Module | None = None,
+        shared_experts_stream: torch.cuda.Stream | None = None,
     ):
         super().__init__()
         self.prepare_finalize = prepare_finalize
         self.fused_experts = fused_experts
         self.shared_experts = shared_experts
+        self.shared_experts_stream = shared_experts_stream
 
         self._post_init_setup()
         assert (
@@ -890,6 +893,34 @@ def _slice_expert_tokens_metadata(
             expert_num_tokens_cpu=c_expert_num_tokens_cpu,
         )
 
+    def _maybe_setup_shared_experts_stream(
+        self, hidden_states: torch.Tensor
+    ) -> tuple[bool, torch.Tensor | None]:
+        # decide whether to run shared experts on a separate CUDA stream to
+        # overlap with the main fused MoE kernel.
+        use_shared_experts_stream = (
+            self.shared_experts is not None
+            and self.shared_experts_stream is not None
+            and hidden_states.is_cuda
+            and (
+                hidden_states.shape[0]
+                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
+            )
+        )
+
+        hidden_states_clone: torch.Tensor | None = None
+        if use_shared_experts_stream and self.shared_experts_stream is not None:
+            # TODO: Optimize this (complicated)
+            # Note: this clone adds overhead but is required
+            # for correctness with multiple CUDA streams and CUDA graph capture.
+            hidden_states_clone = hidden_states.clone()
+            # record that the clone will be used by the separate stream so its
+            # lifetime is correctly tracked.
+            hidden_states_clone.record_stream(self.shared_experts_stream)
+            self.shared_experts_stream.wait_stream(torch.cuda.current_stream())
+
+        return use_shared_experts_stream, hidden_states_clone
+
     def _prepare(
         self,
         hidden_states: torch.Tensor,
@@ -1077,12 +1108,30 @@ def _finalize(
         topk_weights: torch.Tensor,
         topk_ids: torch.Tensor,
         apply_router_weight_on_input: bool,
+        hidden_states_clone: torch.Tensor | None = None,
+        use_shared_experts_stream: bool = False,
     ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
         """
         The _finalize method is a wrapper around self.prepare_finalize.finalize
         that handles DBO, async and shared expert overlap.
         """
-        shared_output: torch.Tensor | None = None
+
+        def maybe_run_shared_experts() -> torch.Tensor | None:
+            if self.shared_experts is None:
+                return None
+
+            if (
+                not use_shared_experts_stream
+                or self.shared_experts_stream is not None
+                and (not hidden_states.is_cuda or not torch.cuda.is_available())
+            ):
+                # fall back to running on the current stream
+                return self.shared_experts(hidden_states)
+
+            assert hidden_states_clone is not None
+            # launch shared experts on the dedicated stream.
+            with torch.cuda.stream(self.shared_experts_stream):
+                return self.shared_experts(hidden_states_clone)
 
         if not self.prepare_finalize.supports_async():
             assert not dbo_enabled()
@@ -1095,8 +1144,7 @@ def _finalize(
                 apply_router_weight_on_input,
                 self.fused_experts.finalize_weight_and_reduce_impl(),
             )
-            if self.shared_experts is not None:
-                shared_output = self.shared_experts(hidden_states)
+            shared_output = maybe_run_shared_experts()
         else:
             finalize_ret = self.prepare_finalize.finalize_async(
                 output,
@@ -1107,8 +1155,7 @@ def _finalize(
                 self.fused_experts.finalize_weight_and_reduce_impl(),
             )
 
-            if self.shared_experts is not None:
-                shared_output = self.shared_experts(hidden_states)
+            shared_output = maybe_run_shared_experts()
 
             # TODO(lucas): refactor this in the alternative schedules followup
             # currently unpack if we have hook + receiver pair or just
@@ -1131,12 +1178,28 @@ def _finalize(
 
             receiver()
 
+        self._wait_for_shared_experts_stream(hidden_states, use_shared_experts_stream)
+
         if self.shared_experts is None:
             return output
         else:
             assert shared_output is not None
             return shared_output, output
 
+    def _wait_for_shared_experts_stream(
+        self, hidden_states: torch.Tensor, use_shared_experts_stream: bool
+    ) -> None:
+        # ensure that any work enqueued on the shared_experts_stream is
+        # completed before the shared_output tensor is consumed
+        if (
+            self.shared_experts is not None
+            and use_shared_experts_stream
+            and self.shared_experts_stream is not None
+            and hidden_states.is_cuda
+            and current_platform.is_cuda()
+        ):
+            torch.cuda.current_stream().wait_stream(self.shared_experts_stream)
+
     def forward(
         self,
         hidden_states: torch.Tensor,
@@ -1183,6 +1246,10 @@ def forward(
         else:
             output = torch.zeros_like(hidden_states)
 
+        use_shared_experts_stream, hidden_states_clone = (
+            self._maybe_setup_shared_experts_stream(hidden_states)
+        )
+
         local_num_experts = w1.size(0)
         if global_num_experts == -1:
             global_num_experts = local_num_experts
@@ -1219,4 +1286,6 @@ def forward(
             topk_weights,
             topk_ids,
             apply_router_weight_on_input,
+            hidden_states_clone=hidden_states_clone,
+            use_shared_experts_stream=use_shared_experts_stream,
         )
diff --git a/vllm/model_executor/layers/fused_moe/prepare_finalize.py b/vllm/model_executor/layers/fused_moe/prepare_finalize.py
index 9bb976fb9ec9..e27e2eb32da0 100644
--- a/vllm/model_executor/layers/fused_moe/prepare_finalize.py
+++ b/vllm/model_executor/layers/fused_moe/prepare_finalize.py
@@ -45,7 +45,8 @@ def prepare(
             assert topk == 1, (
                 "apply_router_weight_on_input is only implemented for topk=1"
             )
-            a1.mul_(topk_weights.to(a1.dtype))
+            # Note: do not use inplace for shared experts overlap
+            a1 = a1 * topk_weights.to(a1.dtype)
 
         a1q, a1q_scale = moe_kernel_quantize_input(
             a1,

```
