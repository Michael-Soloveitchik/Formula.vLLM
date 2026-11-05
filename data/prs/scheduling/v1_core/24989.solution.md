# PR #24989: [Core] Get num_encoder_tokens from scheduler config

- field: scheduling  | area: v1_core
- author: russellb
- state: MERGED  created: 2025-09-16T19:00:19Z  merged: 2025-09-17T03:59:07Z
- url: https://github.com/vllm-project/vllm/pull/24989
- labels: ready, v1

# Issue

## Problem

For encoder-decoder models, the scheduler, KV cache manager, and GPU model runner each computed the maximum encoder input length by calling MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(model_config). This registry lookup is significantly slower than reading the equivalent value that is already cached on the scheduler config as max_num_encoder_input_tokens. The redundant lookup was hit on hot paths, notably inside the per-step schedule() loop.

## Example

When serving Whisper (encoder length 1500), every scheduling step recomputed num_encoder_tokens via the multimodal registry instead of reading the pre-cached scheduler_config value, adding needless per-step overhead that capped throughput.

## Use case

Real-time serving of encoder-decoder/multimodal models (e.g. Whisper speech-to-text) where the scheduler runs its loop continuously under load.

## Resources

CPU / scheduling overhead — registry lookups consume cycles on the critical scheduling path, reducing achievable throughput (kernel/GPU time is not the bottleneck here).

## Type of users

Users serving encoder-decoder models (Whisper is the only supported one) on vLLM v1, especially those pushing high request throughput.

## Kind of inputs

Workloads of encoder-decoder requests requiring cross-attention encoder state caching; triggered every scheduling step regardless of sequence shape.

## Mechanism with the issue

schedule() in scheduler.py, CrossAttentionSpec.max_memory_usage_bytes() in kv_cache_interface.py, and GPUModelRunner.__init__ in gpu_model_runner.py all called MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(model_config). That method performs a registry resolution each call, which is slow relative to the identical value already memoized on the scheduler config.

# Solution

## Fix in words

The diff replaces every MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(model_config) call with a direct read of scheduler_config.max_num_encoder_input_tokens (or vllm_config.scheduler_config.max_num_encoder_input_tokens) in three files: the scheduler's schedule() loop, CrossAttentionSpec.max_memory_usage_bytes, and GPUModelRunner.__init__. It also drops the now-unused MULTIMODAL_REGISTRY import from kv_cache_interface.py. Both sources yield the same max encoder length, but the cached config read avoids the slower registry lookup. Effect: ~3% throughput increase with no behavior change.

## Author's solution

Author mirrors the change from PR #24866: swap the slow MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(...) call for the value already cached on the scheduler config, scheduler_config.max_num_encoder_input_tokens, applying it to the scheduler, KV cache manager, and GPU model runner. Notes it is a modest ~3% throughput gain and a good step, while a larger throughput limiter is still being tracked down.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index c1e59423e9a1..85ca858ad7bd 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -465,9 +465,8 @@ def schedule(self) -> SchedulerOutput:
                             in self.vllm_config.model_config.model.lower()), (
                                 "Whisper is the only supported "
                                 "encoder-decoder model.")
-                    num_encoder_tokens = MULTIMODAL_REGISTRY.\
-                        get_encdec_max_encoder_len(
-                        self.vllm_config.model_config)
+                    num_encoder_tokens =\
+                        self.scheduler_config.max_num_encoder_input_tokens
                 else:
                     num_encoder_tokens = 0
 
diff --git a/vllm/v1/kv_cache_interface.py b/vllm/v1/kv_cache_interface.py
index 6e8f569fff0e..0cf92a680a68 100644
--- a/vllm/v1/kv_cache_interface.py
+++ b/vllm/v1/kv_cache_interface.py
@@ -11,7 +11,6 @@
 
 from vllm.config import VllmConfig
 from vllm.logger import init_logger
-from vllm.multimodal import MULTIMODAL_REGISTRY
 from vllm.utils import cdiv, get_dtype_size
 
 logger = init_logger(__name__)
@@ -230,8 +229,8 @@ class CrossAttentionSpec(AttentionSpec):
     def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
         # For cross-attention, we need to cache encoder states
         # Get encoder length (e.g., 1500 for Whisper).
-        max_encoder_len = MULTIMODAL_REGISTRY.\
-            get_encdec_max_encoder_len(vllm_config.model_config)
+        max_encoder_len = vllm_config.scheduler_config.\
+            max_num_encoder_input_tokens
         return cdiv(max_encoder_len, self.block_size) * self.page_size_bytes
 
 
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 2ae748dee43c..2e00660cccc4 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -234,8 +234,8 @@ def __init__(
         if self.model_config.is_encoder_decoder:
             # Maximum length of the encoder input, only for encoder-decoder
             # models.
-            self.max_encoder_len = self.mm_registry.\
-                get_encdec_max_encoder_len(model_config)
+            self.max_encoder_len = scheduler_config.\
+                            max_num_encoder_input_tokens
         else:
             self.max_encoder_len = 0
 

```
