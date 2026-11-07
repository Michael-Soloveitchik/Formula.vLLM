# PR #23777: [LoRA] Much faster startup when LoRA is enabled

- field: scheduling  | area: v1_worker
- author: andylolu2
- state: MERGED  created: 2025-08-27T21:50:18Z  merged: 2025-08-30T15:37:39Z
- url: https://github.com/vllm-project/vllm/pull/23777
- labels: ready, v1

# Issue

## Problem

When LoRA is enabled, server startup is extremely slow because every call to `_dummy_run` destroys and re-creates dummy LoRA weights. During warmup and CUDA graph capture, `_dummy_run` is invoked many times, so the full set of dummy LoRA adapters is torn down and rebuilt on each invocation. This redundant setup/teardown dominates startup time, scaling badly with `--max-loras`.

## Example

Serving `meta-llama/Meta-Llama-3-70B` with `--enable-lora --tensor-parallel-size 4 --max-loras 32`, CUDA graph capture of 67 cases took 6:09 (5.52 s/it) before the fix versus 0:15 (4.38 it/s) after — a >20x speedup.

## Use case

Launching a vLLM server with LoRA enabled, particularly with a large `--max-loras` value for multi-adapter serving.

## Resources

CPU/kernel setup time during warmup and CUDA graph capture — repeated adapter weight allocation/destruction, not steady-state GPU compute.

## Type of users

Users serving LoRA adapters, especially multi-adapter deployments with large `--max-loras` and multi-GPU tensor-parallel setups.

## Kind of inputs

Startup-time dummy runs across many warmup sizes and CUDA graph capture cases; triggered purely by LoRA being enabled with many configured adapter slots.

## Mechanism with the issue

In `lora_model_runner_mixin.py`, the `maybe_setup_dummy_loras` context manager unconditionally called `self.lora_manager.remove_all_adapters()` in its `__exit__`. Since `_dummy_run` enters this context on every call during warmup (`gpu_worker.compile_or_warm_up_model`) and capture (`_capture_cudagraphs`), dummy LoRAs were destroyed and recreated for each of the dozens of dummy runs.

# Solution

## Fix in words

The diff adds a `remove_lora: bool = True` parameter threaded through `_dummy_run`, `maybe_dummy_run_with_lora`, and `maybe_setup_dummy_loras`. When `remove_lora=False`, the context manager's exit skips `remove_all_adapters()`, so consecutive dummy runs reuse the same dummy LoRA weights instead of tearing them down each time. Callers in `_capture_cudagraphs` and `compile_or_warm_up_model` now pass `remove_lora=False`, and a new `maybe_remove_all_loras` helper is called once after all those loops complete to do the single cleanup. This collapses O(num_dummy_runs) adapter setup/teardown into a single teardown, yielding the >20x faster CUDA graph capture.

## Author's solution

The author makes LoRA destruction optional so that consecutive `_dummy_run` calls during warmup and CUDA graph capture can share the same dummy LoRAs instead of destroying and re-creating them each time.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/lora_model_runner_mixin.py`

### Diff
```diff
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index c6d50c17f2b4..d6717892d4ae 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -2213,6 +2213,7 @@ def _dummy_run(
         uniform_decode: bool = False,
         skip_eplb: bool = False,
         is_profile: bool = False,
+        remove_lora: bool = True,
     ) -> tuple[torch.Tensor, torch.Tensor]:
         """
         Run a dummy forward pass to warm up/profile run or capture the
@@ -2230,6 +2231,7 @@ def _dummy_run(
             uniform_decode: If True, the batch is a uniform decode batch.
             skip_eplb: If True, skip EPLB state update.
             is_profile: If True, this is a profile run.
+            remove_lora: If False, dummy LoRAs are not destroyed after the run
         """
         assert cudagraph_runtime_mode in {
             CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
@@ -2317,7 +2319,7 @@ def _dummy_run(
                         attn_metadata[layer_name] = attn_metadata_i
 
         with self.maybe_dummy_run_with_lora(self.lora_config,
-                                            num_scheduled_tokens):
+                                            num_scheduled_tokens, remove_lora):
             if self.supports_mm_inputs:
                 input_ids = None
                 inputs_embeds = self.inputs_embeds[:num_tokens]
@@ -2708,11 +2710,14 @@ def _capture_cudagraphs(self, compilation_cases: list[int],
                                 cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                 force_attention=force_attention,
                                 uniform_decode=uniform_decode,
-                                skip_eplb=True)
+                                skip_eplb=True,
+                                remove_lora=False)
             self._dummy_run(num_tokens,
                             cudagraph_runtime_mode=cudagraph_runtime_mode,
                             uniform_decode=uniform_decode,
-                            skip_eplb=True)
+                            skip_eplb=True,
+                            remove_lora=False)
+        self.maybe_remove_all_loras(self.lora_config)
 
     def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
         """
diff --git a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py
index 2088bfff5bb3..2e7d6685377f 100644
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -308,7 +308,10 @@ def compile_or_warm_up_model(self) -> None:
         # We skip EPLB here since we don't want to record dummy metrics
         for size in sorted(warmup_sizes, reverse=True):
             logger.info("Compile and warming up model for size %d", size)
-            self.model_runner._dummy_run(size, skip_eplb=True)
+            self.model_runner._dummy_run(size,
+                                         skip_eplb=True,
+                                         remove_lora=False)
+        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)
 
         # Warmup and tune the kernels used during model execution before
         # cuda graph capture.
diff --git a/vllm/v1/worker/lora_model_runner_mixin.py b/vllm/v1/worker/lora_model_runner_mixin.py
index 84ed46989ea9..4b5f27d27541 100644
--- a/vllm/v1/worker/lora_model_runner_mixin.py
+++ b/vllm/v1/worker/lora_model_runner_mixin.py
@@ -5,7 +5,7 @@
 """
 
 from contextlib import contextmanager
-from typing import Union
+from typing import Optional, Union
 
 import numpy as np
 import torch
@@ -87,7 +87,9 @@ def set_active_loras(self, input_batch: InputBatch,
                                       lora_requests)
 
     @contextmanager
-    def maybe_setup_dummy_loras(self, lora_config):
+    def maybe_setup_dummy_loras(self,
+                                lora_config: Optional[LoRAConfig],
+                                remove_lora: bool = True):
         if lora_config is None:
             yield
         else:
@@ -114,10 +116,11 @@ def maybe_setup_dummy_loras(self, lora_config):
                 yield
 
             # __exit__ code
-            self.lora_manager.remove_all_adapters()
+            if remove_lora:
+                self.lora_manager.remove_all_adapters()
 
     @contextmanager
-    def maybe_select_dummy_loras(self, lora_config: LoRAConfig,
+    def maybe_select_dummy_loras(self, lora_config: Optional[LoRAConfig],
                                  num_scheduled_tokens: np.ndarray):
         if lora_config is None:
             yield
@@ -151,13 +154,22 @@ def maybe_select_dummy_loras(self, lora_config: LoRAConfig,
             yield
 
     @contextmanager
-    def maybe_dummy_run_with_lora(self, lora_config: LoRAConfig,
-                                  num_scheduled_tokens: np.ndarray):
-        with self.maybe_setup_dummy_loras(
-                lora_config), self.maybe_select_dummy_loras(
-                    lora_config, num_scheduled_tokens):
+    def maybe_dummy_run_with_lora(self,
+                                  lora_config: Optional[LoRAConfig],
+                                  num_scheduled_tokens: np.ndarray,
+                                  remove_lora: bool = True):
+        with (
+                self.maybe_setup_dummy_loras(lora_config, remove_lora),
+                self.maybe_select_dummy_loras(lora_config,
+                                              num_scheduled_tokens),
+        ):
             yield
 
+    def maybe_remove_all_loras(self, lora_config: Optional[LoRAConfig]):
+        if lora_config is None:
+            return
+        self.lora_manager.remove_all_adapters()
+
     def add_lora(self, lora_request: LoRARequest) -> bool:
         if not self.lora_manager:
             raise RuntimeError("LoRA is not enabled.")

```
