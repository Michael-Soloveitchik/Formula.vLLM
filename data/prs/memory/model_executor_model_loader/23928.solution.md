# PR #23928: [Model loader]: support multi-thread model weight loading

- field: memory  | area: model_executor_model_loader
- author: BraveY
- state: MERGED  created: 2025-08-29T10:34:05Z  merged: 2025-09-08T18:49:40Z
- url: https://github.com/vllm-project/vllm/pull/23928
- labels: ready

# Issue

## Problem

DefaultModelLoader loads model weight files sequentially, one safetensor or pt/bin shard at a time, via single-threaded iterators. Each file is read and deserialized to CPU before the next starts. Disk/NVMe bandwidth stays underutilized because a single thread cannot saturate I/O. Model initialization time scales linearly with shard count instead of parallel throughput.

## Example

QwQ-32B (62GB) on single NVMe SSD took 20s to load weights (3.1 GB/s) with sequential loading, far below the drive's capability.

## Use case

Server startup / model initialization when bringing up a vLLM inference endpoint, especially large multi-shard checkpoints.

## Resources

Disk/NVMe read bandwidth — single-threaded load leaves PCIe/SSD throughput idle.

## Type of users

Users serving large models (tens of GB, many shards) on fast storage (NVMe SSD, PCIe Gen4), often multi-GPU TP setups where fast cold-start matters.

## Kind of inputs

Multi-file safetensor or pt/bin checkpoints where total weight size is large and split across many shard files.

## Mechanism with the issue

_get_weights_iterator in default_loader.py calls safetensors_weights_iterator / pt_weights_iterator in weight_utils.py, which loop over hf_weights_files sequentially — each load_file/torch.load blocks until done. No concurrency = one shard read at a time = bandwidth-bound serial path.

# Solution

## Fix in words

Adds multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator to weight_utils.py, which submit each shard's load_file/torch.load to a concurrent.futures.ThreadPoolExecutor and yield state dicts via as_completed. DefaultModelLoader._get_weights_iterator reads model_loader_extra_config: when enable_multithread_load is set, it picks the multi-threaded iterator with max_workers=num_threads (default 8), else keeps the old sequential path. The __init__ previously rejected any extra config outright; now it validates keys against {enable_multithread_load, num_threads} and only raises on unexpected keys. Effect: parallel shard reads saturate NVMe bandwidth — QwQ-32B load dropped 20s→14s (3.1→4.42 GB/s, 1.43x).

## Author's solution

Author ports Sglang PR #7277 logic: add multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator using Python ThreadPoolExecutor to parallelize loading safetensor and pt/bin files. DefaultModelLoader uses these when enable_multithread_load extra config is set, with configurable num_threads (default 8). Also adds validation of extra config keys, allowing only enable_multithread_load and num_threads and raising on unexpected keys.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/model_executor/model_loader/default_loader.py`
- `vllm/model_executor/model_loader/weight_utils.py`

### Diff
```diff
diff --git a/vllm/model_executor/model_loader/default_loader.py b/vllm/model_executor/model_loader/default_loader.py
index c8ad3a55d932..4badc3175344 100644
--- a/vllm/model_executor/model_loader/default_loader.py
+++ b/vllm/model_executor/model_loader/default_loader.py
@@ -18,8 +18,9 @@
     download_safetensors_index_file_from_hf, download_weights_from_hf,
     fastsafetensors_weights_iterator, filter_duplicate_safetensors_files,
     filter_files_not_needed_for_inference, maybe_download_from_modelscope,
-    np_cache_weights_iterator, pt_weights_iterator,
-    safetensors_weights_iterator)
+    multi_thread_pt_weights_iterator,
+    multi_thread_safetensors_weights_iterator, np_cache_weights_iterator,
+    pt_weights_iterator, safetensors_weights_iterator)
 from vllm.platforms import current_platform
 
 logger = init_logger(__name__)
@@ -28,6 +29,9 @@
 class DefaultModelLoader(BaseModelLoader):
     """Model loader that can load different file types from disk."""
 
+    # default number of thread when enable multithread weight loading
+    DEFAULT_NUM_THREADS = 8
+
     @dataclasses.dataclass
     class Source:
         """A source for weights."""
@@ -52,9 +56,15 @@ class Source:
 
     def __init__(self, load_config: LoadConfig):
         super().__init__(load_config)
-        if load_config.model_loader_extra_config:
-            raise ValueError(f"Model loader extra config is not supported for "
-                             f"load format {load_config.load_format}")
+
+        extra_config = load_config.model_loader_extra_config
+        allowed_keys = {"enable_multithread_load", "num_threads"}
+        unexpected_keys = set(extra_config.keys()) - allowed_keys
+
+        if unexpected_keys:
+            raise ValueError(f"Unexpected extra config keys for load format "
+                             f"{load_config.load_format}: "
+                             f"{unexpected_keys}")
 
     def _prepare_weights(
         self,
@@ -145,6 +155,7 @@ def _get_weights_iterator(
             self, source: "Source"
     ) -> Generator[tuple[str, torch.Tensor], None, None]:
         """Get an iterator for the model weights based on the load format."""
+        extra_config = self.load_config.model_loader_extra_config
         hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
             source.model_or_path, source.revision, source.fall_back_to_pt,
             source.allow_patterns_overrides)
@@ -165,16 +176,34 @@ def _get_weights_iterator(
                     self.load_config.use_tqdm_on_load,
                 )
             else:
-                weights_iterator = safetensors_weights_iterator(
+                if extra_config.get("enable_multithread_load"):
+                    weights_iterator = (
+                        multi_thread_safetensors_weights_iterator(
+                            hf_weights_files,
+                            self.load_config.use_tqdm_on_load,
+                            max_workers=extra_config.get(
+                                "num_threads", self.DEFAULT_NUM_THREADS),
+                        ))
+                else:
+                    weights_iterator = safetensors_weights_iterator(
+                        hf_weights_files,
+                        self.load_config.use_tqdm_on_load,
+                    )
+        else:
+            if extra_config.get("enable_multithread_load"):
+                weights_iterator = multi_thread_pt_weights_iterator(
                     hf_weights_files,
                     self.load_config.use_tqdm_on_load,
+                    self.load_config.pt_load_map_location,
+                    max_workers=extra_config.get("num_threads",
+                                                 self.DEFAULT_NUM_THREADS),
+                )
+            else:
+                weights_iterator = pt_weights_iterator(
+                    hf_weights_files,
+                    self.load_config.use_tqdm_on_load,
+                    self.load_config.pt_load_map_location,
                 )
-        else:
-            weights_iterator = pt_weights_iterator(
-                hf_weights_files,
-                self.load_config.use_tqdm_on_load,
-                self.load_config.pt_load_map_location,
-            )
 
         if current_platform.is_tpu():
             from vllm.platforms.tpu import USE_TPU_COMMONS
diff --git a/vllm/model_executor/model_loader/weight_utils.py b/vllm/model_executor/model_loader/weight_utils.py
index 50056038b650..a4eda36148d7 100644
--- a/vllm/model_executor/model_loader/weight_utils.py
+++ b/vllm/model_executor/model_loader/weight_utils.py
@@ -1,6 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Utilities for downloading and initializing model weights."""
+import concurrent.futures
 import fnmatch
 import glob
 import hashlib
@@ -531,6 +532,36 @@ def safetensors_weights_iterator(
                 yield name, param
 
 
+def multi_thread_safetensors_weights_iterator(
+    hf_weights_files: list[str],
+    use_tqdm_on_load: bool,
+    max_workers: int = 4,
+) -> Generator[tuple[str, torch.Tensor], None, None]:
+    """Multi-Thread iterate over the weights in the model safetensor files."""
+
+    def _load_file(st_file: str):
+        result = load_file(st_file, device="cpu")
+        return result
+
+    with concurrent.futures.ThreadPoolExecutor(
+            max_workers=max_workers) as executor:
+        futures = [
+            executor.submit(_load_file, st_file)
+            for st_file in hf_weights_files
+        ]
+        futures_iter = tqdm(
+            concurrent.futures.as_completed(futures),
+            total=len(hf_weights_files),
+            desc="Multi-thread loading shards",
+            disable=not enable_tqdm(use_tqdm_on_load),
+            bar_format=_BAR_FORMAT,
+        )
+
+        for future in futures_iter:
+            state_dict = future.result()
+            yield from state_dict.items()
+
+
 def runai_safetensors_weights_iterator(
     hf_weights_files: list[str],
     use_tqdm_on_load: bool,
@@ -611,6 +642,39 @@ def pt_weights_iterator(
         del state
 
 
+def multi_thread_pt_weights_iterator(
+    hf_weights_files: list[str],
+    use_tqdm_on_load: bool,
+    pt_load_map_location: Union[str, dict[str, str]] = "cpu",
+    max_workers: int = 4,
+) -> Generator[tuple[str, torch.Tensor], None, None]:
+    """Multi-Thread iterate over the weights in the model bin/pt files."""
+
+    def _load_file(bin_file: str):
+        return torch.load(bin_file,
+                          map_location=pt_load_map_location,
+                          weights_only=True)
+
+    with concurrent.futures.ThreadPoolExecutor(
+            max_workers=max_workers) as executor:
+        futures = [
+            executor.submit(_load_file, bin_file)
+            for bin_file in hf_weights_files
+        ]
+        futures_iter = tqdm(
+            concurrent.futures.as_completed(futures),
+            total=len(hf_weights_files),
+            desc="Multi-thread loading pt checkpoint shards",
+            disable=not enable_tqdm(use_tqdm_on_load),
+            bar_format=_BAR_FORMAT,
+        )
+
+        for future in futures_iter:
+            state = future.result()
+            yield from state.items()
+            del state
+
+
 def get_gguf_extra_tensor_names(
         gguf_file: str, gguf_to_hf_name_map: dict[str, str]) -> list[str]:
     reader = gguf.GGUFReader(gguf_file)

```
