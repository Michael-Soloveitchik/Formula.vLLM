# PR #36012: [Performance] Add prefetch for checkpoints to OS page cache (#36012)

# Issue

## Original PR description (captured source)

## Purpose

Add a new `safetensors_load_strategy` option **"prefetch"** to speed up the model loading phase. Prefetch is designed for speeding up the model loading step in case a model is fully on disk and not yet in RAM. When enabled, checkpoint files are preloaded into the OS page cache (via sequential reads in 16MB blocks) in a background thread in parallel with workers parsing tensors from those files. This overlaps prefetch I/O with tensor parsing and is useful on network storage or high-latency storage.

## Test Result

Testing was done on model `deepseek-ai/DeepSeek-V3-0324` on 8xB200 with:

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 -tp 8 --safetensors-load-strategy prefetch
```

Results on two internal servers:

| Server   | Weights loading time without prefetch (s) | Weights loading time with prefetch (s) |
|----------|-------------------------------------------|----------------------------------------|
| Server 1 | 1500                                      | 435                                    |
| Server 2 | 3660                                      | 215                                    |

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>


## Full discussion

See [36012.md](./36012.md) and [36012.json](./36012.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/36012
- Diff retrieved from: https://api.github.com/repos/vllm-project/vllm/pulls/36012
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `a56986ee2a06edef5307d38c4a7097ee47ed72e1d31b73faf4d9e38c80cee2a5`

Existing semantic notes in [36012.sem.json](./36012.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/config/load.py`
- `vllm/model_executor/model_loader/weight_utils.py`

### Diff

```diff
diff --git a/vllm/config/load.py b/vllm/config/load.py
index b771556d83f2..c36c1adfed89 100644
--- a/vllm/config/load.py
+++ b/vllm/config/load.py
@@ -62,6 +62,9 @@ class LoadConfig:
       This is recommended for models on network filesystems (e.g., Lustre, NFS)
       as it avoids inefficient random reads, significantly speeding up model
       initialization. However, it uses more CPU RAM.
+    - "prefetch": Checkpoint files are read into the OS page cache before
+      workers load them, speeding up the model loading phase. Useful on
+      network or high-latency storage.
     - "torchao": Weights are loaded in upfront and then reconstructed
       into torchao tensor subclasses. This is used when the checkpoint
       was quantized using torchao and saved using safetensors.
diff --git a/vllm/model_executor/model_loader/weight_utils.py b/vllm/model_executor/model_loader/weight_utils.py
index 0a67a6a42aba..dd4bf636e0af 100644
--- a/vllm/model_executor/model_loader/weight_utils.py
+++ b/vllm/model_executor/model_loader/weight_utils.py
@@ -2,6 +2,7 @@
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 """Utilities for downloading and initializing model weights."""
 
+import asyncio
 import concurrent.futures
 import fnmatch
 import glob
@@ -9,6 +10,7 @@
 import json
 import os
 import tempfile
+import threading
 import time
 from collections import defaultdict
 from collections.abc import Callable, Generator
@@ -720,6 +722,71 @@ def np_cache_weights_iterator(
         yield name, torch.from_numpy(param)
 
 
+def _prefetch_checkpoint(file_path: str) -> None:
+    """Prefetch a checkpoint file into the OS page cache.
+
+    Reads the file in 16MB blocks so the kernel caches its pages before
+    workers load the same file.
+    """
+    block_size = 16 * 1024 * 1024  # 16MB
+    with open(file_path, "rb") as f:
+        while f.read(block_size):
+            pass
+
+
+def _prefetch_all_checkpoints(sorted_files: list[str]) -> None:
+    """Start prefetching checkpoint files into page cache in a background thread."""
+    if torch.distributed.is_initialized():
+        rank = torch.distributed.get_rank()
+        world_size = torch.distributed.get_world_size()
+    else:
+        rank = 0
+        world_size = 1
+    num_prefetch_threads = 8
+    paths_to_prefetch = sorted_files[rank::world_size]
+    total_for_rank = len(paths_to_prefetch)
+
+    async def _prefetch_all() -> None:
+        semaphore = asyncio.Semaphore(num_prefetch_threads)
+        completed = 0
+        next_log_pct = 10
+
+        async def prefetch_one(path: str) -> None:
+            nonlocal completed, next_log_pct
+            try:
+                async with semaphore:
+                    await asyncio.to_thread(_prefetch_checkpoint, path)
+                completed += 1
+                if total_for_rank > 0 and next_log_pct <= 100:
+                    pct = 100 * completed / total_for_rank
+                    if pct >= next_log_pct:
+                        logger.info(
+                            "Prefetching checkpoint files: %d%% (%d/%d)",
+                            next_log_pct,
+                            completed,
+                            total_for_rank,
+                        )
+                        next_log_pct += 10
+            except Exception:
+                logger.warning(
+                    "Failed to prefetch checkpoint file %r.", path, exc_info=True
+                )
+
+        await asyncio.gather(*(prefetch_one(p) for p in paths_to_prefetch))
+
+    def _run_prefetch() -> None:
+        start = time.perf_counter()
+        asyncio.run(_prefetch_all())
+        elapsed = time.perf_counter() - start
+        logger.info(
+            "Prefetching checkpoint files into page cache finished in %.2fs",
+            elapsed,
+        )
+
+    logger.info("Prefetching checkpoint files into page cache started (in background)")
+    threading.Thread(target=_run_prefetch, daemon=True).start()
+
+
 def safetensors_weights_iterator(
     hf_weights_files: list[str],
     use_tqdm_on_load: bool,
@@ -736,9 +803,14 @@ def safetensors_weights_iterator(
     if safetensors_load_strategy == "eager":
         loading_desc += " (eager)"
 
+    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)
+
+    if safetensors_load_strategy == "prefetch":
+        _prefetch_all_checkpoints(sorted_files)
+
     leftover_state_dict: dict[str, torch.Tensor] = {}
     for st_file in tqdm(
-        sorted(hf_weights_files, key=_natural_sort_key),
+        sorted_files,
         desc=loading_desc,
         disable=not enable_tqdm(use_tqdm_on_load),
         bar_format=_BAR_FORMAT,
```
