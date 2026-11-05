# PR #33839: [Kernel][perf] optimize NCCL symm_mem vs custom_AR selection thresholds

- field: gpu_kernels  | area: distributed_device_communicators
- author: pkousha
- state: MERGED  created: 2026-02-04T22:39:30Z  merged: 2026-02-26T22:35:58Z
- url: https://github.com/vllm-project/vllm/pull/33839
- labels: performance, ready

# Issue

## Problem

The selection logic for NCCL symmetric-memory allreduce vs custom allreduce used a single lower-bound size threshold per world size, only routing to symm_mem when a tensor met-or-exceeded that threshold (≥2MB for 4 GPUs, ≥1MB for 8 GPUs). This left all small tensors on custom_AR even though symm_mem is measurably faster there, so a whole latency-sensitive size band ran on the slower path. The threshold model could not express a non-monotonic winner pattern where symm_mem wins at both extremes but loses in the middle.

## Example

On 8x H100, a 2KB allreduce took 6.8μs on custom_AR vs 4.6μs on PyNCCL-symm (1.48x), yet the old code forced custom_AR because 2KB < 1MB threshold. Same story at 16KB (7.2μs vs 5.3μs, 1.35x).

## Use case

Multi-GPU tensor-parallel inference on NVLink-connected single nodes, where every decode/prefill step issues allreduce collectives whose latency directly adds to token latency.

## Resources

NVLink bandwidth and collective launch/latency overhead; small-tensor allreduce is latency-bound, large-tensor is bandwidth-bound.

## Type of users

Users running TP across 4+ GPUs on H100/GB200 single nodes with symmetric memory enabled (VLLM_USE_NCCL_SYMM_MEM=1, NCCL_NVLS_ENABLE=1).

## Kind of inputs

Allreduce tensors across a wide size range (2KB–1GB); specifically the small band (≤16KB) that the old threshold excluded, plus mid-range (32K–256K) where custom_AR still wins.

## Mechanism with the issue

`should_nccl_symm_mem_allreduce` in all_reduce_utils.py read a per-world-size `thresholds` dict and returned True only when `input_tensor.nbytes >= threshold`. A single lower bound cannot encode 'win for small AND large but lose in the middle', so small tensors below the threshold always fell through to custom_AR despite being faster on symm_mem.

# Solution

## Fix in words

The diff replaces the single-value `thresholds` dict with `custom_ar_preferred_ranges`, a per-world-size `(lower_bound, upper_bound)` tuple marking the mid-range band where custom_AR wins: (16KiB, 512KiB) for 4 GPUs and (16KiB, 128KiB) for 8 GPUs. `should_nccl_symm_mem_allreduce` now looks up this range and returns `tensor_size <= lower_bound or tensor_size >= upper_bound`, so symm_mem is used for both small and large tensors and custom_AR only for the interior band. Adds a `KiB` constant and docstrings explaining the benchmark-derived cutoffs. Net effect: small tensors (≤16KB) now route to symm_mem for 1.21x–1.48x speedup while mid-range still uses custom_AR, and world_size >8 still always uses symm_mem.

## Author's solution

Replace the simple size threshold with a range-based approach that defines the mid-range band where custom_AR is preferred, enabling symm_mem for both small and large tensor sizes. The `custom_ar_preferred_ranges` config holds (16KiB, 512KiB) for 4 GPUs and (16KiB, 128KiB) for 8 GPUs, derived from H100 and GB200 benchmarks showing symm_mem wins outside these ranges.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/distributed/device_communicators/all_reduce_utils.py`

### Diff
```diff
diff --git a/vllm/distributed/device_communicators/all_reduce_utils.py b/vllm/distributed/device_communicators/all_reduce_utils.py
index ff2d7436b270..3c347ef756d4 100644
--- a/vllm/distributed/device_communicators/all_reduce_utils.py
+++ b/vllm/distributed/device_communicators/all_reduce_utils.py
@@ -27,6 +27,7 @@
 
 logger = init_logger(__name__)
 
+KiB = 1024
 MiB = 1024 * 1024
 # Max size for each world size in case symmetric memory is available
 # For different SM architectures
@@ -60,17 +61,44 @@
     },
 }
 
+# NCCL symmetric memory allreduce configuration based on H100 and GB200 benchmarks.
+# PyNCCL-symm outperforms custom_AR for small and large tensor sizes,
+# while custom_AR wins for mid-range sizes.
+#
+# Benchmark results (8 GPUs):
+#   2K - 16K:   PyNCCL-symm wins (1.35x - 1.48x faster)
+#   32K - 64K:  custom_AR wins
+#   128K - 1G:  PyNCCL-symm wins (1.12x - 6.14x faster)
+#
+# Benchmark results (4 GPUs):
+#   2K - 16K:   PyNCCL-symm wins (1.21x - 1.30x faster)
+#   32K - 256K: custom_AR wins (1.07x - 1.35x faster)
+#   512K - 1G:  PyNCCL-symm wins (1.10x - 2.32x faster)
+#
+# The config defines ranges where custom_AR is preferred (symm_mem disabled).
 NCCL_SYMM_MEM_ALL_REDUCE_CONFIG: dict[str, Any] = {
     "min_world_size": 4,
-    "thresholds": {
-        4: 2 * MiB,  # 2 MB
-        8: 1 * MiB,  # 1 MB
+    # Ranges where custom_AR outperforms NCCL symm_mem: (lower_bound, upper_bound)
+    # NCCL symm_mem will NOT be used for sizes in range: lower < size < upper
+    "custom_ar_preferred_ranges": {
+        4: (16 * KiB, 512 * KiB),  # custom_AR wins for 32K-256K
+        8: (16 * KiB, 128 * KiB),  # custom_AR wins for 32K-64K
     },
     "always_use_above_world_size": 8,  # Always use symm mem for world_size > 8
 }
 
 
 def should_nccl_symm_mem_allreduce(world_size: int, input_tensor: torch.Tensor) -> bool:
+    """
+    Determine if NCCL symmetric memory allreduce should be used.
+
+    Based on H100 and GB200 benchmarks, NCCL symm_mem is preferred for:
+    - Small tensors (≤16K): Lower latency than custom_AR
+    - Large tensors (≥128K for 8 GPUs, ≥512K for 4 GPUs): Better bandwidth
+
+    Custom_AR is preferred for mid-range sizes where its P2P approach
+    has lower overhead than the symm_mem copy-in/copy-out pattern.
+    """
     from vllm.distributed.device_communicators.pynccl_allocator import (
         is_symmetric_memory_enabled,
     )
@@ -80,11 +108,20 @@ def should_nccl_symm_mem_allreduce(world_size: int, input_tensor: torch.Tensor)
 
     if not is_symmetric_memory_enabled():
         return False
+
     if world_size < NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["min_world_size"]:
         return False
-    threshold = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["thresholds"].get(world_size)
-    if threshold is not None and input_tensor.nbytes >= threshold:
-        return True
+
+    tensor_size = input_tensor.nbytes
+    custom_ar_range = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["custom_ar_preferred_ranges"].get(
+        world_size
+    )
+
+    if custom_ar_range is not None:
+        lower_bound, upper_bound = custom_ar_range
+        # Use symm_mem for small sizes (≤ lower_bound) and large sizes (≥ upper_bound)
+        # Use custom_AR (not symm_mem) for mid-range sizes
+        return tensor_size <= lower_bound or tensor_size >= upper_bound
     return world_size > NCCL_SYMM_MEM_ALL_REDUCE_CONFIG["always_use_above_world_size"]
 
 

```
