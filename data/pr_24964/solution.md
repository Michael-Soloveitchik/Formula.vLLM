# PR #24964: [Core] Reuse empty block lists whenever possible in KVCacheBlocks to mitigate GC costs

- field: scheduling  | area: v1_core
- author: Jialin
- state: MERGED  created: 2025-09-16T12:31:33Z  merged: 2025-10-14T19:58:43Z
- url: https://github.com/vllm-project/vllm/pull/24964
- labels: frontend, ready, v1, multi-modality

# Issue

## Problem

In the vLLM v1 KV cache manager, every request that produced no computed/new blocks still allocated fresh `KVCacheBlocks` dataclasses wrapping brand-new empty Python lists (one list per kv_cache_group). These short-lived empty-block objects dominated Generation 0 garbage collection, since most allocated `KVCacheBlocks` referred to empty blocks. The constant churn of allocating and collecting these objects added measurable GC overhead to the scheduler hot path.

## Example

With facebook/opt-125m under a decode-heavy workload (input 48, output 2000), nearly every decode step allocated empty `KVCacheBlocks` wrappers, generating heavy Gen-0 GC pressure; eliminating it yielded roughly 3-4% throughput improvement.

## Use case

High-throughput online serving where the scheduler repeatedly calls into the KV cache manager each step, especially long-output decode-heavy generation.

## Resources

CPU time spent in Python garbage collection (Gen-0 collection), which steals cycles from the scheduler's per-step critical path.

## Type of users

Users running vLLM v1 engine at scale for throughput-sensitive serving, particularly decode-heavy (long output) and prefill-heavy workloads.

## Kind of inputs

Requests with no new computed blocks or no newly allocated blocks — e.g. cache-miss prefills, prompt_logprobs requests, and the many decode steps that allocate nothing new. Triggered by large request volume and long output lengths.

## Mechanism with the issue

`KVCacheManager` methods (`get_computed_blocks`, `allocate_slots`, `get_blocks`, scheduler's WAITING path) constructed `KVCacheBlocks(tuple([] for _ in range(num_groups)))` on every call via `create_empty_block_list()`. Each call minted new list objects even when empty, so GC saw a flood of transient empty-block containers.

# Solution

## Fix in words

The diff pre-constructs a single immutable `empty_kv_cache_blocks` member in `KVCacheManager.__init__` (nested empty tuples `()` instead of lists, per kv_cache_group) and reuses it everywhere an empty result was previously freshly allocated. `create_empty_block_list()` is replaced by `create_kv_cache_blocks()`, which returns the shared `empty_kv_cache_blocks` when `any(blocks)` is false and only allocates a new `KVCacheBlocks` for non-empty block lists. `KVCacheBlocks.blocks` is retyped from `tuple[list[...], ...]` to `tuple[Sequence[...], ...]` (propagated through `block_pool.touch`, `kv_cache_coordinator`, and `single_type_kv_cache_manager` signatures) to allow immutable empty tuples, and `__add__`/`new_empty` updated to use `itertools.chain` and empty tuples. Effect: empty-block objects are no longer churned, cutting Gen-0 GC cost and gaining ~3-4% throughput.

## Author's solution

The author made two changes: (1) changed `KVCacheBlocks.blocks` from `tuple[list]` to `tuple[tuple]` to signal immutability after creation, and (2) pre-constructed a single empty `KVCacheBlocks` member variable in `KVCacheManager` to be reused instead of allocating new empty block objects, thereby avoiding GC overhead. Design discussion with @heheda12345 aimed to minimize code-signature changes and avoid if/else branching, ideally via a static empty-list representation.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`

### Diff
```diff
diff --git a/vllm/v1/core/block_pool.py b/vllm/v1/core/block_pool.py
index cd22db410a6e..15c06a0b107d 100644
--- a/vllm/v1/core/block_pool.py
+++ b/vllm/v1/core/block_pool.py
@@ -1,6 +1,6 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
-from collections.abc import Iterable
+from collections.abc import Iterable, Sequence
 from typing import Any
 
 from vllm.distributed.kv_events import (
@@ -328,7 +328,7 @@ def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
             )
         return True
 
-    def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
+    def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
         """Touch a block increases its reference count by 1, and may remove
         the block from the free queue. This is used when a block is hit by
         another request with the same prefix.
diff --git a/vllm/v1/core/kv_cache_coordinator.py b/vllm/v1/core/kv_cache_coordinator.py
index ece382277255..137e5e0cdb6d 100644
--- a/vllm/v1/core/kv_cache_coordinator.py
+++ b/vllm/v1/core/kv_cache_coordinator.py
@@ -1,6 +1,7 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 from abc import ABC, abstractmethod
+from collections.abc import Sequence
 
 from vllm.v1.core.block_pool import BlockPool
 from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
@@ -51,7 +52,7 @@ def get_num_blocks_to_allocate(
         self,
         request_id: str,
         num_tokens: int,
-        new_computed_blocks: tuple[list[KVCacheBlock], ...],
+        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
         num_encoder_tokens: int,
     ) -> int:
         """
@@ -84,7 +85,7 @@ def get_num_blocks_to_allocate(
         return num_blocks_to_allocate
 
     def save_new_computed_blocks(
-        self, request_id: str, new_computed_blocks: tuple[list[KVCacheBlock], ...]
+        self, request_id: str, new_computed_blocks: tuple[Sequence[KVCacheBlock], ...]
     ) -> None:
         """
         Add the new computed blocks to the request.
diff --git a/vllm/v1/core/kv_cache_manager.py b/vllm/v1/core/kv_cache_manager.py
index 7a1025fc2bb4..ff221048dbd1 100644
--- a/vllm/v1/core/kv_cache_manager.py
+++ b/vllm/v1/core/kv_cache_manager.py
@@ -1,6 +1,8 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
 
+import itertools
+from collections.abc import Sequence
 from dataclasses import dataclass
 from typing import Literal, overload
 
@@ -23,7 +25,7 @@ class KVCacheBlocks:
     structure from the Scheduler.
     """
 
-    blocks: tuple[list[KVCacheBlock], ...]
+    blocks: tuple[Sequence[KVCacheBlock], ...]
     """
     `blocks[i][j]` refers to the i-th kv_cache_group
     and the j-th block of tokens.We don't use block of
@@ -31,12 +33,20 @@ class KVCacheBlocks:
     kv_cache_groups have the same number of blocks, which is true for now but
     will be broken if we want to give different block_size to different
     kv_cache_groups in the future.
+
+    Each single type KVCacheBlocks could be represented as:
+    - list[KVCacheBlock] for more than one KVCacheBlock
+    - an empty tuple for requests without KVCacheBlock
+      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
     """
 
     def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
         """Adds two KVCacheBlocks instances."""
         return KVCacheBlocks(
-            tuple(blk1 + blk2 for blk1, blk2 in zip(self.blocks, other.blocks))
+            tuple(
+                list(itertools.chain(blk1, blk2))
+                for blk1, blk2 in zip(self.blocks, other.blocks)
+            )
         )
 
     @overload
@@ -74,8 +84,10 @@ def get_unhashed_block_ids(self) -> list[int]:
         return [block.block_id for block in self.blocks[0] if block.block_hash is None]
 
     def new_empty(self) -> "KVCacheBlocks":
-        """Creates a new KVCacheBlocks instance with no blocks."""
-        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))
+        """
+        Creates a new KVCacheBlocks instance with no blocks.
+        """
+        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))
 
 
 class KVCacheManager:
@@ -131,6 +143,15 @@ def __init__(
         self.block_pool = self.coordinator.block_pool
         self.kv_cache_config = kv_cache_config
 
+        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
+        # via create_kv_cache_blocks instead of creating new ones to avoid GC
+        # overhead.
+        #
+        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
+        self.empty_kv_cache_blocks = KVCacheBlocks(
+            tuple(() for _ in range(self.num_kv_cache_groups))
+        )
+
     @property
     def usage(self) -> float:
         """Get the KV cache usage.
@@ -170,7 +191,7 @@ def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
             request.sampling_params is not None
             and request.sampling_params.prompt_logprobs is not None
         ):
-            return self.create_empty_block_list(), 0
+            return self.empty_kv_cache_blocks, 0
 
         # NOTE: When all tokens hit the cache, we must recompute the last token
         # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
@@ -198,7 +219,7 @@ def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
                 self.prefix_cache_stats.queries += request.num_tokens
                 self.prefix_cache_stats.hits += num_new_computed_tokens
 
-        return KVCacheBlocks(computed_blocks), num_new_computed_tokens
+        return (self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens)
 
     def allocate_slots(
         self,
@@ -251,9 +272,7 @@ def allocate_slots(
         if new_computed_blocks is not None:
             new_computed_block_list = new_computed_blocks.blocks
         else:
-            new_computed_block_list = tuple(
-                [] for _ in range(len(self.kv_cache_config.kv_cache_groups))
-            )
+            new_computed_block_list = self.empty_kv_cache_blocks.blocks
 
         # Free the blocks that are skipped during the attention computation
         # (e.g., tokens outside the sliding window).
@@ -305,7 +324,7 @@ def allocate_slots(
         # P/D: delay caching blocks if we have to recv from
         # remote. Update state for locally cached blocks.
         if not self.enable_caching or delay_cache_blocks:
-            return KVCacheBlocks(new_blocks)
+            return self.create_kv_cache_blocks(new_blocks)
 
         # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
         # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
@@ -316,7 +335,7 @@ def allocate_slots(
         )
         self.coordinator.cache_blocks(request, num_tokens_to_cache)
 
-        return KVCacheBlocks(new_blocks)
+        return self.create_kv_cache_blocks(new_blocks)
 
     def free(self, request: Request) -> None:
         """Free the blocks allocated for the request.
@@ -388,7 +407,7 @@ def take_events(self) -> list[KVCacheEvent]:
 
     def get_blocks(self, request_id: str) -> KVCacheBlocks:
         """Get the blocks of a request."""
-        return KVCacheBlocks(self.coordinator.get_blocks(request_id))
+        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))
 
     def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
         """Get the block ids of a request."""
@@ -399,6 +418,8 @@ def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
         if self.enable_caching:
             self.coordinator.cache_blocks(request, num_computed_tokens)
 
-    def create_empty_block_list(self) -> KVCacheBlocks:
-        """Creates a new KVCacheBlocks instance with no blocks."""
-        return KVCacheBlocks(tuple([] for _ in range(self.num_kv_cache_groups)))
+    def create_kv_cache_blocks(
+        self, blocks: tuple[list[KVCacheBlock], ...]
+    ) -> KVCacheBlocks:
+        # Only create new KVCacheBlocks for non-empty blocks
+        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index cbbdf48c6e0c..27936c96e713 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -421,9 +421,7 @@ def schedule(self) -> SchedulerOutput:
                 # KVTransfer: WAITING reqs have num_computed_tokens > 0
                 # after async KV recvs are completed.
                 else:
-                    new_computed_blocks = (
-                        self.kv_cache_manager.create_empty_block_list()
-                    )
+                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                     num_new_local_computed_tokens = 0
                     num_computed_tokens = request.num_computed_tokens
 
diff --git a/vllm/v1/core/single_type_kv_cache_manager.py b/vllm/v1/core/single_type_kv_cache_manager.py
index 7984a6ce29df..586034182686 100644
--- a/vllm/v1/core/single_type_kv_cache_manager.py
+++ b/vllm/v1/core/single_type_kv_cache_manager.py
@@ -3,6 +3,7 @@
 import itertools
 from abc import ABC, abstractmethod
 from collections import defaultdict
+from collections.abc import Sequence
 
 from vllm.utils import cdiv
 from vllm.v1.core.block_pool import BlockPool
@@ -61,7 +62,10 @@ def __init__(
         self._null_block = block_pool.null_block
 
     def get_num_blocks_to_allocate(
-        self, request_id: str, num_tokens: int, new_computed_blocks: list[KVCacheBlock]
+        self,
+        request_id: str,
+        num_tokens: int,
+        new_computed_blocks: Sequence[KVCacheBlock],
     ) -> int:
         """
         Get the number of blocks needed to be allocated for the request.
@@ -93,7 +97,7 @@ def get_num_blocks_to_allocate(
         return num_new_blocks + num_evictable_computed_blocks
 
     def save_new_computed_blocks(
-        self, request_id: str, new_computed_blocks: list[KVCacheBlock]
+        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock]
     ) -> None:
         """
         Add the new computed blocks to the request.
@@ -593,7 +597,10 @@ def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
         return 0
 
     def get_num_blocks_to_allocate(
-        self, request_id: str, num_tokens: int, new_computed_blocks: list[KVCacheBlock]
+        self,
+        request_id: str,
+        num_tokens: int,
+        new_computed_blocks: Sequence[KVCacheBlock],
     ) -> int:
         # Allocate extra `num_speculative_blocks` blocks for
         # speculative decoding (MTP/EAGLE) with linear attention.
@@ -625,7 +632,7 @@ class CrossAttentionManager(SingleTypeKVCacheManager):
     """Manager for cross-attention KV cache in encoder-decoder models."""
 
     def save_new_computed_blocks(
-        self, request_id: str, new_computed_blocks: list[KVCacheBlock]
+        self, request_id: str, new_computed_blocks: Sequence[KVCacheBlock]
     ) -> None:
         # We do not cache blocks for cross-attention to be shared between
         # requests, so  `new_computed_blocks` should always be empty.

```
