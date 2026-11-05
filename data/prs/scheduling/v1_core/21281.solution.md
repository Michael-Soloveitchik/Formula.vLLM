# PR #21281: [Core] Minimize number of dict lookup in _maybe_evict_cached_block

- field: scheduling  | area: v1_core
- author: Jialin
- state: MERGED  created: 2025-07-21T05:24:50Z  merged: 2025-07-22T05:37:34Z
- url: https://github.com/vllm-project/vllm/pull/21281
- labels: ready, v1

# Issue

## Problem

`_maybe_evict_cached_block` performed redundant dictionary lookups against `self.cached_block_hash_to_block` keyed by the same `block_hash`. The original code did a membership test (`block_hash in ...`), then indexed to delete a sub-entry, then re-indexed to check length, then potentially re-indexed to delete the key — up to 4 lookups for one eviction. After PR #21222 sped up `allocate_new_block`, this redundant-lookup pattern became the new hot-path bottleneck.

## Example

Each block eviction triggered 4 dict lookups on the same key; measured cost per call was ~0.13ms, cut to ~0.03ms after collapsing to 2 lookups.

## Use case

KV-cache block eviction during prefix-cache management when the block pool allocates new blocks and must evict least-recently-used cached blocks.

## Resources

CPU time in the scheduler hot path — redundant Python dict hashing/lookups add latency to every block allocation.

## Type of users

Users running high-throughput serving where the v1 core block pool churns cached blocks frequently; anyone sensitive to scheduler-side per-step overhead.

## Kind of inputs

Workloads with heavy prefix-cache turnover — many requests causing frequent block allocation and eviction cycles.

## Mechanism with the issue

In `vllm/v1/core/block_pool.py`, `_maybe_evict_cached_block` hashed and looked up `block_hash` in `cached_block_hash_to_block` four times: once for `in` check, once for the inner `del`, once for the `len()==0` check, once for the final `del`. Repeated lookups of the same key on a hot path waste CPU cycles.

# Solution

## Fix in words

The diff replaces the membership test + repeated indexing with early returns and a single cached reference. Before: `if block_hash and block_hash in self.cached_block_hash_to_block:` followed by multiple `self.cached_block_hash_to_block[block_hash]` accesses. After: it early-returns `False` when `block_hash is None`, then does one `self.cached_block_hash_to_block.get(block_hash)` into local `blocks_by_id` (returning `False` if `None`), then mutates via `blocks_by_id.pop(block.block_id, None)` and uses the local for the emptiness check, requiring only one additional `del self.cached_block_hash_to_block[block_hash]`. This cuts same-key dict lookups from ~4 to ~2, lowering per-call cost from 0.13ms to 0.03ms.

## Author's solution

Author noticed `_maybe_evict_cached_block` did duplicated key lookups against the same dict 4 times and reduced it to 2 by caching the sub-dict in a local variable and using early returns. Verified via trace comparison on top of PR #21222, showing 0.13ms -> 0.03ms improvement.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/block_pool.py`

### Diff
```diff
diff --git a/vllm/v1/core/block_pool.py b/vllm/v1/core/block_pool.py
index d21f94727cf6..0fd6947ae0bd 100644
--- a/vllm/v1/core/block_pool.py
+++ b/vllm/v1/core/block_pool.py
@@ -243,22 +243,27 @@ def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
             True if the block is evicted, False otherwise.
         """
         block_hash = block.block_hash
-        if block_hash and block_hash in self.cached_block_hash_to_block:
-            block.reset_hash()
-            del self.cached_block_hash_to_block[block_hash][block.block_id]
-
-            if len(self.cached_block_hash_to_block[block_hash]) == 0:
-                del self.cached_block_hash_to_block[block_hash]
-
-            if self.enable_kv_cache_events:
-                # FIXME (Chen): Not sure whether we should return `hash_value`
-                # or `(hash_value, group_id)` here. But it's fine now because
-                # we disable hybrid kv cache manager when kv cache event is
-                # enabled, so there is only one group.
-                self.kv_event_queue.append(
-                    BlockRemoved(block_hashes=[block_hash.get_hash_value()]))
-            return True
-        return False
+        if block_hash is None:
+            # The block doesn't have hash, eviction is not needed
+            return False
+        blocks_by_id = self.cached_block_hash_to_block.get(block_hash)
+        if blocks_by_id is None:
+            # block_hash not found in cached_block_hash_to_block,
+            # eviction is not needed
+            return False
+        block.reset_hash()
+        blocks_by_id.pop(block.block_id, None)
+        if blocks_by_id:
+            del self.cached_block_hash_to_block[block_hash]
+
+        if self.enable_kv_cache_events:
+            # FIXME (Chen): Not sure whether we should return `hash_value`
+            # or `(hash_value, group_id)` here. But it's fine now because
+            # we disable hybrid kv cache manager when kv cache event is
+            # enabled, so there is only one group.
+            self.kv_event_queue.append(
+                BlockRemoved(block_hashes=[block_hash.get_hash_value()]))
+        return True
 
     def touch(self, blocks: tuple[list[KVCacheBlock], ...]) -> None:
         """Touch a block increases its reference count by 1, and may remove

```
