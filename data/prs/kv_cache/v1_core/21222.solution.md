# PR #21222: [Core] Introduce popleft_n and append_n in FreeKVCacheBlockQueue to further optimize block_pool

- field: kv_cache  | area: v1_core
- author: Jialin
- state: MERGED  created: 2025-07-19T09:39:37Z  merged: 2025-07-22T13:17:47Z
- url: https://github.com/vllm-project/vllm/pull/21222
- labels: performance, ready, v1

# Issue

## Problem

FreeKVCacheBlockQueue only supported single-element popleft and append. Both get_new_blocks and free_blocks ran these one-at-a-time in a loop, so each call performed O(n) individual doubly-linked-list splices. Every single popleft re-wired the fake head to the next block and every single append re-wired the fake tail, doing roughly twice the pointer updates actually needed for a bulk operation.

## Example

Popping N blocks did N separate head reconnections even though intermediate blocks are discarded immediately; benchmarked get_new_blocks took 0.15ms and free_blocks took 33us before the change.

## Use case

Allocating KV cache blocks on the prefill/forward path and freeing them at end of each decode cycle — both sit on the critical path that hard-blocks token generation.

## Resources

CPU time / per-step scheduler overhead; linked-list pointer operations on the hot allocation/free path that gate GPU forward progress.

## Type of users

Anyone serving with the v1 engine at high request throughput, where per-step block allocate/free overhead accumulates across many concurrent sequences.

## Kind of inputs

High batch sizes and many requests needing large numbers of KV blocks allocated and freed every scheduler step.

## Mechanism with the issue

In block_pool.get_new_blocks a while loop called free_block_queue.popleft() per block, and each popleft reconnected fake_free_list_head to the second block — redundant since that block may be popped next. Likewise free_blocks called append() per block, each re-wiring fake_free_list_tail. The linked-list splice count scaled at ~2x the necessary work for a batch.

# Solution

## Fix in words

Adds popleft_n(n) and append_n(blocks) to FreeKVCacheBlockQueue that walk the list once and rewire the fake head/tail a single time for the whole batch. get_new_blocks now calls popleft_n(num_blocks) then iterates the returned list once to evict (if caching) and bump ref_cnt inline, instead of the per-block popleft loop. free_blocks materializes the iterable, decrements ref_cnt in one pass, then append_n's the blocks that reached ref_cnt 0 and are non-null. popleft_n detaches popped blocks' prev/next pointers and connects the fake head to the new first block once; append_n chains the new blocks and connects the last one to the fake tail once. Net effect: linked-list operations cut roughly in half — get_new_blocks 0.15ms→0.008ms, free_blocks 33us→9us.

## Author's solution

Introduce bulk popleft_n and append_n operators that exploit the known total count to perform the linked-list manipulation in a single pass, rather than popping/appending one block at a time. Because single-element popleft/append each do an unnecessary fake-head/tail reconnection, batching halves the number of linked-list operations. New unit tests cover append_n and popleft_n, and benchmarks confirm get_blocks and free_blocks times are cut in half.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `tests/v1/core/test_kv_cache_utils.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/core/kv_cache_utils.py`

### Diff
```diff
diff --git a/tests/v1/core/test_kv_cache_utils.py b/tests/v1/core/test_kv_cache_utils.py
index 68b060156901..ccdbe79dfea4 100644
--- a/tests/v1/core/test_kv_cache_utils.py
+++ b/tests/v1/core/test_kv_cache_utils.py
@@ -184,6 +184,111 @@ def test_free_kv_cache_block_queue_operations():
     assert str(e.value) == "No free blocks available"
 
 
+def test_free_kv_cache_block_queue_append_n():
+    # Create an empty FreeKVCacheBlockQueue with these blocks
+    queue = FreeKVCacheBlockQueue([])
+    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
+    # Append 0 block
+    # fake_head->fake_tail
+    queue.append_n([])
+    assert queue.num_free_blocks == 0
+    assert (queue.fake_free_list_head.next_free_block
+            is queue.fake_free_list_tail)
+    assert (queue.fake_free_list_tail.prev_free_block
+            is queue.fake_free_list_head)
+    # Append 1 block
+    # fake_head->b0->fake_tail
+    queue.append_n(blocks[0:1])
+    assert queue.num_free_blocks == 1
+    assert queue.fake_free_list_head.next_free_block is blocks[0]
+    assert blocks[0].prev_free_block is queue.fake_free_list_head
+    assert blocks[0].next_free_block is queue.fake_free_list_tail
+    assert queue.fake_free_list_tail.prev_free_block is blocks[0]
+    # Append 2 blocks
+    # fake_head->b0->b4->b5->fake_tail
+    queue.append_n(blocks[4:6])
+    assert queue.num_free_blocks == 3
+    assert queue.fake_free_list_head.next_free_block is blocks[0]
+    assert blocks[0].prev_free_block is queue.fake_free_list_head
+    assert blocks[0].next_free_block is blocks[4]
+    assert blocks[4].prev_free_block is blocks[0]
+    assert blocks[4].next_free_block is blocks[5]
+    assert blocks[5].prev_free_block is blocks[4]
+    assert blocks[5].next_free_block is queue.fake_free_list_tail
+    assert queue.fake_free_list_tail.prev_free_block is blocks[5]
+    # Append 3 blocks
+    # fake_head->b0->b4->b5->b1->b2->b3->fake_tail
+    queue.append_n(blocks[1:4])
+    assert queue.num_free_blocks == 6
+    assert queue.fake_free_list_head.next_free_block is blocks[0]
+    assert blocks[0].prev_free_block is queue.fake_free_list_head
+    assert blocks[0].next_free_block is blocks[4]
+    assert blocks[4].prev_free_block is blocks[0]
+    assert blocks[4].next_free_block is blocks[5]
+    assert blocks[5].prev_free_block is blocks[4]
+    assert blocks[5].next_free_block is blocks[1]
+    assert blocks[1].prev_free_block is blocks[5]
+    assert blocks[1].next_free_block is blocks[2]
+    assert blocks[2].prev_free_block is blocks[1]
+    assert blocks[2].next_free_block is blocks[3]
+    assert blocks[3].prev_free_block is blocks[2]
+    assert blocks[3].next_free_block is queue.fake_free_list_tail
+    assert queue.fake_free_list_tail.prev_free_block is blocks[3]
+
+
+def test_free_kv_cache_block_queue_popleft_n():
+    blocks = [KVCacheBlock(block_id=i) for i in range(6)]
+    # Create a empty FreeKVCacheBlockQueue with these blocks
+    queue = FreeKVCacheBlockQueue(
+        [blocks[1], blocks[3], blocks[5], blocks[4], blocks[0], blocks[2]])
+    assert queue.num_free_blocks == 6
+    assert queue.fake_free_list_head.next_free_block is blocks[1]
+    assert blocks[1].prev_free_block is queue.fake_free_list_head
+    assert blocks[1].next_free_block is blocks[3]
+    assert blocks[3].prev_free_block is blocks[1]
+    assert blocks[3].next_free_block is blocks[5]
+    assert blocks[5].prev_free_block is blocks[3]
+    assert blocks[5].next_free_block is blocks[4]
+    assert blocks[4].prev_free_block is blocks[5]
+    assert blocks[4].next_free_block is blocks[0]
+    assert blocks[0].prev_free_block is blocks[4]
+    assert blocks[0].next_free_block is blocks[2]
+    assert blocks[2].prev_free_block is blocks[0]
+    assert blocks[2].next_free_block is queue.fake_free_list_tail
+    assert queue.fake_free_list_tail.prev_free_block is blocks[2]
+
+    # Pop 0 block
+    # fake_head->b1->b3->b5->b4->b0->b2->fake_tail
+    assert len(queue.popleft_n(0)) == 0
+    # Pop 1 block
+    # fake_head->b3->b5->b4->b0->b2->fake_tail
+    result_blocks = queue.popleft_n(1)
+    assert len(result_blocks) == 1
+    assert result_blocks[0] is blocks[1]
+    for block in result_blocks:
+        assert block.prev_free_block is None
+        assert block.next_free_block is None
+    # Pop 2 blocks
+    # fake_head->b4->b0->b2->fake_tail
+    result_blocks = queue.popleft_n(2)
+    assert len(result_blocks) == 2
+    assert result_blocks[0] is blocks[3]
+    assert result_blocks[1] is blocks[5]
+    for block in result_blocks:
+        assert block.prev_free_block is None
+        assert block.next_free_block is None
+    # Pop 3 blocks
+    # fake_head->fake_tail
+    result_blocks = queue.popleft_n(3)
+    assert len(result_blocks) == 3
+    assert result_blocks[0] is blocks[4]
+    assert result_blocks[1] is blocks[0]
+    assert result_blocks[2] is blocks[2]
+    for block in result_blocks:
+        assert block.prev_free_block is None
+        assert block.next_free_block is None
+
+
 def test_free_kv_cache_block_queue_get_all_free_blocks():
     # Create a list of KVCacheBlock objects
     blocks = [KVCacheBlock(block_id=i) for i in range(5)]
diff --git a/vllm/v1/core/block_pool.py b/vllm/v1/core/block_pool.py
index d21f94727cf6..3fed0048b9fe 100644
--- a/vllm/v1/core/block_pool.py
+++ b/vllm/v1/core/block_pool.py
@@ -214,21 +214,18 @@ def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
             raise ValueError(
                 f"Cannot get {num_blocks} free blocks from the pool")
 
-        ret: list[KVCacheBlock] = []
-        idx = 0
-        while idx < num_blocks:
-            # First allocate blocks.
-            curr_block = self.free_block_queue.popleft()
-            assert curr_block.ref_cnt == 0
-
-            # If the block is cached, evict it.
-            if self.enable_caching:
-                self._maybe_evict_cached_block(curr_block)
-
-            curr_block.incr_ref()
-            ret.append(curr_block)
-            idx += 1
-
+        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
+
+        # In order to only iterate the list once, we duplicated code a bit
+        if self.enable_caching:
+            for block in ret:
+                self._maybe_evict_cached_block(block)
+                assert block.ref_cnt == 0
+                block.ref_cnt += 1
+        else:
+            for block in ret:
+                assert block.ref_cnt == 0
+                block.ref_cnt += 1
         return ret
 
     def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
@@ -284,11 +281,14 @@ def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
             ordered_blocks: A list of blocks to free ordered by their eviction
                 priority.
         """
-        for block in ordered_blocks:
-            block.decr_ref()
-            # null_block should not be added to the free list.
-            if block.ref_cnt == 0 and not block.is_null:
-                self.free_block_queue.append(block)
+        # Materialize the iterable to allow multiple passes.
+        blocks_list = list(ordered_blocks)
+        for block in blocks_list:
+            block.ref_cnt -= 1
+        self.free_block_queue.append_n([
+            block for block in blocks_list
+            if block.ref_cnt == 0 and not block.is_null
+        ])
 
     def reset_prefix_cache(self) -> bool:
         """Reset prefix cache. This function may be used in RLHF
diff --git a/vllm/v1/core/kv_cache_utils.py b/vllm/v1/core/kv_cache_utils.py
index 457d95cc738b..198d79cfb420 100644
--- a/vllm/v1/core/kv_cache_utils.py
+++ b/vllm/v1/core/kv_cache_utils.py
@@ -154,6 +154,8 @@ class KVCacheBlock:
     # Whether the block is a null block that should never be cached.
     is_null: bool = False
 
+    # TODO(Jialin): For performance, let callers handle ref_cnt bumps to
+    # avoid function calls.
     def incr_ref(self):
         self.ref_cnt += 1
 
@@ -273,6 +275,39 @@ def popleft(self) -> KVCacheBlock:
         self.num_free_blocks -= 1
         return first_block
 
+    def popleft_n(self, n: int) -> list[KVCacheBlock]:
+        """Pop the first n free blocks and reduce num_free_blocks by n.
+
+        Args:
+            n: The number of blocks to pop.
+
+        Returns:
+            A list of n free blocks.
+        """
+        if n == 0:
+            return []
+        assert self.num_free_blocks >= n
+        self.num_free_blocks -= n
+
+        curr_block = self.fake_free_list_head.next_free_block
+        # Pop n blocks from the head of the list
+        ret = []
+        for _ in range(n):
+            assert curr_block is not None
+            ret.append(curr_block)
+            last_block = curr_block
+            curr_block = curr_block.next_free_block
+            # Reset prev_free_block and next_free_block of all popped blocks
+            last_block.prev_free_block = None
+            last_block.next_free_block = None
+
+        if curr_block is not None:
+            # The queue is not empty, connect the fake head to
+            # the new first block.
+            self.fake_free_list_head.next_free_block = curr_block
+            curr_block.prev_free_block = self.fake_free_list_head
+        return ret
+
     def remove(self, block: KVCacheBlock) -> None:
         """Remove a block in the free list and reduce num_free_blocks by 1.
 
@@ -315,6 +350,29 @@ def append(self, block: KVCacheBlock) -> None:
 
         self.num_free_blocks += 1
 
+    def append_n(self, blocks: list[KVCacheBlock]) -> None:
+        """Put a list of blocks back into the free list
+
+        Args:
+            blocks: The blocks to append.
+        """
+        if len(blocks) == 0:
+            return
+        self.num_free_blocks += len(blocks)
+
+        last_block = self.fake_free_list_tail.prev_free_block
+        assert last_block is not None, (
+            "prev_free_block of fake_free_list_tail should always exist")
+        # Add inter-connections between consecutive blocks
+        for block in blocks:
+            block.prev_free_block = last_block
+            last_block.next_free_block = block
+            last_block = block
+
+        # Connect the last block of <blocks> to the fake tail
+        last_block.next_free_block = self.fake_free_list_tail
+        self.fake_free_list_tail.prev_free_block = last_block
+
     def get_all_free_blocks(self) -> list[KVCacheBlock]:
         """Get all free blocks in the free list. Mainly used for testing.
 

```
