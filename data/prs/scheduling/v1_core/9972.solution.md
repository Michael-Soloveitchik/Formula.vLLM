# PR #9972: [V1] Prefix caching (take 2) (#9972)

# Issue

## Original PR description (captured source)

This PR adds prefix caching to V1 (take 2). Take 1 is in #9668.
The main difference in take 2 is we adopt a custom doubly linked list to operate free blocks with eviction. This doubly linked list has the following features over the Python builtin `deque`:
- It supports `.remove()` operator in O(1) time.
- It does not allocate any new Python objects, but directly manipulates the given objects with pointers.

## Benchmarks

### Offline Batching

```
VLLM_USE_V1=1 python3 benchmarks/benchmark_prefix_caching.py \
--model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 \
--num-prompts 200 --repeat-count 2 \
--input-length-range 256:512 \
--dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
--seed 0 [--enable-prefix-caching]
```

Version | Input (tok/s) | Output (tok/s) | Cost Time (s)
--|--|--|--
main (598b6d7) | 17916.36 | 485.64 | 8.49
This PR w/o cache | 17749.25 | 481.11 | 8.57
This PR w. cache (49%) | 32258.08 | 874.38 | 4.83

### Online Serving

Server
```
VLLM_USE_V1=1 vllm serve neuralmagic/Meta-Llama-3-8B-Instruct-FP8 --disable-log-requests [--enable-prefix-caching]
```

Client
```
PREFIX_LEN = 550 * hit_rate
INPUT_LEN = 550 - PREFIX_LEN

python3 benchmarks/benchmark_serving.py --backend vllm \
--model neuralmagic/Meta-Llama-3-8B-Instruct-FP8 \
--dataset-name random --random-input-len $INPUT_LEN --random-output-len 150 \
--random-prefix-len $PREFIX_LEN --seed 0 --request-rate 8 --num-prompts 500
```

Hit Rate | MeanTTFT | MeanTPOT
--|--|--
main (598b6d7) | 107.24 | 28.99
Disable | 110.23 | 29.14
0% | 107.97 | 28.97
20%| 87.73 | 26.21
40%| 79.63 | 25.50
60%| 70.93 | 24.50
80%| 67.61 | 25.22

## Data Structure

**The same as Take 1.**

- Block pool: A pool of kv-cache blocks corresponding to block IDs that will be used in the entire engine lifecycle.
- Free block queue: A queue of free blocks to be allocated. The blocks in this queue may be able to be reused (cache hit) by other requests.
- Cached block map: Mapping from block hash to a list of blocks. The reason to have a list of blocks is we don't do de-duplication (see "Duplication" below for details). When cache hit, we always allocate the first block in the list to aggregate the references.

## Algorithms

**Almost the same as Take 1 except for not lazy removal, because we now support remove in O(1) time.**

### Allocate Slots

When a request is scheduled for the first time, `allocate_slots()` is used to allocate blocks based on the current scheduled prompt tokens. If the prompt is chunked due to chunked prefill, we will only allocate blocks for the scheduled tokens. In addition to the scheduled tokens, we also pre-allocate empty blocks to reduce allocation overheads.

With prefix caching, when we attempt to allocate a full block, we will compute its block hash and query the cached block map. There are 3 possible outcomes:
1. **Cache miss:** Allocate a new block from free block queue: The new allocated block may be evicted from the cache.
2. **Cache hit and the block is in free block queue:** Reuse the block and mark it to be removed from the queue.
3. **Cache hit and the block is not in free block queue (being used by other requests as well):** Reuse the block.

Note: When cache miss and we allocate a new block, the token IDs will be added to the allocated block to construct its hash. The block will also be added to the cache if it is full.

### Append Slots

When a request is scheduled again, `append_slots()` is used to maybe allocate more blocks. This can be the case of continuous chunked prefill or decode. Here are the steps in the append slots:
1. Check the allocated slots (empty slots in a partial block and preallocated blocks), and add token IDs to these slots.
2. If the allocated blocks are full, add them to the cache.
3. If the allocated slots are insufficient, allocate new blocks.

### Free

When a request is done, all its blocks will decrease the reference count by 1. If a block now has 0 reference, it will be freed (push to the free block queue). Note that since we allocate new blocks by popping the free block queue, the block order in the free block queue is also the eviction order. Since we now use LRU eviction policy, the eviction order is
1. The least accessed block.
2. When a sequence of blocks has the same access time, the one with the longest hashed tokens will be evicted first, because this is the last block in a sequence and is less likely to be shared with other requests.

We maintain the above order by pushing free blocks to the queue in the **reversed** order, so that:
1. The order of free requests implies the access time. An early free block will appear at the front of the queue.
2. When pushing a sequence of blocks to the queue, the last block with more hashed tokens goes first.

### Get Computed Blocks

Before calling `allocate_slots()`, the scheduler calls `get_computed_block_ids()` to know how many blocks hits the cache. This function simply computes the hash of full blocks and queries the cache for existing block IDs. This function won't allocate any block or change the block metadata.

## Duplication

Since V1 has incremental prepare inputs, the block table is append-only. This results in potential duplications as shown below. Suppose we have 2 identical requests (same prompt with greedy sampling) arriving at different time:

TIme 1
```
req1: [0, 1, 2, 3 (partial, 14/16)]
```
Time 2
```
req1: [0, 1, 2, 3 (partial, 15/16)]
req2: [0, 1, 2, 4 (partial, 14/16)] # Partial block cannot be shared so we allocate a new block for req2
```
TIme 3
```
req1: [0, 1, 2, 3 (full)] # Block 3 is now sharable
req2: [0, 1, 2, 4 (partial, 15/16)]
```
TIme 4
```
req1: [0, 1, 2, 3 (full)]
req2: [0, 1, 2, 4 (full)]
```

At time 4, block becomes full and has the same hash and content as block 3. In vLLM V0 block manager, we will free block 4 and assign block 3 to req2 in the next step. However, we cannot do this in V1 because block table is append only. As a result, at this moment the cache will look like:

```
block_0_hash: [block0]
block_1_hash: [block1]
block_2_hash: [block2]
block_3_hash: [block3, block4]
```

- When another request hits block 3 hash, we always allocate block 3.
- Block 4 will be free once req2 is done.

We consider that this is fine with practical use cases, because:
1. Only partial blocks will potentially have duplications. This happens at the last block of a prompt, or the first N blocks of decode.
2. Only the same prompt with greedy sampling will encounter this issue, which is not a practical use case.

cc @WoosukKwon @zhuohan123 @njhill 

## Full discussion

See [9972.md](./9972.md) and [9972.json](./9972.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/9972
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/9972.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `73b3c6877f32ebcf554ec496b65500d5a39013756b36f9352e44fa9203733803`

Existing semantic notes in [9972.sem.json](./9972.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `benchmarks/benchmark_prefix_caching.py`
- `tests/v1/core/test_prefix_caching.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/scheduler.py`
- `vllm/v1/engine/llm_engine.py`

### Diff

```diff
diff --git a/benchmarks/benchmark_prefix_caching.py b/benchmarks/benchmark_prefix_caching.py
index 1aac029992db..6d33096ca1d1 100644
--- a/benchmarks/benchmark_prefix_caching.py
+++ b/benchmarks/benchmark_prefix_caching.py
@@ -118,7 +118,7 @@ def main(args):
     random.seed(args.seed)
     if args.dataset_path is not None:
         print(f"Start to sample {args.num_prompts} prompts"
-              "from {args.dataset_path}")
+              f"from {args.dataset_path}")
         filtered_datasets = sample_requests(
             dataset_path=args.dataset_path,
             num_requests=args.num_prompts,
@@ -142,13 +142,6 @@ def main(args):
                                        repeat_count=args.repeat_count,
                                        sort=args.sort)
 
-    print("------warm up------")
-    test_prefix(
-        llm=llm,
-        prompts=prompts,
-        sampling_params=sampling_params,
-    )
-
     print("------start generating------")
     test_prefix(
         llm=llm,
diff --git a/tests/v1/core/test_prefix_caching.py b/tests/v1/core/test_prefix_caching.py
new file mode 100644
index 000000000000..e5a3b62258dd
--- /dev/null
+++ b/tests/v1/core/test_prefix_caching.py
@@ -0,0 +1,219 @@
+"""Compare the with and without prefix caching."""
+from vllm.inputs import DecoderOnlyInputs
+from vllm.sampling_params import SamplingParams
+from vllm.v1.core.kv_cache_manager import KVCacheManager, Request
+from vllm.v1.core.kv_cache_utils import hash_block_tokens
+
+
+def make_request(request_id, prompt_token_ids):
+    return Request(
+        request_id=request_id,
+        inputs=DecoderOnlyInputs(prompt_token_ids=prompt_token_ids),
+        sampling_params=SamplingParams(max_tokens=17),
+        eos_token_id=100,
+        arrival_time=0,
+        lora_request=None,
+    )
+
+
+def test_prefill():
+    manager = KVCacheManager(
+        block_size=16,
+        num_gpu_blocks=10,
+        sliding_window=False,
+        enable_caching=True,
+        num_preallocate_tokens=16,
+    )
+
+    # Complete 3 blocks (48 tokens)
+    common_token_ids = [i for i in range(3) for _ in range(16)]
+
+    # Fully cache miss
+    # Incomplete 1 block (7 tokens)
+    unique_token_ids = [3] * 7
+    req0 = make_request("0", common_token_ids + unique_token_ids)
+    computed_blocks = manager.get_computed_blocks(req0)
+    assert not computed_blocks
+    blocks = manager.allocate_slots(req0, 55, computed_blocks)
+    assert [b.block_id for b in blocks] == [0, 1, 2, 3, 4]
+
+    # Check full block metadata
+    parent_block_hash = None
+    for block_id in (0, 1, 2):
+        block_hash = hash_block_tokens(parent_block_hash,
+                                       manager.block_pool[block_id].token_ids)
+        assert manager.block_pool[block_id].block_hash == block_hash
+        assert manager.block_pool[block_id].ref_cnt == 1
+        assert manager.block_pool[block_id].num_hashed_tokens == 16 * (
+            block_id + 1)
+        assert manager.block_pool[block_id].token_ids == tuple([block_id] * 16)
+        parent_block_hash = block_hash
+
+    # Check partial/preallocated block metadata
+    for block_id in (3, 4):
+        assert manager.block_pool[block_id].block_hash is None
+        assert manager.block_pool[block_id].ref_cnt == 1
+        assert manager.block_pool[block_id].num_hashed_tokens == 0
+        if block_id == 3:
+            assert manager.block_pool[block_id].token_ids == [3] * 7
+        else:
+            assert not manager.block_pool[block_id].token_ids
+
+    # Cache hit in the common prefix when the original block is still in use.
+    # Incomplete 1 block (5 tokens)
+    unique_token_ids = [3] * 5
+    req1 = make_request("1", common_token_ids + unique_token_ids)
+    computed_blocks = manager.get_computed_blocks(req1)
+    assert [b.block_id for b in computed_blocks] == [0, 1, 2]
+    num_new_tokens = 53 - 3 * 16
+    blocks = manager.allocate_slots(req1, num_new_tokens, computed_blocks)
+    assert [b.block_id for b in blocks] == [5, 6]
+    for block in computed_blocks:
+        assert block.ref_cnt == 2
+
+    # At this point, we should have 3 free blocks left.
+    assert manager.free_block_queue.num_free_blocks == 3
+
+    manager.free(req0)
+    manager.free(req1)
+
+    # All blocks should be available.
+    assert manager.free_block_queue.num_free_blocks == 10
+    # The order should be
+    # [unallocated (7, 8)]
+    # [unique_req0 (4, 3)]
+    # [unique_req1 (6, 5)]
+    # [common (2, 1, 0)]
+    assert [
+        b.block_id for b in manager.free_block_queue.get_all_free_blocks()
+    ] == [7, 8, 9, 4, 3, 6, 5, 2, 1, 0]
+
+    # Cache hit in the common prefix when the original block is already free.
+    # Incomplete 1 block (6 tokens)
+    unique_token_ids = [3] * 6
+    req2 = make_request("2", common_token_ids + unique_token_ids)
+    computed_block = manager.get_computed_blocks(req2)
+    assert [b.block_id for b in computed_block] == [0, 1, 2]
+    num_new_tokens = 53 - 3 * 16
+    blocks = manager.allocate_slots(req2, num_new_tokens, computed_blocks)
+    assert [b.block_id for b in blocks] == [7, 8]
+
+    # Although we only have 5 free blocks, we have 8 blocks in
+    # the free block queue due to lazy removal.
+    assert manager.free_block_queue.num_free_blocks == 5
+    assert all([
+        b.ref_cnt == 0 for b in manager.free_block_queue.get_all_free_blocks()
+    ])
+    assert len([b
+                for b in manager.free_block_queue.get_all_free_blocks()]) == 5
+
+    manager.free(req2)
+
+    # Cache miss and eviction.
+    req3 = make_request("3", [99] * (16 * 9))
+    computed_blocks = manager.get_computed_blocks(req3)
+    assert not computed_blocks
+    blocks = manager.allocate_slots(req2, 16 * 9, computed_blocks)
+    # This block ID order also checks the eviction order.
+    assert [b.block_id for b in blocks] == [9, 4, 3, 6, 5, 8, 7, 2, 1, 0]
+    assert manager.free_block_queue.num_free_blocks == 0
+    assert manager.free_block_queue.free_list_head is None
+    assert manager.free_block_queue.free_list_tail is None
+
+
+def test_decode():
+    manager = KVCacheManager(
+        block_size=16,
+        num_gpu_blocks=10,
+        sliding_window=False,
+        enable_caching=True,
+        num_preallocate_tokens=16,
+    )
+
+    # Complete 3 blocks (48 tokens)
+    common_token_ids = [i for i in range(3) for _ in range(16)]
+
+    # Fully cache miss
+    # Incomplete 1 block (7 tokens)
+    unique_token_ids = [3] * 7
+    req0 = make_request("0", common_token_ids + unique_token_ids)
+    computed_blocks = manager.get_computed_blocks(req0)
+    assert not computed_blocks
+    blocks = manager.allocate_slots(req0, 55, computed_blocks)
+    assert [b.block_id for b in blocks] == [0, 1, 2, 3, 4]
+
+    # Append slots without allocating a new block.
+    req0.num_computed_tokens = 55
+    for _ in range(4):
+        req0.append_output_token_ids(8)
+    new_blocks = manager.append_slots(req0, 4)
+    assert new_blocks is not None and len(new_blocks) == 0
+    assert len(manager.block_pool[3].token_ids) == 11
+
+    # Append slots without allocating a new block, but start using the
+    # preallocated block.
+    req0.num_computed_tokens = 59
+    # 6 tokens to fill the previous block, and 10 tokens to fill
+    # the preallocated block.
+    for _ in range(5 + 10):
+        req0.append_output_token_ids(7)
+    new_blocks = manager.append_slots(req0, 15)
+    assert new_blocks is not None and len(new_blocks) == 0
+    assert len(manager.block_pool[3].token_ids) == 16
+    assert len(manager.block_pool[4].token_ids) == 10
+
+    # Append slots with allocating a new block.
+    req0.num_computed_tokens = 74
+    # 6 tokens to fill the previous block, and 10 tokens to fill
+    # the preallocated block.
+    for _ in range(6 + 11):
+        req0.append_output_token_ids(12)
+    new_blocks = manager.append_slots(req0, 17)
+    # Plus one preallocated block.
+    assert new_blocks is not None and len(new_blocks) == 2
+    assert len(manager.block_pool[4].token_ids) == 16
+    assert len(manager.block_pool[5].token_ids) == 11
+    assert len(manager.block_pool[6].token_ids) == 0
+
+
+def test_evict():
+    manager = KVCacheManager(
+        block_size=16,
+        num_gpu_blocks=10,
+        sliding_window=False,
+        enable_caching=True,
+        num_preallocate_tokens=16,
+    )
+
+    last_token_id = 5 * 16 + 7
+    req0 = make_request("0", list(range(last_token_id)))
+    computed_blocks = manager.get_computed_blocks(req0)
+    assert not computed_blocks
+    blocks = manager.allocate_slots(req0, 5 * 16 + 7, computed_blocks)
+    assert len(blocks) == 7  # 5 full + 1 partial + 1 preallocated
+
+    # 3 blocks.
+    req1 = make_request("1", list(range(last_token_id,
+                                        last_token_id + 3 * 16)))
+    computed_blocks = manager.get_computed_blocks(req1)
+    assert not computed_blocks
+    blocks = manager.allocate_slots(req1, 3 * 16, computed_blocks)
+    assert len(blocks) == 3  # 3 full blocks
+    last_token_id += 3 * 16
+
+    assert manager.free_block_queue.num_free_blocks == 0
+
+    manager.free(req0)
+    manager.free(req1)
+    assert manager.free_block_queue.num_free_blocks == 10
+    assert [
+        b.block_id for b in manager.free_block_queue.get_all_free_blocks()
+    ] == [6, 5, 4, 3, 2, 1, 0, 9, 8, 7]
+
+    # Touch the first 2 blocks.
+    req2 = make_request("2", list(range(2 * 16 + 3)))
+    computed_blocks = manager.get_computed_blocks(req2)
+    assert [b.block_id for b in computed_blocks] == [0, 1]
+    blocks = manager.allocate_slots(req2, 3, computed_blocks)
+    assert [b.block_id for b in blocks] == [6, 5]
+    assert manager.free_block_queue.num_free_blocks == 6
diff --git a/vllm/v1/core/kv_cache_manager.py b/vllm/v1/core/kv_cache_manager.py
index 9b735a8be10d..82094fb65dd1 100644
--- a/vllm/v1/core/kv_cache_manager.py
+++ b/vllm/v1/core/kv_cache_manager.py
@@ -1,9 +1,11 @@
+from collections import defaultdict
 from typing import Dict, List, Optional
 
-import numpy as np
-
 from vllm.logger import init_logger
 from vllm.utils import cdiv
+from vllm.v1.core.kv_cache_utils import (BlockHashType, FreeKVCacheBlockQueue,
+                                         KVCacheBlock, hash_block_tokens,
+                                         hash_request_tokens)
 from vllm.v1.request import Request
 
 logger = init_logger(__name__)
@@ -36,73 +38,359 @@ def __init__(
         self.num_preallocate_tokens = num_preallocate_tokens
         self.num_preallocate_blocks = cdiv(num_preallocate_tokens, block_size)
 
-        self.free_block_ids = list(range(num_gpu_blocks))
-        self.req_to_block_ids: Dict[str, List[int]] = {}
-        self.ref_cnts = np.zeros(num_gpu_blocks, dtype=np.int32)
+        # A Block pool of all kv-cache blocks.
+        self.block_pool: List[KVCacheBlock] = [
+            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
+        ]
+        # Free block queue that constructs and manipulates a doubly linked
+        # list of free blocks (including eviction candidates when caching is
+        # enabled).
+        self.free_block_queue = FreeKVCacheBlockQueue(self.block_pool)
+
+        # {block_hash: {block ID: block}}. A cached block is
+        # a full block with a block hash that can be used for prefix caching.
+        # The cached block may be used by running requests or in the
+        # free_block_queue that could potentially be evicted.
+        # NOTE: We currently don't de-duplicate the blocks in the cache,
+        # meaning that if a block becomes full and is cached, we don't check
+        # if there is already an identical block in the cache. This is because
+        # we want to make sure the allocated block IDs won't change so that
+        # block tables are append-only.
+        self.cached_block_hash_to_block: Dict[BlockHashType, Dict[
+            int, KVCacheBlock]] = defaultdict(dict)
+
+        # Mapping from request ID to blocks to track the blocks allocated
+        # for each request, so that we can free the blocks when the request
+        # is finished.
+        self.req_to_blocks: Dict[str, List[KVCacheBlock]] = {}
 
-    def get_computed_blocks(self, request: Request) -> List[int]:
+    def get_computed_blocks(self, request: Request) -> List[KVCacheBlock]:
+        """Get the computed (cached) blocks for the request.
+        Note that the computed blocks must be full.
+
+        Args:
+            request: The request to get the computed blocks.
+        
+        Returns:
+            A list of blocks that are computed for the request.
+        """
         if not self.enable_caching:
-            # No prefix caching.
+            # Prefix caching is disabled.
             return []
-        # TODO(woosuk): Implement hash-based caching.
-        return []
+
+        computed_blocks = []
+        block_hashes = hash_request_tokens(self.block_size,
+                                           request.all_token_ids)
+
+        for block_hash in block_hashes:
+            # block_hashes is a chain of block hashes. If a block hash is not
+            # in the cached_block_hash_to_id, the following block hashes are
+            # not computed yet for sure.
+            if cached_block := self._get_cached_block(block_hash):
+                computed_blocks.append(cached_block)
+            else:
+                break
+
+        return computed_blocks
 
     def append_slots(
         self,
         request: Request,
         num_tokens: int,
-    ) -> Optional[List[int]]:
+    ) -> Optional[List[KVCacheBlock]]:
+        """Append slots to the block table of the request.
+        We first append slots to already allocated blocks. If the allocated
+        blocks are not enough, we allocate new blocks.
+
+        Args:
+            request: The request to append slots.
+            num_tokens: The number of tokens to append.
+        
+        Returns:
+            A list of new blocks if new blocks are allocated, or None
+            if new blocks are required but cannot be allocated.
+        """
         num_required_blocks = cdiv(request.num_computed_tokens + num_tokens,
                                    self.block_size)
-        req_block_ids = self.req_to_block_ids[request.request_id]
-        if num_required_blocks <= len(req_block_ids):
-            # No new block is needed.
-            return []
+        req_blocks = self.req_to_blocks[request.request_id]
 
-        num_new_blocks = num_required_blocks - len(req_block_ids)
-        num_free_blocks = len(self.free_block_ids)
-        if num_new_blocks > num_free_blocks:
-            # Cannot allocate new blocks.
+        num_new_blocks = num_required_blocks - len(req_blocks)
+        if num_new_blocks > self.free_block_queue.num_free_blocks:
+            # Need to allocate new blocks due to insufficient pre-allocated
+            # slots, but we cannot allocate new blocks due to the limit.
             return None
 
-        # Allocate new blocks.
+        # When caching is enabled, assign token IDs to already allocated blocks.
+        new_token_ids = None
+        parent_block = None
+        if self.enable_caching:
+            # Figure out the token IDs to add to the blocks.
+            new_token_ids = request.all_token_ids[
+                request.num_computed_tokens:request.num_computed_tokens +
+                num_tokens]
+
+            # Find the last full block index.
+            # TODO: This may be optimized by calculating the computed tokens.
+            last_full_block_idx = len(req_blocks) - 1
+            while (last_full_block_idx >= 0
+                   and req_blocks[last_full_block_idx].block_hash is None):
+                last_full_block_idx -= 1
+
+            parent_block = (req_blocks[last_full_block_idx]
+                            if last_full_block_idx >= 0 else None)
+            token_id_idx = self._add_token_ids_to_blocks(
+                blocks=req_blocks[last_full_block_idx + 1:],
+                token_ids=new_token_ids,
+                parent_block=parent_block)
+
+            new_token_ids = new_token_ids[token_id_idx:]
+            parent_block = req_blocks[-1]
+
+        # No new block is needed. When caching is enabled, we make sure
+        # token_id_idx is equal to len(new_token_ids), meaning that all tokens
+        # are added to allocated blocks.
+        if num_required_blocks <= len(req_blocks):
+            assert not self.enable_caching or token_id_idx == num_tokens, \
+                    f"{token_id_idx=} != {num_tokens=}"
+            return []
+
+        # Allocate new blocks considering preallocated blocks, and
+        # add token IDs to them if caching is enabled.
         num_new_blocks = min(num_new_blocks + self.num_preallocate_blocks,
-                             num_free_blocks)
-        new_block_ids = self._get_new_blocks(num_new_blocks)
-        req_block_ids.extend(new_block_ids)
-        self.ref_cnts[new_block_ids] += 1
-        return new_block_ids
+                             self.free_block_queue.num_free_blocks)
+        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
+                                          parent_block)
+        req_blocks.extend(new_blocks)
+        return new_blocks
 
     def allocate_slots(
         self,
         request: Request,
         num_tokens: int,
-        computed_block_ids: List[int],
-    ) -> Optional[List[int]]:
+        computed_blocks: List[KVCacheBlock],
+    ) -> Optional[List[KVCacheBlock]]:
+        """Allocate slots for a new request.
+
+        Args:
+            request: The request to allocate slots.
+            num_tokens: The number of tokens to allocate. Note that this does
+                not include the tokens that have already been computed.
+            computed_blocks: The blocks that have already been computed.
+        
+        Returns:
+            A list of new allocated blocks.
+        """
+        if num_tokens == 0:
+            raise ValueError(
+                f"num_tokens must be greater than 0, got {num_tokens}")
+
+        # If a computed block of a request is an eviction candidate (in the
+        # free queue and ref_cnt == 0), it cannot be counted as a free block
+        # when allocating this request.
+        num_evictable_computed_blocks = len(
+            [blk for blk in computed_blocks if blk.ref_cnt == 0])
+
         num_required_blocks = cdiv(num_tokens, self.block_size)
-        num_free_blocks = len(self.free_block_ids)
-        if num_required_blocks > num_free_blocks:
+        if (num_required_blocks > self.free_block_queue.num_free_blocks -
+                num_evictable_computed_blocks):
             # Cannot allocate new blocks.
             return None
 
-        num_new_blocks = min(num_required_blocks + self.num_preallocate_blocks,
-                             num_free_blocks)
-        new_block_ids = self._get_new_blocks(num_new_blocks)
-        block_ids = computed_block_ids + new_block_ids
-        self.req_to_block_ids[request.request_id] = block_ids
-        self.ref_cnts[block_ids] += 1
-        return new_block_ids
+        # Determine the number of new blocks to allocate considering
+        # preallocated blocks.
+        num_new_blocks = min(
+            num_required_blocks + self.num_preallocate_blocks,
+            self.free_block_queue.num_free_blocks -
+            num_evictable_computed_blocks)
+
+        num_computed_tokens = len(computed_blocks) * self.block_size
+
+        # When caching is enabled, get the new token IDs and the parent block
+        # ID to generate cache keys.
+        new_token_ids = None
+        parent_block = None
+        if self.enable_caching:
+            # Touch the computed blocks to make sure they won't be evicted.
+            self._touch(computed_blocks)
+
+            # Get the token IDs for the blocks being allocated for hashing.
+            new_token_ids = request.all_token_ids[
+                num_computed_tokens:num_computed_tokens + num_tokens]
+            if not new_token_ids:
+                raise RuntimeError(
+                    "Failed to infer the token IDs for allocation. "
+                    f"#all_tokens={len(request.all_token_ids)} < "
+                    f"#computed_tokens={num_computed_tokens}")
+
+            # Get the parent block ID to construct the block chain.
+            parent_block = computed_blocks[-1] if computed_blocks else None
+
+        new_blocks = self._get_new_blocks(num_new_blocks, new_token_ids,
+                                          parent_block)
+
+        # Concatenate the computed block IDs and the new block IDs.
+        self.req_to_blocks[request.request_id] = computed_blocks + new_blocks
+        return new_blocks
 
     def free(self, request: Request) -> None:
-        block_ids = self.req_to_block_ids.pop(request.request_id)
-        self.ref_cnts[block_ids] -= 1
-        for block_id in block_ids:
-            ref_cnt = self.ref_cnts[block_id]
-            if ref_cnt == 0:
-                self.free_block_ids.append(block_id)
-
-    def _get_new_blocks(self, num_blocks: int) -> List[int]:
-        assert num_blocks <= len(self.free_block_ids)
-        new_block_ids = self.free_block_ids[-num_blocks:]
-        self.free_block_ids = self.free_block_ids[:-num_blocks]
-        return new_block_ids
+        """Free the blocks allocated for the request.
+        When caching is enabled, we free the blocks in reverse order so that
+        the tail blocks are evicted first.
+
+        Args:
+            request: The request to free the blocks.
+        """
+        blocks = self.req_to_blocks.pop(request.request_id)
+        if self.enable_caching:
+            # Free blocks in reverse order so that the tail blocks are
+            # freed first.
+            blocks = reversed(blocks)
+
+        for block in blocks:
+            block.ref_cnt -= 1
+            if block.ref_cnt == 0:
+                self.free_block_queue.append(block)
+
+    def _get_new_blocks(
+            self,
+            num_blocks: int,
+            token_ids: Optional[List[int]] = None,
+            parent_block: Optional[int] = None) -> List[KVCacheBlock]:
+        """Get new blocks from the free block pool, and add token IDs to
+        allocated blocks if caching is enabled.
+        Note that we do not check block cache in this function.
+        
+        Args:
+            num_blocks: The number of blocks to allocate.
+            token_ids: The token IDs in the blocks. None if caching is disabled.
+            parent_block: The parent block. Used to include block chain
+                in the block hash.
+        
+        Returns:
+            A list of new block.
+        """
+        if num_blocks > self.free_block_queue.num_free_blocks:
+            raise ValueError(
+                f"Cannot get {num_blocks} free blocks from the pool")
+
+        # First allocate blocks.
+        ret: List[KVCacheBlock] = []
+        idx = 0
+        while idx < num_blocks:
+            curr_block = self.free_block_queue.popleft()
+            assert curr_block.ref_cnt == 0
+
+            # Evict blocks from the cache.
+            if self.enable_caching:
+                block_hash = curr_block.block_hash
+                if (block_hash is not None
+                        and block_hash in self.cached_block_hash_to_block):
+                    if len(self.cached_block_hash_to_block[block_hash]) == 1:
+                        del self.cached_block_hash_to_block[block_hash]
+                    else:
+                        del self.cached_block_hash_to_block[block_hash][
+                            curr_block.block_id]
+                curr_block.reset()
+
+            curr_block.ref_cnt = 1
+            ret.append(curr_block)
+            idx += 1
+
+        # Then assign token IDs to the allocated blocks.
+        if self.enable_caching:
+            assert token_ids is not None
+            token_id_idx = self._add_token_ids_to_blocks(
+                blocks=ret, token_ids=token_ids, parent_block=parent_block)
+            assert token_id_idx == len(token_ids)
+
+        return ret
+
+    def _cache_full_block(self,
+                          block: KVCacheBlock,
+                          parent_block: Optional[KVCacheBlock] = None) -> None:
+        """Cache a full block for prefix caching.
+
+        Args:
+            block: The block to cache.
+            parent_block: The parent block. None if this is the first block.
+        """
+        parent_block_hash = (parent_block.block_hash
+                             if parent_block is not None else None)
+        assert len(block.token_ids) == self.block_size
+        block.token_ids = tuple(block.token_ids)
+        block_hash = hash_block_tokens(parent_block_hash, block.token_ids)
+        block.block_hash = block_hash
+        block.num_hashed_tokens = self.block_size + (
+            parent_block.num_hashed_tokens if parent_block is not None else 0)
+        self.cached_block_hash_to_block[block_hash][block.block_id] = block
+
+    def _get_cached_block(self,
+                          block_hash: BlockHashType) -> Optional[KVCacheBlock]:
+        """Get a cached block by the block hash, or None if cache miss.
+        If there are duplicated blocks, we return the first block in the cache.
+
+        Args:
+            block_hash: The hash value of the block.
+
+        Returns:
+            The cached block if it exists, or None.
+        """
+        if block_hash in self.cached_block_hash_to_block:
+            first_block_id = list(
+                self.cached_block_hash_to_block[block_hash].keys())[0]
+            return self.cached_block_hash_to_block[block_hash][first_block_id]
+        return None
+
+    def _touch(self, blocks: List[KVCacheBlock]) -> None:
+        """Touch a block increases its reference count by 1, and may remove
+        the block from the free queue. This is used when a block is hit by
+        another request with the same prefix.
+
+        Args:
+            blocks: A list of blocks to touch.
+        """
+        for block in blocks:
+            # ref_cnt=0 means this block is in the free list (i.e. eviction
+            # candidate), so remove it.
+            if block.ref_cnt == 0:
+                self.free_block_queue.remove(block)
+            block.ref_cnt += 1
+
+    def _add_token_ids_to_blocks(
+            self,
+            blocks: List[KVCacheBlock],
+            token_ids: List[int],
+            parent_block: Optional[KVCacheBlock] = None) -> int:
+        """Add token IDs to a list of allocated blocks.
+        If a block becomes full after adding token IDs, cache it.
+        Return the token ID index that has not been added to the blocks
+        if the blocks are not enough to hold all the token IDs.
+
+        Args:
+            blocks: A list of blocks to add token IDs.
+            token_ids: A list of token IDs to add.
+            parent_block: The parent block. None if this is the
+                first block.
+
+        Returns:
+            The starting token ID index that has not been added to the blocks
+            due to insufficient given blocks.
+        """
+        token_id_start = 0
+        for curr_block in blocks:
+            # If all token IDs are added, then the rest of the blocks are
+            # preallocated blocks, so we only need to update the
+            # parent_block_id. FIXME
+            if token_id_start == len(token_ids):
+                continue
+
+            # Add token IDs to the empty slots in the block.
+            empty_slots = self.block_size - len(curr_block.token_ids)
+            token_id_end = min(token_id_start + empty_slots, len(token_ids))
+            curr_block.token_ids.extend(token_ids[token_id_start:token_id_end])
+            # Cache the block if it becomes full.
+            if len(curr_block.token_ids) == self.block_size:
+                self._cache_full_block(curr_block, parent_block)
+            parent_block = curr_block
+            token_id_start = token_id_end
+        return token_id_start
diff --git a/vllm/v1/core/kv_cache_utils.py b/vllm/v1/core/kv_cache_utils.py
new file mode 100644
index 000000000000..33dbfb7377bf
--- /dev/null
+++ b/vllm/v1/core/kv_cache_utils.py
@@ -0,0 +1,194 @@
+"""KV-Cache Utilities."""
+from dataclasses import dataclass, field
+from typing import List, Optional, Tuple, Union
+
+from vllm.logger import init_logger
+
+logger = init_logger(__name__)
+
+BlockHashType = Tuple[int, Tuple[int]]
+
+
+@dataclass
+class KVCacheBlock:
+    """KV-cache block metadata."""
+    # Block ID, ranging from 0 to num_gpu_blocks - 1.
+    block_id: int
+    # Reference count.
+    ref_cnt: int = 0
+    # Token IDs in the block. When the block is full, the type of token_ids
+    # should be Tuple[int] for fast matching.
+    token_ids: Union[List[int], Tuple[int]] = field(default_factory=list)
+    # The hash of the block composed of (block hash, tuple of token IDs).
+    # It is only available when the block is full.
+    block_hash: Optional[BlockHashType] = None
+    # The number of hashed tokens. More hashed tokens means the block
+    # is closer to the end of a prompt and more likely to be evicted.
+    num_hashed_tokens: int = 0
+
+    # Used to construct a doubly linked list for free blocks.
+    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
+    prev_free_block: Optional["KVCacheBlock"] = None
+    next_free_block: Optional["KVCacheBlock"] = None
+
+    def reset(self):
+        """Reset the block metadata."""
+        self.ref_cnt = 0
+        self.token_ids = []
+        self.block_hash = None
+        self.num_hashed_tokens = 0
+
+
+class FreeKVCacheBlockQueue:
+    """This class organizes a list of KVCacheBlock objects to a doubly linked
+    list of free blocks. We implement this class instead of using Python
+    builtin deque to support removing a block in the middle of the queue
+    in O(1) time. To close the performance gap to the builtin deque which is
+    implemented in C++, this class does not allocate any Python objects when
+    manipulating the linked list. Instead, this class manipulates the 
+    prev_free_block and next_free_block attributes of the given blocks.
+
+    The queue is ordered by block ID in the beginning. When a block is allocated
+    and then freed, it will be appended back with the eviction order:
+    1. The least recent used block is at the front (LRU).
+    2. If two blocks have the same last accessed time (allocated by the
+       same sequence), the one with more hash tokens (the tail of a block
+       chain) is at the front.
+    Note that we maintain this order by reversing the block order when free
+    blocks of a request. This operation is outside of this class.
+
+    Args:
+        blocks: A list of KVCacheBlock objects.
+    """
+
+    def __init__(self, blocks: List[KVCacheBlock]) -> None:
+        self.num_free_blocks = len(blocks)
+
+        # Initialize the doubly linked list of free blocks.
+        self.free_list_head = blocks[0]
+        self.free_list_tail = blocks[-1]
+        for i in range(self.num_free_blocks):
+            if i > 0:
+                blocks[i].prev_free_block = blocks[i - 1]
+            if i < self.num_free_blocks - 1:
+                blocks[i].next_free_block = blocks[i + 1]
+
+    def popleft(self) -> KVCacheBlock:
+        """Pop the first free block and reduce num_free_blocks by 1.
+        
+        Returns:
+            The first free block.
+        """
+        if not self.free_list_head:
+            raise ValueError("No free blocks available")
+
+        block = self.free_list_head
+        self.remove(block)
+        return block
+
+    def remove(self, block: KVCacheBlock) -> None:
+        """Remove a block in the free list and reduce num_free_blocks by 1.
+        
+        Args:
+            block: The block to remove.
+        """
+        if block.prev_free_block is not None:
+            # Link the previous block to the next block.
+            block.prev_free_block.next_free_block = block.next_free_block
+        if block.next_free_block is not None:
+            # Link the next block to the previous block.
+            block.next_free_block.prev_free_block = block.prev_free_block
+
+        if block == self.free_list_head:
+            # Update the head if the block is the head.
+            self.free_list_head = block.next_free_block
+        if block == self.free_list_tail:
+            # Update the tail if the block is the tail.
+            self.free_list_tail = block.prev_free_block
+
+        # Remove the block from the linked list.
+        block.prev_free_block = block.next_free_block = None
+        self.num_free_blocks -= 1
+
+    def append(self, block: KVCacheBlock) -> None:
+        """Put a block back into the free list and increase
+        num_free_blocks by 1.
+
+        Args:
+            block: The block to append.
+        """
+        if self.free_list_tail is not None:
+            # Link the last block to the new block.
+            self.free_list_tail.next_free_block = block
+            block.prev_free_block = self.free_list_tail
+            self.free_list_tail = block
+        else:
+            # The free list is empty.
+            assert self.free_list_head is None
+            self.free_list_head = self.free_list_tail = block
+
+        block.next_free_block = None
+        self.num_free_blocks += 1
+
+    def get_all_free_blocks(self) -> List[KVCacheBlock]:
+        """Get all free blocks in the free list. Mainly used for testing.
+        
+        Returns:
+            A list of free blocks.
+        """
+        ret = []
+        curr_block = self.free_list_head
+        while curr_block is not None:
+            ret.append(curr_block)
+            curr_block = curr_block.next_free_block
+        return ret
+
+
+def hash_block_tokens(parent_block_hash: Optional[int],
+                      curr_block_token_ids: Tuple[int]) -> BlockHashType:
+    """Computes a hash value corresponding to the contents of a block and
+    the contents of the preceding block(s). The hash value is used for
+    prefix caching. We use LRU cache for this function to avoid recomputing
+    hash values for the same block contents.
+
+    TODO: Support arbitrary metadata so that we could support more
+    features such as LoRA adapter.
+
+    Args:
+        parent_block_hash: The hash of the parent block. None
+            if this is the first block.
+        curr_block_token_ids: A tuple of token ids in the current
+            block. The current block is assumed to be full.
+
+    Returns:
+        The hash value of the block and the token ids in the block.
+        The entire tuple is used as the hash key of the block.
+    """
+    return (hash(
+        (parent_block_hash, *curr_block_token_ids)), curr_block_token_ids)
+
+
+def hash_request_tokens(block_size: int,
+                        token_ids: List[int]) -> List[BlockHashType]:
+    """Computes hash values of a chain of blocks given a sequence of
+    token IDs. The hash value is used for prefix caching.
+
+    Args:
+        block_size: The size of each block.
+        token_ids: A sequence of token ids in the request.
+
+    Returns:
+        The list of computed hash values.
+    """
+    ret = []
+    parent_block_hash = None
+    for start in range(0, len(token_ids), block_size):
+        end = start + block_size
+        block_token_ids = tuple(token_ids[start:end])
+        # Do not hash the block if it is not full.
+        if len(block_token_ids) < block_size:
+            break
+        block_hash = hash_block_tokens(parent_block_hash, block_token_ids)
+        ret.append(block_hash)
+        parent_block_hash = block_hash
+    return ret
diff --git a/vllm/v1/core/scheduler.py b/vllm/v1/core/scheduler.py
index 601790564217..a60f8b8138ec 100644
--- a/vllm/v1/core/scheduler.py
+++ b/vllm/v1/core/scheduler.py
@@ -34,7 +34,7 @@ def __init__(
             block_size=self.cache_config.block_size,
             num_gpu_blocks=num_gpu_blocks,
             sliding_window=self.cache_config.sliding_window,
-            enable_caching=True)
+            enable_caching=self.cache_config.enable_prefix_caching)
         self.block_size = self.cache_config.block_size
 
         # Scheduling constraints.
@@ -91,9 +91,9 @@ def schedule(self) -> "SchedulerOutput":
             assert num_new_tokens > 0
 
             while True:
-                new_block_ids = self.kv_cache_manager.append_slots(
+                new_blocks = self.kv_cache_manager.append_slots(
                     request, num_new_tokens)
-                if new_block_ids is None:
+                if new_blocks is None:
                     # The request cannot be scheduled.
                     # Preempt the lowest-priority request.
                     preempted_req = self.running.pop()
@@ -110,7 +110,9 @@ def schedule(self) -> "SchedulerOutput":
                     # The request can be scheduled.
                     scheduled_running_reqs.append(request)
 
-                    req_to_new_block_ids[request.request_id] = new_block_ids
+                    req_to_new_block_ids[request.request_id] = [
+                        b.block_id for b in new_blocks
+                    ]
                     num_scheduled_tokens[request.request_id] = num_new_tokens
                     token_budget -= num_new_tokens
                     req_index += 1
@@ -126,22 +128,29 @@ def schedule(self) -> "SchedulerOutput":
 
                 request = self.waiting[0]
                 # Get already-cached tokens.
-                computed_block_ids = self.kv_cache_manager.get_computed_blocks(
+                computed_blocks = self.kv_cache_manager.get_computed_blocks(
                     request)
                 # NOTE(woosuk): Since incomplete blocks are not eligible for
                 # sharing, `num_computed_tokens` is always a multiple of
                 # `block_size`.
-                num_computed_tokens = len(computed_block_ids) * self.block_size
+                num_computed_tokens = len(computed_blocks) * self.block_size
                 # Number of tokens to be scheduled.
                 # We use `request.num_tokens` instead of
                 # `request.num_prompt_tokens` to consider the resumed requests,
                 # which have output tokens.
                 num_new_tokens = request.num_tokens - num_computed_tokens
+                if num_new_tokens == 0:
+                    # The happens when prompt length is divisible by the block
+                    # size and all blocks are cached. Now we force to recompute
+                    # the last token.
+                    num_computed_tokens -= 1
+                    num_new_tokens = 1
+                    computed_blocks.pop()
                 num_new_tokens = min(num_new_tokens, token_budget)
                 assert num_new_tokens > 0
-                new_block_ids = self.kv_cache_manager.allocate_slots(
-                    request, num_new_tokens, computed_block_ids)
-                if new_block_ids is None:
+                new_blocks = self.kv_cache_manager.allocate_slots(
+                    request, num_new_tokens, computed_blocks)
+                if new_blocks is None:
                     # The request cannot be scheduled.
                     break
                 request.num_computed_tokens = num_computed_tokens
@@ -156,8 +165,9 @@ def schedule(self) -> "SchedulerOutput":
                     raise RuntimeError(
                         f"Invalid request status: {request.status}")
 
-                req_to_new_block_ids[request.request_id] = (
-                    computed_block_ids + new_block_ids)
+                req_to_new_block_ids[request.request_id] = [
+                    b.block_id for b in computed_blocks + new_blocks
+                ]
                 num_scheduled_tokens[request.request_id] = num_new_tokens
                 token_budget -= num_new_tokens
                 request.status = RequestStatus.RUNNING
diff --git a/vllm/v1/engine/llm_engine.py b/vllm/v1/engine/llm_engine.py
index b538c2c7d63b..cd3f5c75d0d1 100644
--- a/vllm/v1/engine/llm_engine.py
+++ b/vllm/v1/engine/llm_engine.py
@@ -65,6 +65,7 @@ def __init__(
         elif usage_context == UsageContext.OPENAI_API_SERVER:
             scheduler_config.max_num_seqs = 1024
             scheduler_config.max_num_batched_tokens = 2048
+        cache_config.enable_prefix_caching = True
 
         logger.info(
             "Initializing an LLM engine (v%s) with config: "
```
