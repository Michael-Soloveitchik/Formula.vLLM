# Task PR #21222 — [Core] Introduce popleft_n and append_n in FreeKVCacheBlockQueue to further optimize block_pool (#21222)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (the PR's solution)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
A pool of reusable fixed-size memory blocks is kept in a doubly linked list. The system took blocks out one at a time and put them back one at a time. Each single-block operation fixed up neighbor links that were often immediately undone by the next operation, wasting work on a performance-critical path.

### Issue — detailed
In block_pool.py, get_new_blocks called free_block_queue.popleft() once per block in a loop, and free_blocks called free_block_queue.append() once per block. Each popleft() reconnected the fake head to the next block (even though that block might be popped on the very next iteration), and each append() reconnected the fake tail; doing this per block produces links the next iteration overwrites. Both paths are on the critical path: KV block allocation hard-blocks the forward pass and KV block free hard-blocks the end of the decode cycle. The redundant per-block linked-list pointer updates roughly double the necessary work.

## Expert solution (reference — the merged PR)
**General:** Add bulk operations that pop or append many blocks in one pass. The links between neighboring blocks are set once, and only the endpoint links to the list's head/tail are fixed once at the end instead of once per block. This roughly halves the pointer operations.

**Detailed:** Added popleft_n(n) and append_n(blocks) to FreeKVCacheBlockQueue in kv_cache_utils.py. popleft_n walks n blocks from the head, clears their prev/next pointers, then reconnects the fake head to the new first block a single time. append_n chains the incoming blocks to each other in one loop, then connects the last one to the fake tail once. block_pool.get_new_blocks now calls popleft_n(num_blocks) and bumps ref counts (with eviction when caching is enabled) in a single pass; free_blocks materializes the iterable, decrements ref counts, then calls append_n with the blocks that reached ref_cnt 0 and are not null.

**Author's words:** Most block_pool operators are on the critical path: the forward path is hard blocked by KV block allocation and the decode cycle end is hard blocked by KV block free. Popping one at a time triggered unnecessary fake-head connections on the second block, which were wasted since that block might be popped right after. Since the total number of blocks to pop is known ahead, popleft_n does bulk popleft, cutting linked-list operations to about half of n popleft; append_n does the same for bulk append when freeing. New unit tests for append_n and popleft_n are added.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **get_new_blocks** (us/op): reported 150.0 -> 8.0 us/op. workload: get_new_blocks avg
- **free_blocks** (us/op): reported 33.0 -> 9.0 us/op. workload: free_blocks avg

## Reported numbers (PR author)
- metric: block_pool get_new_blocks and free_blocks latency from benchmark scripts; benchmark_blockpool get_blocks/free_blocks times
- value: get_new_blocks 0.15 -> 0.008; free_blocks 33 -> 9
- unit: ms (get_new_blocks); us (free_blocks)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
