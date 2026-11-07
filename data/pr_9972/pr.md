# PR #9972: [V1] Prefix caching (take 2)

- field: scheduling  | area: v1/core
- author: comaniac
- state: MERGED  created: 2024-11-04T00:06:48Z  merged: 2024-11-08T01:34:44Z
- url: https://github.com/vllm-project/vllm/pull/9972
- labels: ready

## Body

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

## Comments

### github-actions @ 2024-11-04T00:07:00Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### WoosukKwon @ 2024-11-05T19:04:48Z

@comaniac Thanks for the great work! Is this PR ready for review?
---
### comaniac @ 2024-11-05T19:23:26Z

> @comaniac Thanks for the great work! Is this PR ready for review?

Yes I don't have more things to add. Please go ahead and review
---
### zhuohan123 @ 2024-11-06T19:14:28Z

Will review the PR tonight
---
### comaniac @ 2024-11-07T07:07:25Z

> A high level question: Right now for a block, we have 3 ways to index it: block hash, block id, and the python `KVCacheBlock` object itself. Do we have to have all 3? I assume we cannot just use block hash since not all blocks have hash. Can we remove block id and only use `KVCacheBlock` object for index? Is the reason we keep `block_id` is that we need to pass `block_id`s to the worker?

So the question is about whether we could remove "block_id" attribute from the data class? I'm not sure but will take a look.

---
### comaniac @ 2024-11-07T19:13:04Z

@zhuohan123 I've indexed blocks using the object itself instead of block IDs. This does simplify code in many places so thanks for the suggestion. Meanwhile I still need to keep the block ID in the object in order to let scheduler build block tables.

For prefix caching default on, I guess it might be better to enable it by default a bit later when VLM is in. If no other objections I plan to merge this PR by today.
---
### njhill @ 2024-11-07T22:53:03Z

> For prefix caching default on, I guess it might be better to enable it by default a bit later when VLM is in. If no other objections I plan to merge this PR by today.

WDYT about enabling by default for non-VLMs? might be nice to have it exercised since we want it to be default soon anyhow.
---
### comaniac @ 2024-11-08T00:08:46Z

> > For prefix caching default on, I guess it might be better to enable it by default a bit later when VLM is in. If no other objections I plan to merge this PR by today.
> 
> WDYT about enabling by default for non-VLMs? might be nice to have it exercised since we want it to be default soon anyhow.

I'm ok with this proposal. WDYT @WoosukKwon 
---
### comaniac @ 2024-11-08T01:04:30Z

The changes in the latest commit:
1. Refactor utilities to `kv_cache_utils.py`.
2. Enhance the block hash type from `int` to `Tuple[int, Tuple[int]]`, which is (hash_value, (toke_ids,)). This guarantees no hash conflicts. I benchmarked the tuple matching latency and it is about 0.025 ms for block size 16,32,48. It is ~3x faster than list matching.
3. Use `request.all_token_ids` and remove "prefill"/"decode" specific logic.
4. Enable prefix caching by default in v1.


---
### WoosukKwon @ 2024-11-08T01:09:20Z

@comaniac I've just merged #10135. Could you please rebase?
---