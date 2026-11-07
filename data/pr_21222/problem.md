# Task PR #21222 — [Core] Introduce popleft_n and append_n in FreeKVCacheBlockQueue to further optimize block_pool (#21222)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> FreeKVCacheBlockQueue only supported single-element popleft and append. Both get_new_blocks and free_blocks ran these one-at-a-time in a loop, so each call performed O(n) individual doubly-linked-list splices. Every single popleft re-wired the fake head to the next block and every single append re-wired the fake tail, doing roughly twice the pointer updates actually needed for a bulk operation.

### Issue — detailed

> In block_pool.get_new_blocks a while loop called free_block_queue.popleft() per block, and each popleft reconnected fake_free_list_head to the second block — redundant since that block may be popped next. Likewise free_blocks called append() per block, each re-wiring fake_free_list_tail. The linked-list splice count scaled at ~2x the necessary work for a batch.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds popleft_n(n) and append_n(blocks) to FreeKVCacheBlockQueue that walk the list once and rewire the fake head/tail a single time for the whole batch. get_new_blocks now calls popleft_n(num_blocks) then iterates the returned list once to evict (if caching) and bump ref_cnt inline, instead of the per-block popleft loop. free_blocks materializes the iterable, decrements ref_cnt in one pass, then append_n's the blocks that reached ref_cnt 0 and are non-null. popleft_n detaches popped blocks' prev/next pointers and connects the fake head to the new first block once; append_n chains the new blocks and connects the last one to the fake tail once. Net effect: linked-list operations cut roughly in half — get_new_blocks 0.15ms→0.008ms, free_blocks 33us→9us.

**Author's words:**

> Introduce bulk popleft_n and append_n operators that exploit the known total count to perform the linked-list manipulation in a single pass, rather than popping/appending one block at a time. Because single-element popleft/append each do an unnecessary fake-head/tail reconnection, batching halves the number of linked-list operations. New unit tests cover append_n and popleft_n, and benchmarks confirm get_blocks and free_blocks times are cut in half.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **get_new_blocks** (us/op): reported 150 -> 8 us/op. workload: `python3 benchmarks/kv_cache/benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000 --allocate-blocks 10 50 100 500 1000`
- **free_blocks** (us/op): reported 33 -> 9 us/op.
- note: Pure Python microbench; imports vllm.v1.core.block_pool.BlockPool + vllm.utils, no CUDA. One run prints both ops.

## Reported numbers (PR author)
- gain: 95% (0.15ms→0.008ms) (unit)
- evidence: benchmark scripts - Get new blocks improved from 0.15ms to 0.008ms - Free new blocks improved from 33us to 9us
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
