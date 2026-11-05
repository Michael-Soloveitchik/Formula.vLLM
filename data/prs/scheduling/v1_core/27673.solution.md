# PR #27673: [Core] Early return in SlidingWindowManager.remove_skipped_blocks

- field: scheduling  | area: v1_core
- author: Jialin
- state: MERGED  created: 2025-10-28T18:53:40Z  merged: 2025-10-29T01:32:34Z
- url: https://github.com/vllm-project/vllm/pull/27673
- labels: frontend, ready, v1

# Issue

## Problem

SlidingWindowManager.remove_skipped_blocks is called on every scheduling step for each request under allocate_slots, but in the common case it removes no blocks. It still runs the full teardown path: indexing into req_to_blocks, building a removed_blocks list, and looping/comprehending over blocks even when nothing qualifies for removal. Under SlidingWindowAttention this accounts for ~25% of allocate_slots cost.

## Example

A request whose computed tokens have not yet exceeded the sliding window: last_useful_block <= 0, so no block can possibly be freed, yet the function still executes list operations against an effectively empty result (measured 6.3us per call).

## Use case

Online serving of models using sliding-window attention, where remove_skipped_blocks fires every decode step per active request inside the scheduler hot path.

## Resources

CPU time / per-step scheduler latency (kernel-launch-adjacent host overhead), not GPU — wasted Python function calls and list comprehensions on the critical scheduling path.

## Type of users

Users serving sliding-window-attention models (e.g. Mistral/Gemma-style SWA) under vLLM v1, especially at high request concurrency where per-step per-request overhead compounds.

## Kind of inputs

Requests early in generation (num_computed_tokens below or near sliding_window), and the steady-state decode case where the trailing block is already a null block so nothing needs removal.

## Mechanism with the issue

In single_type_kv_cache_manager.py remove_skipped_blocks, after computing last_useful_block from num_computed_tokens, sliding_window, and block_size, the code unconditionally fetched req_to_blocks[request_id], allocated removed_blocks, and entered the backward loop. When last_useful_block <= 0 or the block before it is already _null_block, this work produces nothing but still pays the call/alloc/loop cost.

# Solution

## Fix in words

Diff adds two lightweight guard checks before the removal loop. First: if last_useful_block <= 0, return immediately — not enough tokens to fill the sliding window, so no block is removable. Second: after fetching blocks, if blocks[last_useful_block - 1] == self._null_block, return — the boundary block is already null so there is nothing to remove. Both short-circuit before allocating removed_blocks and running the comprehension/loop, cutting the early-stop cost from 6.3us to 1.6us.

## Author's solution

Author leverages that most remove_skipped_blocks invocations remove no blocks, adding lightweight up-front checks to early-return and skip the function body's list comprehensions over empty lists. This eliminates redundant work in the common case, reducing per-call cost from 6.3us to 1.6us when early stop triggers.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/single_type_kv_cache_manager.py`

### Diff
```diff
diff --git a/vllm/v1/core/single_type_kv_cache_manager.py b/vllm/v1/core/single_type_kv_cache_manager.py
index 6699fb9818cb..575ae3d7d83b 100644
--- a/vllm/v1/core/single_type_kv_cache_manager.py
+++ b/vllm/v1/core/single_type_kv_cache_manager.py
@@ -394,7 +394,13 @@ def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> No
         # skipped during the attention computation.
         last_useful_token = num_computed_tokens - self.sliding_window + 1
         last_useful_block = last_useful_token // self.block_size
+        if last_useful_block <= 0:
+            # Early return if tokens are not enough to fill the sliding window
+            return
         blocks = self.req_to_blocks[request_id]
+        if blocks[last_useful_block - 1] == self._null_block:
+            # Early return if there are no blocks to remove
+            return
         removed_blocks: list[KVCacheBlock] = []
         for i in range(last_useful_block - 1, -1, -1):
             if blocks[i] == self._null_block:

```
