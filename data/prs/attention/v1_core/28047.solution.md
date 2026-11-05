# PR #28047: [Hybrid] Mamba2 prefix cache blocks freeing for running requests

- field: attention  | area: v1_core
- author: s3woz
- state: MERGED  created: 2025-11-04T14:46:09Z  merged: 2025-12-25T20:54:06Z
- url: https://github.com/vllm-project/vllm/pull/28047
- labels: ready, v1

# Issue

## Problem

For Mamba2 hybrid models, vLLM's prefix-cache KV manager kept ALL prefix cache blocks pinned for running requests, preventing their eviction. Because Mamba2 is a recurrent/state-space model, only the state of the last computed token is actually required to continue decoding; the state blocks of all earlier computed tokens are never read again. Pinning them wasted the finite KV-cache memory pool and blocked the prefix-caching mechanism from reusing that memory.

## Example

Granite-4.0-tiny with a 65536-token prompt at batch-size 8 (prefix caching on): 18% latency reduction after fix. With a 100000-token prompt in a memory-tight case: latency drops from ~11.0s to ~0.49s (96% reduction) because freed blocks avoid thrashing/recompute.

## Use case

Serving long-context prompts with APC (automatic prefix caching) enabled on hybrid Mamba2 models under tight GPU memory.

## Resources

GPU HBM / KV-cache memory pool. Pinned state blocks starve the eviction pool, forcing recompute and inflating latency.

## Type of users

Users serving hybrid Mamba2/SSM models (e.g. Granite 4.0) with long-context prompts and prefix caching, especially in memory-constrained deployments.

## Kind of inputs

Long prompts (65k–100k tokens), batched requests, prefix-caching enabled, high memory pressure.

## Mechanism with the issue

In `single_type_kv_cache_manager.py`, the Mamba manager lacked a way to signal which computed tokens' blocks were no longer needed. Without `get_num_skipped_tokens`, the base manager treated all computed-token blocks as live for running requests, so none were released to the eviction free-list even though Mamba discards past per-token state.

# Solution

## Fix in words

Diff adds `get_num_skipped_tokens(num_computed_tokens)` to the Mamba manager, returning `num_computed_tokens - 1`. Before: no such override, so every computed-token block stayed pinned for running requests. After: all computed-token blocks except the last one are reported as skippable, so the KV-cache manager marks them available for eviction (added to free-list, not forcibly evicted). This lets the prefix-caching mechanism reclaim that memory under pressure while still keeping blocks reusable until actually needed, cutting latency sharply in tight-memory long-prompt cases.

## Author's solution

Author implements freeing of prefix cache blocks of running requests for Mamba2 hybrid models. Instead of keeping all prefix blocks pinned, the PR marks blocks not strictly needed by running requests as available for eviction — added to the pool of memory potentially reclaimable if more memory is needed, but not evicted immediately, so they stay available for prefix caching unless explicitly reclaimed. Benefits are largest for long prompts under tight memory.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/single_type_kv_cache_manager.py`

### Diff
```diff
diff --git a/vllm/v1/core/single_type_kv_cache_manager.py b/vllm/v1/core/single_type_kv_cache_manager.py
index 4aeb17a156bb..e8a0a39b1f90 100644
--- a/vllm/v1/core/single_type_kv_cache_manager.py
+++ b/vllm/v1/core/single_type_kv_cache_manager.py
@@ -737,6 +737,14 @@ def allocate_new_blocks(
             )
         return super().allocate_new_blocks(request_id, num_tokens)
 
+    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
+        """
+        Get the number of tokens whose mamba state are not needed anymore. Mamba only
+        need to keep the state of the last computed token, so we return
+        num_computed_tokens - 1.
+        """
+        return num_computed_tokens - 1
+
 
 class CrossAttentionManager(SingleTypeKVCacheManager):
     """Manager for cross-attention KV cache in encoder-decoder models."""

```
