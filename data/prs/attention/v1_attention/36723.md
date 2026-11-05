# PR #36723: [DSV3.2][MTP] Optimize Indexer MTP handling

- field: attention  | area: v1/attention
- author: benchislett
- state: MERGED  created: 2026-03-10T23:43:27Z  merged: 2026-03-11T04:16:56Z
- url: https://github.com/vllm-project/vllm/pull/36723
- labels: ready, v1

## Body

## Purpose

Usage of `repeat_interleave` in MLA indexer MTP batch expansion logic causes gpu/cpu synchronization which breaks async scheduling, causing massive performance degredation. ~When the requests in the batch have uniform query length (common case for speculative decoding) the shapes are consistent and we can use a nonblocking implementation.~

Alternative implementation suggested by @MatthewBonanni is to pass output_size to repeat_interleave to avoid the sync. Works just as well.

**Approximately 30% speedup** for DSV3.2 NVFP4 MTP=3 on 8xB200 TP8.

## Testing

GSM8k passes, 95% accuracy for DSV3.2 NVFP4 + FP8 KV Cache and MTP=3.

Confirmed with extensive instrumentation, profiling and benchmarking that the async scheduling is fully restored.

## Comments

### MatthewBonanni @ 2026-03-11T00:44:04Z

This was introduced by #34552, which was included in v0.17.0
---
### MatthewBonanni @ 2026-03-11T01:05:13Z

Would just passing `output_size=actual_expanded` to the `repeat_interleave` calls be sufficient to avoid the sync?
---
### benchislett @ 2026-03-11T01:43:09Z

Yep, done!
---
### robertgshaw2-redhat @ 2026-03-11T01:55:03Z

is this a bugfix or a new feature @MatthewBonanni ?
---
### MatthewBonanni @ 2026-03-11T01:58:57Z

@robertgshaw2-redhat important bugfix, addresses a substantial perf regression in 0.17.0
---