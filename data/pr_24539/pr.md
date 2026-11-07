# PR #24539: [Spec Decode] Efficient padded speculation

- field: decoding  | area: v1/worker
- author: benchislett
- state: MERGED  created: 2025-09-09T22:28:26Z  merged: 2025-09-18T05:07:24Z
- url: https://github.com/vllm-project/vllm/pull/24539
- labels: speculative-decoding, ready, v1

## Body

## Purpose

This PR implements padded speculative decoding as described in #21984. The implementation is the same as the one from #22684 (which itself came from #20078), with some cleanup and compatibility with latest main.

This PR does not add support for any new attention backends, such as #21984 or #24045. It will not take much work to enable for each new backend, but I think it is best to separate concerns into different PRs for this support. This PR represents the common core feature of padded speculation only.

When tested on Llama 3.1 8B-Instruct with EAGLE3 on 1xB200, this PR introduces a small 2% speedup on both TTFT and TPOT. Performance profiling confirmed that the implementation is free of GPU->CPU sync during the EAGLE drafting phase, but this performance gain is offset by the fact that CUDA graphs for the draft model are not working (see #23679) and that many small gpu operations are added in order to do sampled-token bookkeeping fully on the GPU.

Performance is expected to improve once new attention backends are supported, CUDA graph support for draft models is re-introduced, and a custom kernel is added to fuse the many small operations used in the bookkeeping (see #20078 for example).

The feature is now the default with opt-out via "disable_padded_batch" in the speculative_config. Performance is identical and behaviour is unchanged when disable_padded_batch is set.

## Test Plan

Ran locally with the setup described above and saw consistent, reasonable outputs in both cases. No new functionality is introduced so existing testing should be sufficient.

To reproduce benchmarks, the server can be launched with this configuration:
```
VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 4, "disable_padded_drafter_batch": false}' --max-model-len 2048 --no-enable-prefix-caching
```
- `disable_padded_drafter_batch` can be set to `true` to measure the baseline behaviour. When omitted, it will default to "false" (the new behaviour)

## Comments

### mergify @ 2025-09-11T07:36:19Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-16T14:38:10Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### benchislett @ 2025-09-16T17:46:55Z

@WoosukKwon the goal of this PR is not to introduce a raw speedup. Instead, it introduces foundational logic required to enable speculative decoding with new backends and features. To list a few:
- FlashInfer and FlashInfer-MLA (among many other) backends have optimized decode kernels (trtllm-gen) with special support for speculative decoding, but they require the input tensors to have regular shape (num_reqs, query_len, ...). This means that padding the batch is the only way to support these kernels. While it is possible to do the padding directly in the attention op, this approach is limited and does not support the other features below
- Padded batching makes overlapped execution and full-cuda graph support much easier to implement. See #24799 and #23679 which both rely on this feature. Eliminating the cpu-gpu synchronization allows hiding a lot of metadata preparation work, which cannot be properly benchmarked at this time since currently there is only support for FlashAttention. 
---
### LucasWilkinson @ 2025-09-17T04:16:58Z

We could simplify the situation by just making it the only pathway (per @benchislett initial suggestion; i.e. completely remove unpadded speculation) if that makes it more palatable @WoosukKwon , I generally was in favor of landing it feature flagged since it allows:

1) easier debugging if users run into issues (can ask them to set `--disable-padded-drafter-batch` if issues crop up to see if its related to the padding)
2) allows for easier benchmark comparisons to see how this impacts less common cases (i.e. cases with higher num-speculated tokens and thus likely higher number of rejected tokens) 
     - however the speculator generally doesn't take that long compared to the main model so I can definitely see justifications for just supporting one pathway to simplify the code given the differences are likely only to be at most a couple of percent
3) leaves a pathway for the tree attention implementation but since we are now de-priortizing tree attention; we don't need to take this into account as heavily (I think if we end up prioritizing tree attention in the future there's pathway to making compatible with padded speculation)

however I can be convinced just committing to padded speculation always to simplify the code could make sense (especially in light of de-prioritizing tree attention)

I do believe pretty strongly though that padded speculation is needed for the reasons @benchislett already mentioned.
---
### benchislett @ 2025-09-17T14:39:04Z

Re-requesting review from @WoosukKwon after an offline discussion. I understand if this is too much complexity to introduce without comprehensive review and careful refactoring, but I still feel that it is a powerful feature that we will need at some point.

In the immediate term, we need to make a decision quickly to unblock FlashInfer support. If this PR is still decided to be too complex we can merge another approach that will get it working. Something like #24045 with https://github.com/vllm-project/vllm/pull/24539/files#diff-a4809a837fbf535a8f0999b11087a53ec1c53948b50c0a1fe64396bc86de9461R282-R338 would probably be sufficient to get it functional and compatible with piecewise graphs.
---
### LucasWilkinson @ 2025-09-17T15:00:37Z

can you expand on "Something like https://github.com/vllm-project/vllm/pull/24045"? Im not sure this addresses the issue of ending up with query lengths like `[1, 2, 2, 1]` after rejections which will force FlashMLA to run as `[Decode, Prefill, Prefill, Prefill]` which will be quite a bit slower for that specific MTP layer; granted relative to the full model that may not be a concern
---