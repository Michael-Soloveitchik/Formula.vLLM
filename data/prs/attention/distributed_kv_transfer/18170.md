# PR #18170: [V1][P/D] Local attention optimization for NIXL

- field: scheduling  | area: distributed/kv_transfer
- author: mgoin
- state: MERGED  created: 2025-05-14T21:04:22Z  merged: 2025-05-17T01:16:33Z
- url: https://github.com/vllm-project/vllm/pull/18170
- labels: ready, v1

## Body

Optimizes the V1 NixlConnector for Llama 4 models by implementing layer-specific KV cache transfers.

During the worker's `register_kv_caches` phase, it pre-calculates the attention window size (in blocks) for each layer based on the Llama 4 config (`no_rope_layers` and `attention_chunk_size`). 
For RoPE layers (local attention), this identifies the specific chunk of KV cache needed, while NoPE layers (global attention) are marked to use their full cache.

When a KV transfer is initiated in `start_load_kv`, the connector iterates through each layer. If a layer is identified as having a local attention window, the list of physical block IDs to be transferred is truncated to match this window, selecting only the most recent, relevant blocks. The `_get_block_descs_ids` method now uses a `layer_idx` to generate NIXL descriptors corresponding only to the memory regions of that specific layer and the selected (potentially chunked) block IDs.

This reduces the data transferred for Llama 4's RoPE layers (local attention with chunk_size=8192), leading to improved TTFT. Thanks to @tlrmchlsmth for benchmarking, showing a reduction in multi-node NIXL P/D overhead from ~10% to ~5% on 4xH200 Llama-4-Scout for 100k input tokens, compared to TTFT on a single node.

For a prefill of 100k tokens, we are able to transfer 69% less data since 3/4 layers have an attention_chunk_size of 8192 (`(100000*1/4)+(8192*3/4) = 31,144`)

Calculating per-layer transfer limits and using layer-indexed descriptor generation provides an example for supporting other attention types like sliding window attention in the future. 


## Comments

### github-actions @ 2025-05-14T21:04:32Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### WoosukKwon @ 2025-05-15T02:10:04Z

I think this PR is not needed once #17996 and a followup PR (for Llama4 support) are merged.
cc @heheda12345 for confirmation.
---
### heheda12345 @ 2025-05-15T07:22:06Z

Agree with @WoosukKwon. You can get the block_ids that need to be transfer from P node to D node with https://github.com/vllm-project/vllm/pull/17996. Let's discuss more after that PR.
---
### mgoin @ 2025-05-15T13:00:29Z

It makes sense to use the hybrid memory allocator eventually, however we needed this optimization now so that is why I worked on it.
Either way, the logic to get the block descs will need to change similarly. Since the changes are local to the NIXL connector I don't see much harm in landing with intent to remove in the long-term, but I understand.
---
### WoosukKwon @ 2025-05-16T05:30:31Z

@mgoin Got it. I'm comfortable merging this PR as a temporary fix. Please just add a `TODO` comment.
---