# PR #14930: [V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels

- field: decoding  | area: v1/sample
- author: WoosukKwon
- state: MERGED  created: 2025-03-17T06:29:41Z  merged: 2025-03-18T21:31:54Z
- url: https://github.com/vllm-project/vllm/pull/14930
- labels: speculative-decoding, ready, v1

## Body

This PR optimizes the rejection sampler in #13933 with custom Triton kernels. 

By using the Triton kernels, the PR brings the following benefits:
1. Now we use the flattened shape `[num_tokens, vocab_size]` for the logits tensors, instead of `[batch_size, max_spec_len, vocab_size]`. This reduces the GPU memory usage a lot.
2. Zero synchronization between CPU and GPU.
3. Remove inefficient data movement (i.e., a bunch of `cat`, `gather`, etc.)
4. (Arguably) easier-to-read code

### Performance benchmark: Llama 3.1 8B, ShareGPT, 1xH100, temperature 0.1
SD config: `--speculative-model "[ngram]" --ngram_prompt_lookup_min 5 --ngram-prompt-lookup-max 5 --num_speculative_tokens 3`

<img width="880" alt="Screenshot 2025-03-17 at 11 28 14 AM" src="https://github.com/user-attachments/assets/a80e58b7-00de-4eda-9503-9bd3c3195623" />


| | Throughput (reqs/s) |
|----------|:----------:|
| main (w/o SD) | 51.49 |
| main (w/ SD) | 54.41 |
| This PR (w/ SD) | 64.16 |

25% throughput increase compared to main w/o SD, and 18% increase compared to main w/ SD.

### Accuracy benchmark: GSM8K, Llama 3.1 8B Instruct, 5 shots

| | Temperature | Exact match |
|----------|:----------:|:----------:|
| w/o SD | 0.0 | 75.7 |
| | 1.0 | 50.9 |
| w/ SD | 0.0 | 75.9 |
| | 1.0 | 51.8|

## Comments

### github-actions @ 2025-03-17T06:29:50Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---