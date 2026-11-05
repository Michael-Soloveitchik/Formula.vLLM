# PR #11394: [V1] Use FlashInfer Sampling Kernel for Top-P & Top-K Sampling

- field: decoding  | area: v1/sample
- author: WoosukKwon
- state: MERGED  created: 2024-12-21T06:58:08Z  merged: 2024-12-27T00:32:38Z
- url: https://github.com/vllm-project/vllm/pull/11394
- labels: ready

## Body

This PR brings FlashInfer's top-p & top-k sampling kernel to the V1 sampler.

The PyTorch-native implementation of top-p and top-k sampling requires sorting the entire logits tensor, which could be very large when the batch size is large. As a result, the sorting op can take ~20% performance overhead in high-throughput settings. By leveraging the FlashInfer's kernel, we can reduce this down to almost zero.

ShareGPT benchmark: 
* `main` w/o top-p & top-k: 47.40 reqs/s
* `main` w/ top-p: 40.20 reqs/s
* This RP w/ top-p: 47.15 reqs/s

## Comments

### github-actions @ 2024-12-21T06:58:18Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### WoosukKwon @ 2024-12-26T12:41:19Z

@sroy745 @comaniac Could you please take a look?
---
### sirouk @ 2025-05-31T04:10:35Z

@WoosukKwon thanks for this 🙏 

Do you think this could be related?
https://github.com/vllm-project/vllm/issues/18811
---