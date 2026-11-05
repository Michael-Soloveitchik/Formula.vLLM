# PR #8651: [Core] Default to using per_token quantization for fp8 when cutlass is supported.

- field: quantization  | area: model_executor/layers
- author: elfiegg
- state: MERGED  created: 2024-09-20T04:56:02Z  merged: 2025-01-16T04:31:28Z
- url: https://github.com/vllm-project/vllm/pull/8651
- labels: ready

## Body

Default to using per_token quantization for fp8 when cutlass is supported (so that dequantization will be fused with cutlass gemm kernel), which improves performance by enabling fusion of amax + quantization. 

Tested with Llama2-7B on H100:
Before: Throughput: 152.60 requests/s, 10681.75 tokens/s. 
After: Throughput: 159.73 requests/s, 11181.40 tokens/s vs.
About 4.68% speedup. Accuracy remains similar if not slightly higher. 


## Comments

### github-actions @ 2024-09-20T04:56:15Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---