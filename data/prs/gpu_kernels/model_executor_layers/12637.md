# PR #12637: Apply torch.compile to fused_moe/grouped_topk

- field: parallelism  | area: model_executor/layers
- author: mgoin
- state: MERGED  created: 2025-02-01T03:16:21Z  merged: 2025-02-01T16:16:19Z
- url: https://github.com/vllm-project/vllm/pull/12637
- labels: ready

## Body

Inspired by https://github.com/sgl-project/sglang/commit/1ebe1d6de5e0ce082e0be059c222baf0c5ee340a. This improves the generation latency for deepseek models due to the large number of experts. For `DeepSeek-Coder-V2-Lite-Instruct` we see an increase of 154 -> 163 tokens/s or 5% improvement at bs=1 on H100

script:
```
VLLM_MLA_DISABLE=1 python benchmarks/benchmark_latency.py --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --max-model-len 2048 --trust-remote-code --batch-size 1 --input-len 128 --output-len 512
```

main:
```
Avg prompt throughput: 51.1 tokens/s, Avg generation throughput: 154.8 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
```

this pr:
```
Avg prompt throughput: 51.2 tokens/s, Avg generation throughput: 163.8 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%.
```

## Comments

### github-actions @ 2025-02-01T03:16:32Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### mergify @ 2025-02-01T05:53:35Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @mgoin.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---