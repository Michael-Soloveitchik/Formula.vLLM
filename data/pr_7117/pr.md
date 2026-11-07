# PR #7117: [core][misc] simply output processing with shortcut for non-parallel sampling and non-beam search usecase

- field: scheduling  | area: engine/output_processor
- author: youkaichao
- state: MERGED  created: 2024-08-04T01:49:58Z  merged: 2024-08-04T07:22:19Z
- url: https://github.com/vllm-project/vllm/pull/7117
- labels: 

## Body

(empty)

## Comments

### youkaichao @ 2024-08-04T01:50:07Z

it is a tiny step of https://github.com/vllm-project/vllm/issues/7116
---
### github-actions @ 2024-08-04T01:50:09Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### youkaichao @ 2024-08-04T01:52:15Z

On H100:

```python
python benchmarks/benchmark_throughput.py --input-len 256 --output-len 256 --model meta-llama/Meta-Llama-3-8B
```

Main branch (3 trials):

```text
Throughput: 24.32 requests/s, 12453.04 tokens/s
Throughput: 24.11 requests/s, 12342.76 tokens/s
Throughput: 24.29 requests/s, 12434.40 tokens/s
```

This PR (3 trials):

```text
Throughput: 24.52 requests/s, 12555.73 tokens/s
Throughput: 24.59 requests/s, 12591.35 tokens/s
Throughput: 24.74 requests/s, 12667.68 tokens/s
```

Already 1.5% improvement in throughput.
---