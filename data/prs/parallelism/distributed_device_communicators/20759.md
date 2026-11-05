# PR #20759: [PERF] PyTorch Symmetric Memory All-Reduce

- field: parallelism  | area: distributed/device_communicators
- author: ilmarkov
- state: MERGED  created: 2025-07-10T14:31:44Z  merged: 2025-08-22T21:39:09Z
- url: https://github.com/vllm-project/vllm/pull/20759
- labels: documentation, performance, ready

## Body

Add an alternative to custom_allreduce and nccl on cuda - pytorch symmetric memory

Enabled by environment variable `VLLM_ALLREDUCE_USE_SYMM_MEM=1`.
Improves performance of TP allreduce for middle size input.

Bounds input sizes for custom allreduce as long as performance of two shot custom allreduce appears to be worse than nccl or pytorch symmetric memory based allreduce.

Max sizes for various world_sizes for custom allreduce and symmetric memory were chosen based on empirical results.

For world_sizes 2 and 4 pytorch two shot allreduce is used, for world sizes 6 and 8 pytorch multimem_all_reduce

## Benchmark results:
Server: `VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-70B-Instruct  --disable-log-requests --no-enable-prefix-caching -tp $tp`
Client: 
```
DURATION_SECONDS=60; \
vllm bench serve \
--model meta-llama/Llama-3.1-70B-Instruct \
--dataset-name sonnet  \
--dataset-path benchmarks/sonnet.txt \
--request-rate "$qps" \
--num-prompts $((DURATION_SECONDS * qps))`
```

On Blackwell, B200
### TP=4:
Baseline:

| QPS | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) |
|---|---|---|---|---|
| 1 | 45.021 | 44.248 | 12.880 | 12.866 |
| 5 | 48.574 | 46.048 | 14.120 | 14.084 |
| 10 | 55.482 | 51.188 | 17.674 | 17.710 |

PR:

|QPS|Mean TTFT (ms)|Median TTFT (ms)|Mean TPOT (ms)|Median TPOT (ms)|
|---|---|---|---|---|
|1|41.456|40.226|12.681|12.666|
|5|46.916|45.233|13.754|13.683|
|10|54.584|50.382|17.186|17.296|

### TP=8
Baseline:

|QPS|Mean TTFT (ms)|Median TTFT (ms)|Mean TPOT (ms)|Median TPOT (ms)|
|---|---|---|---|---|
|1|42.838|41.157|10.286|10.240|
|5|48.772|46.882|13.583|13.924|
|10|54.444|51.688|16.513|16.592|

PR:

|QPS|Mean TTFT (ms)|Median TTFT (ms)|Mean TPOT (ms)|Median TPOT (ms)|
|---|---|---|---|---|
|1|37.005|34.039|10.254|10.223|
|5|44.329|44.387|13.353|13.762|
|10|50.864|48.993|15.442|15.404|


Up to 8% TTFT speedup for TP=4
From 7 to 10% TTFT improvement, and up to 5-7% TPOT improvement for TP=8.

On Hopper, H100
### TP=8

Baseline:
|QPS|Mean TTFT (ms)|Median TTFT (ms)|Mean TPOT (ms)|Median TPOT (ms)|
|---|---|---|---|---|
|1|49.461|47.587|11.502|11.476|
|5|55.581|52.763|14.621|14.707|
|10|65.523|59.391|17.937|17.999|

PR:
| QPS | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | 
|-----|----------------|------------------|----------------|------------------|
| 1   | 46.016         | 44.858           | 11.449         | 11.424           | 
| 5   | 50.944         | 49.687           | 13.498         | 13.497           | 
| 10  | 59.219         | 54.619           | 16.132         | 16.202           |

Up to 10% TTFT improvement, and minor TPOT speedups.

## Validation:
model="meta-llama/Llama-3.1-70B-Instruct"
client: `lm_eval --model local-completions --model_args model=${model},base_url=http://localhost:8000/v1/completions --batch_size auto --trust_remote_code  --cache_requests true --tasks gsm8k --num_fewshot 5 --batch_size 200`
server: `VLLM_ALLREDUCE_USE_SYMM_MEM=1 vllm serve $model \
--disable-log-requests --no-enable-prefix-caching -tp 4` 
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9204|±  |0.0075|
|     |       |strict-match    |     5|exact_match|↑  |0.8817|±  |0.0089|
```



## Comments

### github-actions @ 2025-07-10T14:31:53Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### mergify @ 2025-07-11T13:00:12Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @ilmarkov.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### shixianc @ 2025-07-15T15:16:35Z

@ilmarkov hey do you know any reason why mutlicast_ptr == 0 on hopper? do I need specific version of torch?
---
### ilmarkov @ 2025-07-16T11:04:51Z

> @ilmarkov hey do you know any reason why mutlicast_ptr == 0 on hopper? do I need specific version of torch?

@shixianc I had similar issue, I had to use torch-nightly in order to make it work.
`pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128`

---
### mergify @ 2025-07-25T03:14:29Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @ilmarkov.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-08-13T13:17:48Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @ilmarkov.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mgoin @ 2025-08-22T21:39:05Z

Thanks for the additional review! Given the questions are not blockers, I think we should merge the functionality now and follow up with future work/default usage in discussion/PRs. This PR has been up for a while
---
### renjie0 @ 2025-08-25T19:39:38Z

A few questions:
1. Why is the limit lower for smaller world size?
2. Does the same improvement hold for MOE models
---