# PR #11870: [perf]fix current stream

- field: parallelism  | area: distributed/device_communicators
- author: youkaichao
- state: MERGED  created: 2025-01-09T02:00:36Z  merged: 2025-01-09T07:18:22Z
- url: https://github.com/vllm-project/vllm/pull/11870
- labels: ready

## Body

fix the performance regression reported from https://github.com/vllm-project/vllm/pull/11744#issuecomment-2578717013

on my local benchmark:

`python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-70B --load-format dummy --enforce-eager -tp 4`

main branch:

```text
Avg latency: 2.945735554069203 seconds
10% percentile latency: 2.924619035271462 seconds
25% percentile latency: 2.937671729727299 seconds
50% percentile latency: 2.9460502695292234 seconds
75% percentile latency: 2.955668824230088 seconds
90% percentile latency: 2.9639973257959356 seconds
99% percentile latency: 2.979829666109872 seconds
```

this PR:

```text
Avg latency: 2.851606635436959 seconds
10% percentile latency: 2.8231707043829375 seconds
25% percentile latency: 2.834942308269092 seconds
50% percentile latency: 2.85484445450129 seconds
75% percentile latency: 2.8674310567148495 seconds
90% percentile latency: 2.872856835933635 seconds
99% percentile latency: 2.875793117735884 seconds
```

it can have 3% perf diff.

Hopefully this can fix the perf regression observed in the benchmark:

<img width="1247" alt="image" src="https://github.com/user-attachments/assets/2f0698a3-88fe-425a-b461-8458e875bf23" />


## Comments

### github-actions @ 2025-01-09T02:03:00Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### youkaichao @ 2025-01-09T02:17:05Z

I find measuring the pure forward time makes more sense, it will not be affected by the scheduling, etc:

`VLLM_LOG_BATCHSIZE_INTERVAL=1 python benchmarks/benchmark_latency.py --model meta-llama/Meta-Llama-3-70B --load-format dummy --enforce-eager -tp 4`

main branch:

```text
Batchsize forward time stats (batchsize, count, median_time(ms)): [(8, 4998, 20.77), (256, 40, 28.99)]
```

this PR:

```text
Batchsize forward time stats (batchsize, count, median_time(ms)): [(8, 5027, 20.45), (256, 40, 28.95)]
```

The forward time for every step (batchsize 8) reduces from 20.77ms to 20.45ms.
---