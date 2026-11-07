# PR #28280: [Kernel] Optimization of the mm_k operator.

- field: gpu_kernels  | area: lora/ops
- author: caozuoba
- state: MERGED  created: 2025-11-07T08:43:39Z  merged: 2025-11-10T16:03:47Z
- url: https://github.com/vllm-project/vllm/pull/28280
- labels: ready

## Body

<!-- markdownlint-disable -->

## Purpose

The most significant change lies in the handling of the not `EVEN_K` scenario. In the main branch implementation, each loop iteration in this case incurs additional masking operations and `tl.dot()` calls. However, my analysis reveals that masking is only necessary for partially out-of-bound accesses. The other two scenarios are:

1. Completely within bounds: This case avoids the overhead of masking and requires only the `tl.dot()` operation.
2. Completely out of bounds: This can be skipped entirely, with no need for masking or `tl.dot()` operations.

By reducing redundant masking and `tl.dot()` operations, this modification improves computational speed. The optimization is particularly effective and friendly for GPUs with less advanced architectures. Performance results on the Metax C500 and H800 GPUs are as follows:

---

### use C500
**main**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  37.34
Total input tokens:                      14351
Total generated tokens:                  10000
Request throughput (req/s):              2.68
Output token throughput (tok/s):         267.82
Total Token throughput (tok/s):          652.17
---------------Time to First Token----------------
Mean TTFT (ms):                          726.72
Median TTFT (ms):                        738.68
P99 TTFT (ms):                           817.86
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          67.93
Median TPOT (ms):                        67.48
P99 TPOT (ms):                           72.62
---------------Inter-token Latency----------------
Mean ITL (ms):                           67.26
Median ITL (ms):                         67.19
P99 ITL (ms):                            74.41
==================================================
```

**this PR**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  33.68
Total input tokens:                      14351
Total generated tokens:                  10000
Request throughput (req/s):              2.97
Output token throughput (tok/s):         296.95
Total Token throughput (tok/s):          723.10
---------------Time to First Token----------------
Mean TTFT (ms):                          610.68
Median TTFT (ms):                        697.51
P99 TTFT (ms):                           799.74
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          61.74
Median TPOT (ms):                        60.56
P99 TPOT (ms):                           66.31
---------------Inter-token Latency----------------
Mean ITL (ms):                           61.12
Median ITL (ms):                         60.00
P99 ITL (ms):                            66.01
==================================================
```

---

### use H800
**main**
```text
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             20
Benchmark duration (s):                  19.24
Total input tokens:                      28840
Total generated tokens:                  20000
Request throughput (req/s):              10.40
Output token throughput (tok/s):         1039.72
Peak output token throughput (tok/s):    1176.00
Peak concurrent requests:                40.00
Total Token throughput (tok/s):          2539.00
---------------Time to First Token----------------
Mean TTFT (ms):                          138.79
Median TTFT (ms):                        157.32
P99 TTFT (ms):                           197.43
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.95
Median TPOT (ms):                        17.76
P99 TPOT (ms):                           19.06
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.77
Median ITL (ms):                         17.53
P99 ITL (ms):                            27.64
==================================================
```

**this PR**
```text
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             20
Benchmark duration (s):                  18.98
Total input tokens:                      28840
Total generated tokens:                  20000
Request throughput (req/s):              10.54
Output token throughput (tok/s):         1053.84
Peak output token throughput (tok/s):    1160.00
Peak concurrent requests:                40.00
Total Token throughput (tok/s):          2573.47
---------------Time to First Token----------------
Mean TTFT (ms):                          120.86
Median TTFT (ms):                        103.84
P99 TTFT (ms):                           198.83
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.91
Median TPOT (ms):                        17.98
P99 TPOT (ms):                           18.68
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.74
Median ITL (ms):                         17.39
P99 ITL (ms):                            30.92
==================================================
```

---

## Test plan
```bash
pytest /vllm/tests/lora/test_punica_ops.py -v
```

## Test result
All test cases passed.


---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### github-actions @ 2025-11-07T08:43:47Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### mergify @ 2025-11-07T08:44:20Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @caozuoba.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### caozuoba @ 2025-11-09T10:35:37Z

### use 4090
*Add an additional set of experiments on the RTX 4090 GPU. Benchmark duration ↓5.2%，Output token throughput ↑5.5%*

**main**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  7.29
Total input tokens:                      12308
Total generated tokens:                  20000
Request throughput (req/s):              13.71
Output token throughput (tok/s):         2742.02
Total Token throughput (tok/s):          4429.46
---------------Time to First Token----------------
Mean TTFT (ms):                          925.91
Median TTFT (ms):                        920.06
P99 TTFT (ms):                           1522.73
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.53
Median TPOT (ms):                        31.57
P99 TPOT (ms):                           34.45
---------------Inter-token Latency----------------
Mean ITL (ms):                           31.53
Median ITL (ms):                         28.88
P99 ITL (ms):                            214.31
==================================================
```

**this PR**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  6.91
Total input tokens:                      12308
Total generated tokens:                  20000
Request throughput (req/s):              14.48
Output token throughput (tok/s):         2895.75
Total Token throughput (tok/s):          4677.79
---------------Time to First Token----------------
Mean TTFT (ms):                          901.27
Median TTFT (ms):                        980.17
P99 TTFT (ms):                           1513.23
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.69
Median TPOT (ms):                        29.33
P99 TPOT (ms):                           32.19
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.69
Median ITL (ms):                         27.30
P99 ITL (ms):                            216.03
==================================================
```

---
### caozuoba @ 2025-11-09T17:09:09Z

### use 3090
*Experiments on the RTX 3090 GPU are complete. Output token throughput increases by approximately 8%, consistent with the intuition that this change is more friendly to older GPU architectures.*

**main**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  11.66
Total input tokens:                      12308
Total generated tokens:                  20000
Request throughput (req/s):              8.58
Output token throughput (tok/s):         1715.49
Total Token throughput (tok/s):          2771.21
---------------Time to First Token----------------
Mean TTFT (ms):                          1896.88
Median TTFT (ms):                        1893.38
P99 TTFT (ms):                           3099.23
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          48.36
Median TPOT (ms):                        48.42
P99 TPOT (ms):                           53.71
---------------Inter-token Latency----------------
Mean ITL (ms):                           48.37
Median ITL (ms):                         43.09
P99 ITL (ms):                            455.56
==================================================
```

**this PR**
```text
============ Serving Benchmark Result ============
Successful requests:                     100
Benchmark duration (s):                  10.80
Total input tokens:                      12308
Total generated tokens:                  20000
Request throughput (req/s):              9.26
Output token throughput (tok/s):         1851.84
Total Token throughput (tok/s):          2991.46
---------------Time to First Token----------------
Mean TTFT (ms):                          1775.82
Median TTFT (ms):                        1767.30
P99 TTFT (ms):                           3039.51
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.70
Median TPOT (ms):                        44.76
P99 TPOT (ms):                           50.04
---------------Inter-token Latency----------------
Mean ITL (ms):                           44.70
Median ITL (ms):                         39.33
P99 ITL (ms):                            445.80
==================================================
```

---
### caozuoba @ 2025-11-10T11:45:43Z

### use H800 (rebenchmarked after rebase)
*After resolving rebase conflicts, reran the benchmark on the H800:*

```text
============ Serving Benchmark Result ============
Successful requests:                     200
Failed requests:                         0
Maximum request concurrency:             20
Benchmark duration (s):                  18.73
Total input tokens:                      28840
Total generated tokens:                  20000
Request throughput (req/s):              10.68
Output token throughput (tok/s):         1067.73
Peak output token throughput (tok/s):    1180.00
Peak concurrent requests:                40.00
Total Token throughput (tok/s):          2607.39
---------------Time to First Token----------------
Mean TTFT (ms):                          127.40
Median TTFT (ms):                        138.11
P99 TTFT (ms):                           204.81
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.56
Median TPOT (ms):                        17.44
P99 TPOT (ms):                           18.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.39
Median ITL (ms):                         17.07
P99 ITL (ms):                            27.75
==================================================
```

---