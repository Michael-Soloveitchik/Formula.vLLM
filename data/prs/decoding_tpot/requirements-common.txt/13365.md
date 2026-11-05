# PR #13365: [V1][Spec Decode] Optimize N-gram matching with Numba

- field: decoding  | area: requirements-common.txt
- author: WoosukKwon
- state: MERGED  created: 2025-02-17T00:00:28Z  merged: 2025-02-18T21:19:58Z
- url: https://github.com/vllm-project/vllm/pull/13365
- labels: ready, ci/build, v1

## Body

This PR optimizes the N-gram matching algorithm by JIT compiling it with Numba.
I've observed 20-30x speedup with large batch sizes: For ShareGPT benchmark with 5K requests, the cumulative overhead reduces from 54.3 sec to 1.9 sec, which is ~2.5% of the entire running time.

## Comments

### github-actions @ 2025-02-17T00:00:39Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### WoosukKwon @ 2025-02-17T23:49:42Z

cc @LiuXiaoxuanPKU This PR is ready. Could you please take a look?
---
### michaelfeil @ 2025-02-19T18:58:58Z

Very excited about this!
---
### WoosukKwon @ 2025-02-19T19:05:57Z

@michaelfeil Thanks! Happy to see you again :)
We still have some headroom for performance: #13498 Please let us know if you are interested in working on this.
---