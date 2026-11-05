# PR #14990: [Feature] Enhance EAGLE Architecture with Proper RMS Norms

- field: attention  | area: vllm/config.py
- author: luyuzhe111
- state: MERGED  created: 2025-03-18T01:00:15Z  merged: 2025-03-26T08:24:07Z
- url: https://github.com/vllm-project/vllm/pull/14990
- labels: ready

## Body

In this PR, we allow the EAGLE architecture to optionally use proper RMS normalizations similar to the DeepSeek MTP module. It has two main benefits.

1) It improves acceptance length by ~5% when pre-attention norm, the output norm, and parallel norms (highlighted in red in the diagram below, taken from the DeepSeek V3 technical report) are used during training. Through ablation studies, we found all these RMS norms contribute to the improvement.
<img width="381" alt="Screenshot 2025-03-17 at 5 31 23 PM" src="https://github.com/user-attachments/assets/12c5d2b1-4998-42a2-ad02-5c2ed59e9e77" />

2) It alleviates the approximated KV cache bug currently in the vLLM EAGLE implementation as it reduces the caused performance degradation from 15% to 9% https://github.com/vllm-project/vllm/issues/14649 . Essentially, these additional RMS norms make sure the approximated KV cache is not too off, since the hidden states, from which those KV cache are computed, are normalized. The following table show the acceptance length for Llama 3 8B on GSM8K using this [example script](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/eagle.py).

| Number of Speculated Tokens | 1    | 2    | 3    | 4    | 5    |
|----------------------------|------|------|------|------|------|
| EAGLE Repo                | 1.84 | 2.45 | 2.83 | 3.07 | 3.17 |
| vLLM                      | 1.84 | 2.37 | 2.65 | 2.84 | 2.89 |
| Acceptance Length Drop      | 0%   | 3%   | 6%   | 7%   | 9%   |

With these normalizations, the performance degradation due to the approximated KV cache is now 0 ~ 9%, compared to the 8% ~ 15% drop without these normalizations. However, I want to emphasize that I do not intend to use this PR as a fix of https://github.com/vllm-project/vllm/issues/14649, but rather to show the community how to alleviate this bug temporarily while the bug is being fixed. 

cc @LiuXiaoxuanPKU Would appreciate your review. Thanks!



## Comments

### github-actions @ 2025-03-18T01:00:23Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### luyuzhe111 @ 2025-03-25T21:43:49Z

@WoosukKwon would appreciate your review as well!
---
### WoosukKwon @ 2025-03-26T03:49:53Z

Thank you @DarkLight1337!
---