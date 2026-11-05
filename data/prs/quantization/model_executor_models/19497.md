# PR #19497: [Models] Improve iteration over layers

- field: quantization  | area: model_executor/models
- author: lgeiger
- state: MERGED  created: 2025-06-11T15:46:15Z  merged: 2025-08-29T01:26:35Z
- url: https://github.com/vllm-project/vllm/pull/19497
- labels: ready, llama, qwen, deepseek

## Body

## Purpose

Slicing a `torch.nn.ModuleList` as done in `self.layers[self.start_layer:self.end_layer]` [creates a new `torch.nn.ModuleList`](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/modules/container.py#L319-L324). This is pretty fast but not free and can't be overlapped with GPU compute as it executes at the start of the `Module.forward()`.

This also shows up in a model profile (this example uses eager mode):
![Screenshot 2025-06-11 at 13 11 41](https://github.com/user-attachments/assets/78d9ecea-d906-4258-8f87-74d4d4b1834f)

This PR changes the iteration to use `itertools.islice` which removes this overhead since [iteration doesn't create new module lists](https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/torch/nn/modules/container.py#L344-L346) and can now be overlapped with GPU computation.

This only has a very minor performance impact. In a quick benchmark I'm seeing 0-0.9% faster time to first token and improved throughput though results are noisy. I don't think this hurts readability so I still think it's a worthwhile change.


## Test Plan

Covered by existing unittest

## Test Result

CI


## Comments

### github-actions @ 2025-06-11T15:46:24Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### lgeiger @ 2025-06-18T12:59:58Z

Not sure who's the best person to ask for a review. Maybe @DarkLight1337 or @ywang96?
---
### mergify @ 2025-07-02T13:05:10Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @lgeiger.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### njhill @ 2025-08-26T22:31:32Z

I like this change but others may have opinions about negatively impacting code readability.
---
### DarkLight1337 @ 2025-08-27T00:43:39Z

I'm fine with this change - `islice` isn't really a big deal
---
### lgeiger @ 2025-08-27T09:37:11Z

Sounds great! I rebased the PR and resolved all the conflicts.

I didn't change layer iterations like
```python
for i in range(self.start_layer, self.end_layer):
    layer = self.layers[i]
```
to `islice` since there are no performance downsides to this but let me know if I should also switch this to `islice`.
---
### DarkLight1337 @ 2025-08-27T09:52:47Z

Feel free to switch them over as well
---
### lgeiger @ 2025-08-27T10:05:55Z

> Feel free to switch them over as well

Done in 8175873ed3fcbd11c823b6eae06e51271c9e0d35
---
### DarkLight1337 @ 2025-08-27T10:25:56Z

Thanks for the optimization!
---
### lgeiger @ 2025-08-28T19:28:45Z

Rebased to make CI green 💚 
---