# PR #19169: [Core] Batch multi modal input using pinned memory

- field: scheduling  | area: multimodal/inputs.py
- author: lgeiger
- state: MERGED  created: 2025-06-04T23:20:27Z  merged: 2025-06-10T05:44:59Z
- url: https://github.com/vllm-project/vllm/pull/19169
- labels: ready, v1, multi-modality

## Body

Since #18862 we simply copy the batched tensors to GPU memory without any additional casting.
This allows use to stack the inputs into pinned memory for faster CPU->GPU copy and maybe slightly faster torch.stack speed.

**Before:**
<img width="523" alt="Screenshot 2025-06-05 at 01 16 38" src="https://github.com/user-attachments/assets/03affad6-5e0f-4afd-bb54-c9b46c745126" />

**After:**
<img width="869" alt="Screenshot 2025-06-05 at 01 19 22" src="https://github.com/user-attachments/assets/db77091d-d2a4-47d2-b441-17734b1bd7c6" />


## Comments

### github-actions @ 2025-06-04T23:20:37Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### DarkLight1337 @ 2025-06-05T03:43:40Z

The idea looks reasonable. How much speed up have you observed from this?
---
### lgeiger @ 2025-06-05T15:50:14Z

> The idea looks reasonable. How much speed up have you observed from this?

On some internal with multiple images per prompt I'm seeing around a 0.8-1.7% improvement in throughput on a gemma 3 4b model. With the `lmarena-ai/VisionArena-Chat` it's very noise maybe a 0.3% improvement averaged across 3 runs.
---
### DarkLight1337 @ 2025-06-09T02:23:57Z

PTAL at the failing multimodal test
---
### lgeiger @ 2025-06-09T15:45:28Z

> PTAL at the failing multimodal test

Good catch, fixed in 091e858d8e071d423494adc829e987bf8193ea64
---
### gemini-code-assist @ 2025-06-10T05:45:02Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---