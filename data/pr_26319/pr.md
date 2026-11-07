# PR #26319: [Lora]Load tuned multi-lora kernel configs from json files

- field: gpu_kernels  | area: lora/ops
- author: li2haipeng
- state: MERGED  created: 2025-10-06T21:11:02Z  merged: 2025-10-15T09:45:15Z
- url: https://github.com/vllm-project/vllm/pull/26319
- labels: ready

## Body

## Purpose
Add support to load tuned multi-lora shrink/expand triton kernels configs from json files. 
I understand there was a discussion in https://github.com/vllm-project/vllm/pull/13096 that we don't want to save those json files. I was wondering if we can just simply keep the option for users in case they want to tune the kernels themselves. According to our benchmarking, we see 12% to 17% ITL gain on Llama3.3 70B with 16 lora adapters with rank=16:

<img width="569" height="344" alt="Screenshot 2025-10-06 at 14 08 46" src="https://github.com/user-attachments/assets/d3a0cacf-ea29-4762-8588-f75cbc14b8ad" />

The PR leverages the exact kernel loading function proposed from https://github.com/vllm-project/vllm/pull/13096.

Edit:
The lora config folder should be passed in by `export VLLM_TUNED_CONFIG_FOLDER=/path/to/configs`. Without it, the kernel would use default configs.



## Comments

### github-actions @ 2025-10-06T21:11:11Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### li2haipeng @ 2025-10-06T21:15:14Z

@varun-sundar-rabindranath @jeejeelee Could you help review the PR?
---
### jeejeelee @ 2025-10-07T00:58:14Z

Will look at this PR ASAP
---
### jeejeelee @ 2025-10-08T01:16:47Z

Yes, as you can see, these configs are too large, and considering there are too many various LoRA configurations, we don't want to keep these configurations. However, tuning can indeed improve performance. Our previous idea was:

-  Tell users how to tune and save the results through documentation
-  Use the env `VLLM_TUNED_CONFIG_FOLDER` to allow users to pass in tuned configurations (what this PR does)

@li2haipeng  If you agree or have better ideas, feel free to share your feedback. We can work together to move this forward

---
### li2haipeng @ 2025-10-08T17:19:20Z

Thanks for the feedback @jeejeelee. Yes, I agree with you. Users usually don't publish their own adapters so it's not necessary for us to keep those configs, an option to allow users to load their own configs is sufficient enough. Regarding the tuning doc, where should we save it? I can add the env in this PR.
---
### jeejeelee @ 2025-10-09T12:07:30Z

> Thanks for the feedback @jeejeelee. Yes, I agree with you. Users usually don't publish their own adapters so it's not necessary for us to keep those configs, an option to allow users to load their own configs is sufficient enough. Regarding the tuning doc, where should we save it? I can add the env in this PR.

Maybe `vllm/vllm/lora/ops/triton_ops/` is a pretty good choice,  also  cc @varun-sundar-rabindranath 
---
### li2haipeng @ 2025-10-10T00:53:08Z

@jeejeelee @varun-sundar-rabindranath A simple README is added, let me know if the PR is ready to merge.
---
### jeejeelee @ 2025-10-15T02:06:20Z

Aferting Removing  the `config/` and  fixing the pre-commit issues, LGTM
---