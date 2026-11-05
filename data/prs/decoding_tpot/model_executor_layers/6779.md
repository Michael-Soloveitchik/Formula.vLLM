# PR #6779: [Core] Use array to speedup padding

- field: decoding  | area: model_executor/layers
- author: peng1999
- state: MERGED  created: 2024-07-25T09:08:36Z  merged: 2024-07-26T04:31:31Z
- url: https://github.com/vllm-project/vllm/pull/6779
- labels: ready

## Body

Using `array.array` in SequenceData greatly improves performance of `make_tensor_with_pad` in Sampler.

Micro-benchmark using 1024 input length and 2048 batch size shows a great latency improvment (79ms to 22ms):

Before:
<img width="1146" alt="图片" src="https://github.com/user-attachments/assets/1862534d-9096-4d0a-91a2-bb9f20f441d1">

After:
<img width="1146" alt="图片" src="https://github.com/user-attachments/assets/e49ea252-00fc-4336-ba95-7e639b683df6">

End-to-end test on qwen-1.5-0.5b model also shows improvement on throughput:

main:
```
Processed prompts: 100%|███| 2048/2048 [01:22<00:00, 24.76it/s, est. speed input: 25352.26 toks/s, output: 3165.44 toks/s]
```
This PR:
```
Processed prompts: 100%|███| 2048/2048 [01:09<00:00, 29.44it/s, est. speed input: 30150.97 toks/s, output: 3764.60 toks/s]
```

**BEFORE SUBMITTING, PLEASE READ THE CHECKLIST BELOW AND FILL IN THE DESCRIPTION ABOVE**

---

<details>
<!-- inside this <details> section, markdown rendering does not work, so we use raw html here. -->
<summary><b> PR Checklist (Click to Expand) </b></summary>

<p>Thank you for your contribution to vLLM! Before submitting the pull request, please ensure the PR meets the following criteria. This helps vLLM maintain the code quality and improve the efficiency of the review process.</p>

<h3>PR Title and Classification</h3>
<p>Only specific types of PRs will be reviewed. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:</p>
<ul>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Doc]</code> for documentation fixes and improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Frontend]</code> For changes on the vLLM frontend (e.g., OpenAI API server, <code>LLM</code> class, etc.) </li>
    <li><code>[Kernel]</code> for changes affecting CUDA kernels or other compute kernels.</li>
    <li><code>[Core]</code> for changes in the core vLLM logic (e.g., <code>LLMEngine</code>, <code>AsyncLLMEngine</code>, <code>Scheduler</code>, etc.)</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix (e.g., <code>[Hardware][AMD]</code>).</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If the PR spans more than one category, please include all relevant prefixes.</p>

<h3>Code Quality</h3>

<p>The PR need to meet the following code quality standards:</p>

<ul>
    <li>We adhere to <a href="https://google.github.io/styleguide/pyguide.html">Google Python style guide</a> and <a href="https://google.github.io/styleguide/cppguide.html">Google C++ style guide</a>.</li>
    <li>Pass all linter checks. Please use <a href="https://github.com/vllm-project/vllm/blob/main/format.sh"><code>format.sh</code></a> to format your code.</li>
    <li>The code need to be well-documented to ensure future contributors can easily understand the code.</li>
    <li>Include sufficient tests to ensure the project to stay correct and robust. This includes both unit tests and integration tests.</li>
    <li>Please add documentation to <code>docs/source/</code> if the PR modifies the user-facing behaviors of vLLM. It helps vLLM user understand and utilize the new features or changes.</li>
</ul>

<h3>Notes for Large Changes</h3>
<p>Please keep the changes as concise as possible. For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue (RFC) discussing the technical design and justification. Otherwise, we will tag it with <code>rfc-required</code> and might not go through the PR.</p>

<h3>What to Expect for the Reviews</h3>

<p>The goal of the vLLM team is to be a <i>transparent reviewing machine</i>. We would like to make the review process transparent and efficient and make sure no contributor feel confused or frustrated. However, the vLLM team is small, so we need to prioritize some PRs over others. Here is what you can expect from the review process: </p>

<ul>
    <li> After the PR is submitted, the PR will be assigned to a reviewer. Every reviewer will pick up the PRs based on their expertise and availability.</li>
    <li> After the PR is assigned, the reviewer will provide status update every 2-3 days. If the PR is not reviewed within 7 days, please feel free to ping the reviewer or the vLLM team.</li>
    <li> After the review, the reviewer will put an <code> action-required</code> label on the PR if there are changes required. The contributor should address the comments and ping the reviewer to re-review the PR.</li>
    <li> Please respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.
 </li>
</ul>

<h3>Thank You</h3>

<p> Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM. Your contributions make vLLM a great tool for everyone! </p>


</details>




## Comments

### github-actions @ 2024-07-25T09:08:49Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### casper-hansen @ 2024-07-25T12:28:40Z

Nice to see an 18% speedup from this optimization. Is it mainly for small models?
---
### peng1999 @ 2024-07-25T14:43:46Z

> Is it mainly for small models?

Yes. This PR is for small models and large batch sizes. The `from_sampling_metadata` function, optimized by this PR, primarily runs on the CPU and is independent of logists. Therefore, it can overlap with the GPU work of model inference. It will only be on the critical path if its execution time exceeds that of model inference, which occurs with smaller models.
---
### daquexian @ 2024-07-25T15:31:33Z

Great PR! Would you mind sharing what tool you used to get this image, is it nsight system?

![image](https://github.com/user-attachments/assets/d35f7511-da3c-41fd-b0fe-30034136d468)

---
### peng1999 @ 2024-07-26T02:06:13Z

> Great PR! Would you mind sharing what tool you used to get this image, is it nsight system?

Yes. The blue spans are recorded using NVTX.
---