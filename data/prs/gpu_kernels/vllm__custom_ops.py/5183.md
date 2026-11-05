# PR #5183: [Kernel] Switch fp8 layers to use the CUTLASS kernels

- field: gpu_kernels  | area: vllm/_custom_ops.py
- author: tlrmchlsmth
- state: MERGED  created: 2024-06-01T18:55:56Z  merged: 2024-06-07T08:42:35Z
- url: https://github.com/vllm-project/vllm/pull/5183
- labels: 

## Body

Switching from `torch._scaled_mm` to vLLM's cutlass fp8 kernels when supported as we are seeing 5-15% improvement in e2e performance on `neuralmagic/Meta-Llama-3-8B-Instruct-FP8`

see https://docs.google.com/spreadsheets/d/1GiAnmzyGHgZ6zL_LDSTm35Bdrt4A8AaFEurDlISYYA4/ for some quick e2e benchmarks and https://github.com/vllm-project/vllm/pull/5144 for comparisons across different GEMM sizes.

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

### robertgshaw2-redhat @ 2024-06-01T18:58:46Z

@tlrmchlsmth models:

- https://huggingface.co/nm-testing/Meta-Llama-3-70B-Instruct-FP8
- https://huggingface.co/neuralmagic/Meta-Llama-3-8B-Instruct-FP8
- https://huggingface.co/nm-testing/Meta-Llama-3-8B-Instruct-FP8-KV << with Quantized KV Cache
---
### tlrmchlsmth @ 2024-06-01T19:20:51Z

Just ran a quick sanity check for correctness. Output looks good on all three. I tried tensor_parallel_size=2 as well for the 70B model, and that looks good
---
### pcmoritz @ 2024-06-01T20:04:49Z

Did you run benchmarks to compare the end-to-end performance? ITL for different qps
---
### robertgshaw2-redhat @ 2024-06-01T20:06:47Z

> Did you run benchmarks to compare the end-to-end performance? ITL for different qps

Not yet. But obviously need this before we merge
---
### comaniac @ 2024-06-01T20:34:30Z

I'll do a benchmark on Monday anyways. btw it'd be great if this PR is rebased onto the latest main that includes all required changes (it's likely the case already I suppose
---
### tlrmchlsmth @ 2024-06-01T21:04:15Z

>btw it'd be great if this PR is rebased onto the latest main that includes all required changes (it's likely the case already I suppose

It's on a very recent main (from this morning) so it's good to use as is. In particular both #5144 and #5137 were needed for the switchover and they are both in.
---
### comaniac @ 2024-06-06T16:47:53Z

@tlrmchlsmth @robertgshaw2-neuralmagic per offline discussion, this PR should be ok to go at least for now?
---
### tlrmchlsmth @ 2024-06-06T17:00:19Z

Yeah, let's get it landed. It needs to check a few more cases for falling back to scaled_mm. I'll get to that today and then mark it ready for review 
---
### cli99 @ 2024-06-07T20:13:43Z

@tlrmchlsmth Awesome work! Was trying this but ran into a problem when checking the cutlass fp8 support. Made a fix that works in my case in https://github.com/vllm-project/vllm/pull/5352. 
---