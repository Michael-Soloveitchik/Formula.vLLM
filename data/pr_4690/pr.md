# PR #4690: [Core] Implement sharded state loader

- field: scheduling  | area: examples
- author: aurickq
- state: MERGED  created: 2024-05-08T19:24:47Z  merged: 2024-05-16T05:11:55Z
- url: https://github.com/vllm-project/vllm/pull/4690
- labels: 

## Body

This PR implements a new model loader that directly loads the sharded states of each worker when using `DistributedGPUExecutor`. When using tensor parallelism, this avoids each worker reading the full checkpoint just to load a small shard of it. Our tests using Arctic (https://github.com/vllm-project/vllm/pull/4652) showed a 10x improvement in model loading speed from NVMe when using 8x tensor parallelism.

For quantization methods like DeepSpeed's FP_Quantize (also used in https://github.com/vllm-project/vllm/pull/4652) that quantize after loading, this PR allows easy creation of a quantized checkpoint that is directly loaded into each worker, further speeding up model loading.

This PR is separated out from https://github.com/vllm-project/vllm/pull/4652.

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

### Yard1 @ 2024-05-09T00:29:47Z

We should add a test to ensure it's working correctly (loaded weights and outputs are the same as with the default loader)
---
### aurickq @ 2024-05-09T15:19:34Z

> We should add a test to ensure it's working correctly (loaded weights and outputs are the same as with the default loader)

Sure, that makes sense. What's the spec of the test runner machine we should target? OK to assume cuda device is present?
---
### Yard1 @ 2024-05-09T17:05:07Z

@aurickq IIRC we are using L4 GPUs in CI, @simon-mo to confirm
---
### aurickq @ 2024-05-09T22:45:11Z

@Yard1 added test, please take a look
---
### simon-mo @ 2024-05-10T00:32:15Z

One question I have is that can this be implemented using safetensor's partial read? safetensors have all the metadata in headers so you can access the tensors partially
---
### aurickq @ 2024-05-10T02:57:42Z

> One question I have is that can this be implemented using safetensor's partial read? safetensors have all the metadata in headers so you can access the tensors partially

Conceptually, I think so. Though for larger models like Arctic we prefer this implementation for a few reasons:

1. With the current implementation and multi-node inference, we may download only the required files to each node.
2. The deepspeedfp quantizer will reshape the parameters and may change its number of elements, which makes it a bit tricky to calculate the right slice to load (need to understand internals of the quantizer to slice the quantized weights).
---
### njhill @ 2024-05-10T15:05:47Z

@aurickq curious how this relates to https://github.com/vllm-project/vllm/pull/3729?
---
### aurickq @ 2024-05-10T15:45:42Z

> @aurickq curious how this relates to #3729?

Ah, I hadn't seen that PR, thanks for bringing it up! From what I understand, both PRs address the problem of model loading speed, particularly for larger models, but the approaches seem pretty different. #3729 modifies the existing model loading path so each worker loads a different set of tensors, then broadcasts the shards to each other. This PR loads the model once using the default path and dumps each worker state to disk, then subsequent model loads can direct read these states, skipping the default model loading path altogether.

Some first thoughts on the pros/cons:

- This PR requires the creation of a new checkpoint (the worker states), and this checkpoint is only compatible with a fixed configuration (parallelism, quantization, etc.). Whereas #3729 seems more flexible in this regard.
- #3729 seems to require a lot more changes in vllm and in each model definition, whereas this PR only requires a handful of lines of change in vllm, and no changes in any of the models (so any future model is automatically supported).
- #3729 seems to be more coupled with the parallelism implementation (e.g. tensor parallel in this case). However we are currently implementing pipeline-parallelism to support multi-node inference, which also will be automatically supported by this PR.
---