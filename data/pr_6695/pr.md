# PR #6695: PP comm optimization: replace send with partial send + allgather

- field: parallelism  | area: distributed/parallel_state.py
- author: aurickq
- state: MERGED  created: 2024-07-23T16:27:43Z  merged: 2024-08-01T03:15:42Z
- url: https://github.com/vllm-project/vllm/pull/6695
- labels: ready

## Body

Submitting on behalf of @zhisbug as he is on break. Communication optimization for pipeline parallelism, we observed 5% improvement in throughput for llama 3.1 405b.

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

### github-actions @ 2024-07-23T16:27:54Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### andoorve @ 2024-07-23T17:05:44Z

LGTM from a first pass. Thanks for the contribution!
---
### aurickq @ 2024-07-24T18:24:03Z

> Thanks for the PR! I have few questions:
> 
> 1. Do you have benchmark results?
> 2. At the first glance it seems always better to enable this mechanism. Have you observed performance regression with this PR in certain circumstances? Like when the tensor size is small the allgather overhead is not negligible (just guess)?

Re: benchmark results, in our tests using llama 3.1 405b (yellow=before, gray=after):
<img width="1471" alt="Screenshot 2024-07-24 at 11 21 47 AM" src="https://github.com/user-attachments/assets/518d7c19-628c-4bd2-9500-293ff0c2105c">

We haven't observed any performance regressions yet but also haven't tested very many scenarios.
---
### andoorve @ 2024-07-29T18:04:35Z

Hey! @youkaichao any blockers for merge here?
---
### zhisbug @ 2024-07-29T20:16:04Z

This one should have better improvement when your inter-node bandwidth is not high.

---
### andoorve @ 2024-07-29T20:35:31Z

> Does it make sense to add an integration test? vLLM does have some 4 GPU CI runs in buildkite, so it should be possible to add one. Otherwise this code won't be exercised and may break.

+1. Should be just a matter of adding a parameter to existing pipeline tests
---
### youkaichao @ 2024-07-31T04:08:00Z

Sorry for the delay, just notice this PR and the message.

The idea looks good to me, we use `send --> allgather` to replace `send` full tensor on all ranks.

Implementation wise, can we use this by default? As long as `tensor.numel() % tp_size == 0`, you can use it. `send_tensor_dict` is only called in pipeline parallel. And you don't need to create a full suit of `_ENABLE_ALLGATHER_PIPELINE_COMM` stuff. All you need, should be adding a `all_gather_group` argument to `send_tensor_dict`:

https://github.com/vllm-project/vllm/blob/9f0e69b65350fad1d4a9c71ef58d6ae70eb635e8/vllm/distributed/parallel_state.py#L569

```diff
    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
+     all_gather_group=Optional[GroupCoordinator],
    )
...

            if tensor.is_cpu:
                # use metadata_group for CPU tensors
-                torch.distributed.send(tensor,
+                torch.distributed.send(tensor[slice],
                                       dst=self.ranks[dst],
                                       group=metadata_group)
+                # add allgather with `all_gather_group.cpu_group`
            else:
                # use group for GPU tensors
-                torch.distributed.send(tensor,
+                torch.distributed.send(tensor[slice],
                                       dst=self.ranks[dst],
                                       group=group)
+                # add allgather with `all_gather_group.device_group`

# change `recv_tensor_dict` accordingly
```

And update this line 

https://github.com/vllm-project/vllm/blob/9f0e69b65350fad1d4a9c71ef58d6ae70eb635e8/vllm/worker/worker_base.py#L279

to `get_pp_group().send_tensor_dict(output.tensors, all_gather_group=get_tp_group())`
---
### aurickq @ 2024-08-01T01:03:55Z

> Sorry for the delay, just notice this PR and the message.
> 
> The idea looks good to me, we use `send --> allgather` to replace `send` full tensor on all ranks.
> 
> Implementation wise, can we use this by default? As long as `tensor.numel() % tp_size == 0`, you can use it. `send_tensor_dict` is only called in pipeline parallel. And you don't need to create a full suit of `_ENABLE_ALLGATHER_PIPELINE_COMM` stuff. All you need, should be adding a `all_gather_group` argument to `send_tensor_dict`:
> 
> https://github.com/vllm-project/vllm/blob/9f0e69b65350fad1d4a9c71ef58d6ae70eb635e8/vllm/distributed/parallel_state.py#L569
> 
> ```diff
>     def send_tensor_dict(
>         self,
>         tensor_dict: Dict[str, Union[torch.Tensor, Any]],
>         dst: Optional[int] = None,
> +     all_gather_group=Optional[GroupCoordinator],
>     )
> ...
> 
>             if tensor.is_cpu:
>                 # use metadata_group for CPU tensors
> -                torch.distributed.send(tensor,
> +                torch.distributed.send(tensor[slice],
>                                        dst=self.ranks[dst],
>                                        group=metadata_group)
> +                # add allgather with `all_gather_group.cpu_group`
>             else:
>                 # use group for GPU tensors
> -                torch.distributed.send(tensor,
> +                torch.distributed.send(tensor[slice],
>                                        dst=self.ranks[dst],
>                                        group=group)
> +                # add allgather with `all_gather_group.device_group`
> 
> # change `recv_tensor_dict` accordingly
> ```
> 
> And update this line
> 
> https://github.com/vllm-project/vllm/blob/9f0e69b65350fad1d4a9c71ef58d6ae70eb635e8/vllm/worker/worker_base.py#L279
> 
> to `get_pp_group().send_tensor_dict(output.tensors, all_gather_group=get_tp_group())`

Done and tested
---
### youkaichao @ 2024-08-01T03:15:29Z

merge as failing tests are unrelated.
---