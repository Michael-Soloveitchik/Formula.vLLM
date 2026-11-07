# PR #28284: [Feature] Support recording expert indices for rollout router replay

- field: scheduling  | area: vllm/config
- author: xhx1022
- state: MERGED  created: 2025-11-07T09:46:12Z  merged: 2026-01-12T14:23:04Z
- url: https://github.com/vllm-project/vllm/pull/28284
- labels: performance, frontend, ready, v1, rl

## Body

## Purpose

This PR introduces **Rollout Router Replay (R3)** support into vLLM runtime.  
Inspired by the recent research in **reinforcement learning alignment for MoE-based LLMs** ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370), [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)), this implementation allows recording the **expert routing decisions for every token at every layer** during model inference. The recorded routing traces can be used for replaying the expert routing process during RL post traning;

Currently, the initial version supports:
- ✅ Tensor Parallel mode;
- ✅  Prefix Cache mode;
- ✅ Compatibility with CUDA Graph execution;

Below is a minimal reproducible example for running Qwen3-30B-A3B with **tensor_parallel_size = 8** and **async concurrent inference**.
The example also prints the shape of the returned `routed_experts` tensor.

```python
import asyncio

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

# Configure engine args with TP=8
ENGINE_ARGS = AsyncEngineArgs(
    model="/dev/shm/Qwen3-30B-A3B",
    tensor_parallel_size=8,
    enable_return_routed_experts=True,
    # enforce_eager=True,
)

async def test_qwen3_tp8_concurrent():
    """Test concurrent inference of Qwen3-30B-A3B with TP=8"""

    # Create AsyncLLM instance
    async_llm = AsyncLLM.from_engine_args(ENGINE_ARGS)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )

    NUM_REQUESTS = 10
    prompts = [
        f"Request {i}: Hello, please introduce yourself."
        for i in range(NUM_REQUESTS)
    ]

    async def generate_single(request_id: str, prompt: str):
        outputs = []
        async for output in async_llm.generate(
            request_id=request_id,
            prompt=prompt,
            sampling_params=sampling_params
        ):
            outputs.append(output)
        return outputs[-1]  # Return the final output

    # Create concurrent tasks
    tasks = [
        generate_single(f"request-{i}", prompts[i])
        for i in range(NUM_REQUESTS)
    ]

    # Wait for all tasks to finish
    results = await asyncio.gather(*tasks)

    # Validate results
    for i, result in enumerate(results):
        assert result.finished
        assert len(result.outputs[0].text) > 0
        print(result.outputs[0].routed_experts.shape)  # [seq_len, layer_num, topk]

    # Cleanup
    async_llm.shutdown()

# Run test
if __name__ == "__main__":
    asyncio.run(test_qwen3_tp8_concurrent())
```

## Reminder

The number of experts in the output can be 1 less than (prompt_length + response_token_count).
This gap of 1 is expected because the final generated token is sampled, not computed through the forward layer, and therefore is not included in the expert count.

## Acknowledgments
This work is inspired by and builds upon the implementation from SGLang PR [#12162](https://github.com/sgl-project/sglang/pull/12162).  
Special thanks to **@ocss884** and the **SGLang RL team** for their valuable discussions and contributions.

---

> [!NOTE]
> Introduces an opt-in pathway to capture and return MoE routed expert indices per token/layer.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, plumbed through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - Implements `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared memory buffers; `fused_moe/layer.py` records gate `topk_ids` per layer
> - `GPUModelRunner` initializes/clears/saves captured experts and computes `slot_mapping`; TP rank 0 writes to shared memory
> - `Scheduler` derives token slots from KV blocks and attaches `routed_experts` to `EngineCoreOutput`
> - Output pipeline propagates to clients: `CompletionOutput.routed_experts` (numpy array) and related plumbing in v1 engine/output processor
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit b0fb649926346a1a132d8c8dd294a5a95142579f. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces optional routed-expert tracing for MoE models, plumbed end-to-end and exposed in request outputs.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig` and surfaces it via `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - Implements `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared memory buffers; `fused_moe/layer.py` records gate `topk_ids` per layer using `layer_id`
> - `GPUModelRunner` initializes/clears/saves captured experts and computes `slot_mapping`; TP rank 0 writes captured indices to shared memory
> - `Scheduler` reconstructs token slots from KV blocks and attaches `routed_experts` to `EngineCoreOutput`
> - Output pipeline propagates to clients: adds `CompletionOutput.routed_experts` (numpy array) and corresponding plumbing in v1 engine/output processor
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 407fd57d30b3fe321fcbb75bea382b17fa89f349. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables recording and returning MoE routed expert indices for each token/layer when `enable_return_routed_experts` is set.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`; plumbed through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - Implements `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared memory buffers; `fused_moe/layer.py` captures gate `topk_ids` using `layer_id`
> - `GPUModelRunner` initializes/clears/saves captured experts and computes `slot_mapping`; TP rank 0 writes captured indices to shared memory
> - `Scheduler` reconstructs token slots from KV blocks and attaches `routed_experts` to engine outputs
> - Propagates to clients by adding `CompletionOutput.routed_experts` (numpy array) and wiring through v1 engine/output processor
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit ec8ed03f325943d2c63b329e36f018123f91109d. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables optional routed-expert tracing for MoE models and surfaces it in request outputs.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, threads through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - Implements `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared memory buffers; `fused_moe/layer.py` captures gate `topk_ids` using `layer_id`
> - `GPUModelRunner` initializes/clears/saves captured experts and computes `slot_mapping`; TP rank 0 writes captured indices to shared memory
> - `Scheduler` reconstructs token slot indices from KV blocks and attaches `routed_experts` to engine outputs
> - Output pipeline propagates to clients by adding `CompletionOutput.routed_experts` and wiring through v1 engine/output processor
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit ec8ed03f325943d2c63b329e36f018123f91109d. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit ec8ed03f325943d2c63b329e36f018123f91109d. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables tracing MoE router decisions end-to-end when `enable_return_routed_experts` is set, and exposes them to clients.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, plumbed through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - Implements `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared-memory buffers; TP rank 0 writes captured indices
> - Hooks in `fused_moe/layer.py` to capture gate `topk_ids` per `layer_id`
> - `GPUModelRunner` initializes/clears/saves captured experts and computes `slot_mapping`
> - `Scheduler` reconstructs token slot indices from KV blocks and attaches `routed_experts` to `EngineCoreOutput`
> - Propagates to clients by adding `CompletionOutput.routed_experts` (numpy array) and wiring through v1 engine/output processor
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit ec8ed03f325943d2c63b329e36f018123f91109d. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables optional routed-expert tracing for MoE models and surfaces it in request outputs.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`; plumbed through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - New `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared-memory buffers to store/read per-token per-layer router `topk_ids`; TP rank 0 writes, scheduler reads
> - Hooks `fused_moe/layer.py` to capture gate `topk_ids` with `layer_id`; `GPUModelRunner` initializes/clears/saves captures and computes `slot_mapping`
> - `Scheduler` derives token slot indices from KV blocks on request finish and attaches `routed_experts` to engine outputs
> - Propagates to clients by adding `CompletionOutput.routed_experts` (numpy array) and wiring through v1 engine/output processor/engine structures
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit c9d5d3b2729284422782eb1dccd1bc0668ab111c. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit 90ebad2bf212397ec26f7939e47f11c2139fa9e9. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces an opt-in pathway to capture and return MoE router `topk_ids` per token/layer and expose them to clients.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, surfaced via `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - New `RoutedExpertsCapturer`/`RoutedExpertsReader` with shared memory buffers to store/read per-token per-layer routed experts; TP rank 0 writes, scheduler reads
> - Hooks `fused_moe/layer.py` to capture gate `topk_ids` using `layer_id`; `GPUModelRunner` initializes/clears/saves captures and computes `slot_mapping`
> - `Scheduler` reconstructs token slot indices from KV blocks on request finish and attaches `routed_experts` (numpy array, shape `[seq_len, layer_num, topk]`) to outputs
> - Plumbs through output pipeline: adds `CompletionOutput.routed_experts` and wires via v1 engine/output processor
> - Disables async scheduling when `enable_return_routed_experts=True` and asserts no context parallelism (DCP/PCP > 1 unsupported)
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 90ebad2bf212397ec26f7939e47f11c2139fa9e9. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces end-to-end, optional routed-expert tracing for MoE models and exposes it in outputs.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, plumbed via `EngineArgs`/CLI (`--enable-return-routed-experts`) and `LLM` entrypoint; included in config logs
> - Implements `fused_moe/routed_experts_capturer.py` with shared-memory buffers and a reader/writer; `fused_moe/layer.py` hooks capture per-layer `topk_ids` using `layer_id`
> - `GPUModelRunner` initializes/clears/saves captures, computes `slot_mapping`, and writes from TP rank 0
> - `Scheduler` reconstructs token slot indices from KV blocks and attaches `routed_experts` (numpy array) to engine outputs; asserts no context parallelism
> - Surfaces to clients by adding `CompletionOutput.routed_experts` and threading through v1 engine/output processor
> - Disables async scheduling when `enable_return_routed_experts=True`
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 39aefda113994386d32634051e9183c618f25681. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables optional end-to-end capture and return of MoE routed expert indices.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, surfaced via `EngineArgs`/CLI (`--enable-return-routed-experts`) and `LLM`; included in config logs
> - New `fused_moe/routed_experts_capturer.py` (capturer/reader with shared memory) and hooks in `fused_moe/layer.py` to record per-layer router `topk_ids` (`layer_id`)
> - Integrates in `GPUModelRunner`: initialize/clear/save captures, compute `slot_mapping`, TP rank 0 writes to shared memory
> - `Scheduler`: attaches reader, reconstructs token slots from KV blocks on request finish, and attaches `routed_experts` to engine outputs; asserts no DCP/PCP; disables async scheduling when flag is set
> - Output plumbing: adds `CompletionOutput.routed_experts` (numpy array) and threads through v1 engine/output processor/structures
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 819964f71ede807917c1ab2d63fb9c0232fbbd5e. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit 21c26d3001b2bbf73e8704f086b37061355f3907. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces end-to-end, opt-in routed-expert tracing for MoE models and surfaces it to clients.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, threaded through `EngineArgs`/CLI (`--enable-return-routed-experts`), `LLM` entrypoint, and config logging
> - New `fused_moe/routed_experts_capturer.py` implementing capturer/reader singletons with shared memory buffers; TP rank 0 writes captured indices
> - Hooks `fused_moe/layer.py` to record gate `topk_ids` per `layer_id`
> - `GPUModelRunner`: initializes/clears/saves captures and computes `slot_mapping`; writes per-token routed experts to shared memory
> - `Scheduler`: attaches reader, reconstructs token slots from KV blocks on request finish, asserts no context parallelism, and attaches `routed_experts` to engine outputs
> - Output plumbing: adds `CompletionOutput.routed_experts` (numpy array, shape `[seq_len,layer_num,topk]`) and wires through v1 engine/output processor/structures
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 9fd9ac2ad7004208b263a6d16a412a74ea3317cc. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables returning MoE routed expert indices when requested, plumbed across config → execution → outputs.
> 
> - Adds `enable_return_routed_experts` to `ModelConfig`, surfaced via `EngineArgs`/CLI and `LLM`; included in config logs
> - New `fused_moe/routed_experts_capturer.py` (capturer/reader singletons using shared memory); `fused_moe/layer.py` records gate `topk_ids` per `layer_id`
> - `GPUModelRunner` initializes/clears/saves captured experts and computes token `slot_mapping` (TP rank 0 writes)
> - `Scheduler` reconstructs token slots from KV blocks and attaches `routed_experts` to engine outputs; asserts no context parallelism
> - Output pipeline: adds `CompletionOutput.routed_experts` (numpy array) and threads through v1 engine/output processor/structures
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit aeb469eea2a2b110af7d5d0e53bb7a290abba5a5. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

### chatgpt-codex-connector @ 2025-11-07T09:49:59Z


### 💡 Codex Review

https://github.com/vllm-project/vllm/blob/611bc69292546334ddbcc52689ffe86f91da41e1/vllm/v1/worker/gpu_model_runner.py#L2737-L2738
**<sub><sub>![P1 Badge](https://img.shields.io/badge/P1-orange?style=flat)</sub></sub>  Convert slot_mapping to numpy before saving experts**

When routed expert recording is enabled, `GPUModelRunner.execute_model` passes `self.slot_mapping` directly into `RoutedExpertsCapturer.save_captured_experts` (`save_captured_experts(indices=self.slot_mapping)`). `self.slot_mapping` is a CPU `torch.Tensor` (it is assigned as `slot_mapping.cpu()` earlier in the method), but `_RoutedExpertsCapturerReal.save_captured_experts` indexes a NumPy buffer with the `indices` argument (`self._host_buffer_view[indices, :, :] = data`). NumPy does not accept PyTorch tensors as advanced indices, so this call will raise a `TypeError: only integer scalar arrays can be converted to a scalar index` the first time the feature is exercised. Converting the tensor to a NumPy array (e.g. `indices.numpy()`) before the call would avoid the crash.
    

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

Codex has been enabled to automatically review pull requests in this repo. Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




When you [sign up for Codex through ChatGPT](https://openai.com/codex), Codex can also answer questions or update the PR, like "@codex address that feedback".
            
</details>
---
### github-actions @ 2025-11-07T09:55:12Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### heheda12345 @ 2025-11-11T07:48:24Z

CC @youkaichao @22quinn
---
### mergify @ 2025-11-11T12:53:20Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @xhx1022.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### litianjian @ 2025-11-14T08:25:33Z

Will enabling prefix cache have any impact?
---
### xhx1022 @ 2025-11-14T13:28:28Z

> Will enabling prefix cache have any impact?

@litianjian There will be no impact. My implementation manages expert IDs in the same way as KV. As long as the KV for a given token exists, the corresponding expert ID can be retrieved.
---
### david6666666 @ 2025-11-18T07:15:13Z

Interesting pr, any update? this pr and related pr #https://github.com/volcengine/verl/pull/4101
---
### xhx1022 @ 2025-11-18T11:11:37Z

> Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)

@david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
---
### litianjian @ 2025-11-18T11:20:14Z

> > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> 
> @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.

@xhx1022  I will test R3 in verl with this pr.
---
### xhx1022 @ 2025-11-18T12:35:56Z

> > > Interesting pr, any update? this pr and related pr [volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > 
> > 
> > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> 
> @xhx1022 I will test R3 in verl with this pr.

@litianjian ok, May I ask what kind of data format you need vLLM to return?
---
### mergify @ 2025-11-21T16:27:23Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @xhx1022.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### litianjian @ 2025-11-24T08:43:38Z

> > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > 
> > 
> > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> 
> @xhx1022 I will test R3 in verl with this pr.

@xhx1022  In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.
---
### xhx1022 @ 2025-11-24T09:30:12Z

> > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > 
> > > 
> > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > 
> > 
> > @xhx1022 I will test R3 in verl with this pr.
> 
> @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.

OK, I’ll fix this bug. I previously tested vLLM in async mode, and it was able to run correctly. Have you tested it in async mode as well?
---
### xhx1022 @ 2025-11-25T12:19:01Z

> > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > 
> > > 
> > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > 
> > 
> > @xhx1022 I will test R3 in verl with this pr.
> 
> @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.

@litianjian fixed
---
### litianjian @ 2025-11-27T08:08:48Z

> > > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > > 
> > > > 
> > > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > > 
> > > 
> > > @xhx1022 I will test R3 in verl with this pr.
> > 
> > 
> > @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.
> 
> @litianjian fixed

should capture and reader be enable only in rank 0 (tp group)?
---
### mergify @ 2025-12-05T13:05:22Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.
---
### faresobeid @ 2025-12-09T19:22:33Z

Great work, would be great to push this in priority given its increasing importance. Would also be great if we can return the router probabilities for the chosen experts to be able to support something like https://arxiv.org/abs/2510.23027
---
### scut-zx @ 2025-12-10T03:12:59Z

Thank you for this great work. I try to reproduce this project, but I meet some environment issues. My CUDA version is 12.1, but the project is based on vLLM 0.11.2, whose official Docker requires CUDA 12.9. When I download the code and try to build it locally inside a vLLM 0.10.0 Docker, the compilation fails. Have any suggestions for how to solve the issues?
---
### xhx1022 @ 2025-12-10T05:31:08Z

> > > > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > > > 
> > > > > 
> > > > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > > > 
> > > > 
> > > > @xhx1022 I will test R3 in verl with this pr.
> > > 
> > > 
> > > @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.
> > 
> > 
> > @litianjian fixed
> 
> should capture and reader be enable only in rank 0 (tp group)?

Capture runs on every GPU, but reader runs only on rank 0.

---
### xhx1022 @ 2025-12-10T05:35:19Z

> Great work, would be great to push this in priority given its increasing importance. Would also be great if we can return the router probabilities for the chosen experts to be able to support something like https://arxiv.org/abs/2510.23027

We will consider this depending on industry demand, but there are currently no plans to implement it.
---
### litianjian @ 2025-12-10T05:35:32Z

> > > > > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > > > > 
> > > > > > 
> > > > > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > > > > 
> > > > > 
> > > > > @xhx1022 I will test R3 in verl with this pr.
> > > > 
> > > > 
> > > > @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.
> > > 
> > > 
> > > @litianjian fixed
> > 
> > 
> > should capture and reader be enable only in rank 0 (tp group)?
> > Capture runs on every GPU, but reader runs only on rank 0.

It would be better to capture only once.
---
### xhx1022 @ 2025-12-10T05:48:36Z

> > > > > > > > Interesting pr, any update? this pr and related pr #[volcengine/verl#4101](https://github.com/volcengine/verl/pull/4101)
> > > > > > > 
> > > > > > > 
> > > > > > > @david6666666 The overall implementation is basically complete. There might occasionally be some bugs that will be updated as they come up. Currently, the returned result has the shape [seq_len, layer_num, topk]. We may need to check with the verl team to confirm the exact format of the returned data they require.
> > > > > > 
> > > > > > 
> > > > > > @xhx1022 I will test R3 in verl with this pr.
> > > > > 
> > > > > 
> > > > > @xhx1022 In verl （vllm spmd）， each process in one TP group has an independent instance ID, which leads to shmem initialization errors.
> > > > 
> > > > 
> > > > @litianjian fixed
> > > 
> > > 
> > > should capture and reader be enable only in rank 0 (tp group)?
> > > Capture runs on every GPU, but reader runs only on rank 0.
> 
> It would be better to capture only once.

If do that, CUDA Graph would no longer be compatible. Additionally, they run in parallel across GPUs, so this does not introduce any extra overhead.

---
### Kairosxy @ 2025-12-11T11:55:55Z

Does this PR support VLM?
---
### awcvec @ 2025-12-14T13:09:54Z

It looks like the current implementation only supports a single KV cache group (len(self.kv_cache_config.kv_cache_groups) == 1). Does it also support hybrid attention (global + SWA)?
---
### stylke @ 2025-12-14T13:37:41Z

Hi, I tried R3 in verl async mode (vllm spmd)，and found that in `Scheduler.__init__`, `vllm_config.parallel_config.rank` is 0 (thus the `instance_id` is 0) in every dp replica. Consequently, readers with `replica_rank` > 0 will attach to the buffer initialized by the capturer with `replica_rank` = 0, which is wrong or uninitialized.

Currently I modified `vllm_config.parallel_config.rank` in `vLLMHttpServerBase.run_server` in verl, and it seems that the problem is solved. 

Could you please verify if my understanding is correct, or suggest a better way to solve this?
---
### xhx1022 @ 2025-12-15T07:01:52Z

> Does this PR support VLM?

Yes, I believe it does support VLM. This PR does not affect the ViT module or the vision-related components.

---
### mergify @ 2025-12-15T10:31:12Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### xhx1022 @ 2025-12-16T02:10:37Z

> It looks like the current implementation only supports a single KV cache group (len(self.kv_cache_config.kv_cache_groups) == 1). Does it also support hybrid attention (global + SWA)?

At the moment, hybrid attention (global + SWA) is not supported yet. I’ll take a look later to see how it could be adapted or extended to support this use case.
---
### xhx1022 @ 2025-12-16T02:12:10Z

> Hi, I tried R3 in verl async mode (vllm spmd)，and found that in `Scheduler.__init__`, `vllm_config.parallel_config.rank` is 0 (thus the `instance_id` is 0) in every dp replica. Consequently, readers with `replica_rank` > 0 will attach to the buffer initialized by the capturer with `replica_rank` = 0, which is wrong or uninitialized.
> 
> Currently I modified `vllm_config.parallel_config.rank` in `vLLMHttpServerBase.run_server` in verl, and it seems that the problem is solved.
> 
> Could you please verify if my understanding is correct, or suggest a better way to solve this?

Thanks for reporting this. I haven’t encountered this issue on my side yet. Could you please share a bit more detail about your setup and reproduction steps？
---
### stylke @ 2025-12-16T03:01:36Z

> > Hi, I tried R3 in verl async mode (vllm spmd)，and found that in `Scheduler.__init__`, `vllm_config.parallel_config.rank` is 0 (thus the `instance_id` is 0) in every dp replica. Consequently, readers with `replica_rank` > 0 will attach to the buffer initialized by the capturer with `replica_rank` = 0, which is wrong or uninitialized.
> > Currently I modified `vllm_config.parallel_config.rank` in `vLLMHttpServerBase.run_server` in verl, and it seems that the problem is solved.
> > Could you please verify if my understanding is correct, or suggest a better way to solve this?
> 
> Thanks for reporting this. I haven’t encountered this issue on my side yet. Could you please share a bit more detail about your setup and reproduction steps？

I am using verl at commit 38246890efb50e60d2471ac2518cb512ba8361ba (Dec 12), and vLLM from the branch `releases/v0.12.0`, with this R3 patch applied.

My cluster configuration consists of 2 nodes, each with 8 GPUs. The vLLM tensor parallelism (tp) size is 4. In theory, I expected there should be 4 `Scheduler` instances, corresponding to `vllm_config.parallel_config.rank` values of 0, 4, 8, 12 (and `instance_id` values of 0, 1, 2,  3, respectively). However, I found that all 4 Schedulers have a `vllm_config.parallel_config.rank` value of 0.

My specific launch script is as follows:
```
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping

# 0. download the model

# 1. convert the model to mcore format
# change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path
HF_MODEL_PATH=<path>/Qwen3-30B-A3B
DIST_CKPT_PATH=<path>/data/ckpt/mcore_ckpt/Qwen3-30B-A3B
python <path>/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

# 2. run the script
train_files=<path>/data/retool_dapo/train.parquet
test_files=<path>/data/retool_aime2024/train.parquet

NODES=2
PP=2
TP=4
CP=1
EP=4
ETP=1
VLLM_TP=4

offload=True
optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}

ROUTING_REPLAY_MODE="R3" 


# RAY_ADDRESS='auto' ray job submit --working-dir . -- 
python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=768 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.router_replay.mode=${ROUTING_REPLAY_MODE} \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
    actor_rollout_ref.ref.megatron.context_parallel_size=$CP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    critic.optim.lr=1e-5 \
    critic.model.path=$HF_MODEL_PATH \
    critic.ppo_mini_batch_size=256 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.megatron.tensor_model_parallel_size=$TP \
    critic.megatron.pipeline_model_parallel_size=$PP \
    critic.megatron.context_parallel_size=$CP \
    critic.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    critic.megatron.use_dist_checkpointing=True \
    critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='Qwen3-30B-A3B' \
    trainer.experiment_name='ppo-verl-r3' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NODES \
    trainer.total_training_steps=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 2>&1 | tee verl_megatron_r3.log
    
```
---
### lkm2835 @ 2025-12-16T04:33:39Z

> > It looks like the current implementation only supports a single KV cache group (len(self.kv_cache_config.kv_cache_groups) == 1). Does it also support hybrid attention (global + SWA)?
> 
> At the moment, hybrid attention (global + SWA) is not supported yet. I’ll take a look later to see how it could be adapted or extended to support this use case.

Is hybrid attention supported when `--disable_hybrid_kv_cache_manager` is enabled?
---
### xhx1022 @ 2025-12-16T07:40:08Z

> > > Hi, I tried R3 in verl async mode (vllm spmd)，and found that in `Scheduler.__init__`, `vllm_config.parallel_config.rank` is 0 (thus the `instance_id` is 0) in every dp replica. Consequently, readers with `replica_rank` > 0 will attach to the buffer initialized by the capturer with `replica_rank` = 0, which is wrong or uninitialized.
> > > Currently I modified `vllm_config.parallel_config.rank` in `vLLMHttpServerBase.run_server` in verl, and it seems that the problem is solved.
> > > Could you please verify if my understanding is correct, or suggest a better way to solve this?
> > 
> > 
> > Thanks for reporting this. I haven’t encountered this issue on my side yet. Could you please share a bit more detail about your setup and reproduction steps？
> 
> I am using verl at commit 38246890efb50e60d2471ac2518cb512ba8361ba (Dec 12), and vLLM from the branch `releases/v0.12.0`, with this R3 patch applied.
> 
> My cluster configuration consists of 2 nodes, each with 8 GPUs. The vLLM tensor parallelism (tp) size is 4. In theory, I expected there should be 4 `Scheduler` instances, corresponding to `vllm_config.parallel_config.rank` values of 0, 4, 8, 12 (and `instance_id` values of 0, 1, 2, 3, respectively). However, I found that all 4 Schedulers have a `vllm_config.parallel_config.rank` value of 0.
> 
> My specific launch script is as follows:
> 
> ```
> set -x
> 
> export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
> 
> # 0. download the model
> 
> # 1. convert the model to mcore format
> # change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path
> HF_MODEL_PATH=<path>/Qwen3-30B-A3B
> DIST_CKPT_PATH=<path>/data/ckpt/mcore_ckpt/Qwen3-30B-A3B
> python <path>/scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH
> 
> # 2. run the script
> train_files=<path>/data/retool_dapo/train.parquet
> test_files=<path>/data/retool_aime2024/train.parquet
> 
> NODES=2
> PP=2
> TP=4
> CP=1
> EP=4
> ETP=1
> VLLM_TP=4
> 
> offload=True
> optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.}
> 
> ROUTING_REPLAY_MODE="R3" 
> 
> 
> # RAY_ADDRESS='auto' ray job submit --working-dir . -- 
> python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer'\
>     algorithm.adv_estimator=gae \
>     data.train_files="$train_files" \
>     data.val_files="$test_files" \
>     data.train_batch_size=768 \
>     data.max_prompt_length=2048 \
>     data.max_response_length=8192 \
>     data.filter_overlong_prompts=True \
>     data.truncation='error' \
>     +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
>     +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
>     +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
>     +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
>     actor_rollout_ref.model.path=$HF_MODEL_PATH \
>     actor_rollout_ref.model.use_fused_kernels=True \
>     actor_rollout_ref.actor.optim.lr=1e-6 \
>     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
>     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
>     actor_rollout_ref.actor.use_kl_loss=False \
>     actor_rollout_ref.actor.router_replay.mode=${ROUTING_REPLAY_MODE} \
>     actor_rollout_ref.rollout.enable_rollout_routing_replay=True \
>     actor_rollout_ref.actor.megatron.param_offload=${offload} \
>     actor_rollout_ref.actor.megatron.grad_offload=${offload} \
>     actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
>     actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TP \
>     actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$PP \
>     actor_rollout_ref.actor.megatron.context_parallel_size=$CP \
>     actor_rollout_ref.actor.megatron.expert_model_parallel_size=$EP \
>     actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=$ETP \
>     actor_rollout_ref.ref.megatron.param_offload=${offload} \
>     actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TP \
>     actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$PP \
>     actor_rollout_ref.ref.megatron.context_parallel_size=$CP \
>     actor_rollout_ref.rollout.name=vllm \
>     actor_rollout_ref.rollout.mode=async \
>     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
>     actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
>     actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP \
>     actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
>     critic.optim.lr=1e-5 \
>     critic.model.path=$HF_MODEL_PATH \
>     critic.ppo_mini_batch_size=256 \
>     critic.ppo_micro_batch_size_per_gpu=4 \
>     critic.megatron.tensor_model_parallel_size=$TP \
>     critic.megatron.pipeline_model_parallel_size=$PP \
>     critic.megatron.context_parallel_size=$CP \
>     critic.megatron.param_offload=${offload} \
>     actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
>     actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
>     actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
>     actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
>     critic.megatron.use_dist_checkpointing=True \
>     critic.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
>     algorithm.use_kl_in_reward=False \
>     trainer.critic_warmup=0 \
>     trainer.logger='["console","wandb"]' \
>     trainer.project_name='Qwen3-30B-A3B' \
>     trainer.experiment_name='ppo-verl-r3' \
>     trainer.n_gpus_per_node=8 \
>     trainer.nnodes=$NODES \
>     trainer.total_training_steps=100 \
>     trainer.save_freq=-1 \
>     trainer.test_freq=5 \
>     trainer.total_epochs=100 2>&1 | tee verl_megatron_r3.log
>     
> ```

@stylke , I’ve updated the code. You can test it again.
---
### mergify @ 2025-12-16T07:51:48Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### xhx1022 @ 2025-12-16T08:35:31Z

> > > It looks like the current implementation only supports a single KV cache group (len(self.kv_cache_config.kv_cache_groups) == 1). Does it also support hybrid attention (global + SWA)?
> > 
> > 
> > At the moment, hybrid attention (global + SWA) is not supported yet. I’ll take a look later to see how it could be adapted or extended to support this use case.
> 
> Is hybrid attention supported when `--disable_hybrid_kv_cache_manager` is enabled?

@lkm2835 @awcvec I’ve updated the code. You can test it again.
---
### xhx1022 @ 2025-12-16T11:35:06Z

> First of all, thanks for this great feature! Really appreciate the work here. 🙏
> 
> I have a small suggestion regarding dtype compatibility. The current implementation uses torch.int16 / np.int16, but this causes issues when running in multi-GPU or multi-node setups. Specifically, NCCL's broadcast operation (used by broadcast_dict_tensor in verl's torch_functional.py) does not support int16 dtype.
> 
> Would it be possible to change these to torch.int32 / np.int32? This would ensure compatibility with NCCL collective operations while maintaining the intended functionality.

@sungyubkim Thanks a lot for the suggestion and for pointing this out! 
---
### shenh10 @ 2025-12-18T06:38:08Z

This current implementation does not support TP, right? For TP to use MultiProc mode, it needs to pass tensors back from the Worker, and we cannot directly call get_routed_experts in the scheduler.
---
### xhx1022 @ 2025-12-18T07:47:06Z

> This current implementation does not support TP, right? For TP to use MultiProc mode, it needs to pass tensors back from the Worker, and we cannot directly call get_routed_experts in the scheduler.

It can support TP.

In the TP setting, it is sufficient for rank 0 within each TP group to get the routed experts that the worker has saved in shared memory. 

At present, we have verified that DP + TP training works correctly in verl.
---
### shenh10 @ 2025-12-18T10:22:38Z

> > 目前的实现不支持 TP，对吗？TP 要使用 MultiProc 模式，需要从 Worker 传递张量，而我们不能直接在调度器中调用 get_routed_experts。
> 
> 它可以支持TP。
> 
> 在 TP 设置中，每个 TP 组内的 0 级节点只需获取工作节点保存在共享内存中的已路由专家即可。
> 
> 目前，我们已经验证了 DP + TP 训练在 verl 中能够正常工作。

OK... I see. Since you store the file directly into shm, you don't need to pass these results from worker back to scheduler.
---
### mergify @ 2025-12-20T18:25:41Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @xhx1022.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-12-21T08:08:52Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### QPHutu @ 2025-12-26T08:03:34Z

Is this PR be ready to be merged? I have been waiting for quite a long time.
---
### mzusman @ 2025-12-28T17:29:07Z

any news on this one ? : ) 
---
### xhx1022 @ 2025-12-29T06:42:27Z

> Is this PR be ready to be merged? I have been waiting for quite a long time.

@youkaichao 
---
### ForeverDJ-ux @ 2025-12-29T11:32:33Z

I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay
2-node result:
<img width="1312" height="318" alt="image" src="https://github.com/user-attachments/assets/0cadab4c-6875-4475-800d-27cdccaa0136" />

1-node result:
<img width="1305" height="319" alt="image" src="https://github.com/user-attachments/assets/5a395d68-27ec-42b8-b177-583db1dacd99" />
It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022 

---
### xhx1022 @ 2025-12-29T12:38:30Z

> I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> 
> 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022

@ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
---
### gxm651182644 @ 2025-12-29T14:28:24Z

我把vllm这个分支，结合verl框架来做moe的RL训练。
模型qwen3-VL-30B-A3B
使用dapo的策略，
数据集用的eurus-2-RL- Data-Code
 训练环境 单机8卡H200机器
推理Tp=1
训练Ep=8,Tp=1,pp=1
遇到的问题是，对比使用R2训练，R3训练的entropy是很大，差不多是 R2:R3=1:10的样子，这个应该不正确，这个情况有人遇到过么？
---
### robertgshaw2-redhat @ 2025-12-29T14:40:43Z

@youkaichao will follow this effort
---
### ForeverDJ-ux @ 2025-12-30T06:22:48Z

> > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> 
> @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？

I test router replay with  verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8  the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022 
---
### ForeverDJ-ux @ 2025-12-30T06:56:52Z

> 我把vllm这个分支，结合verl框架来做moe的RL训练。 模型qwen3-VL-30B-A3B 使用dapo的策略， 数据集用的eurus-2-RL- Data-Code 训练环境 单机8卡H200机器 推理Tp=1 训练Ep=8,Tp=1,pp=1 遇到的问题是，对比使用R2训练，R3训练的entropy是很大，差不多是 R2:R3=1:10的样子，这个应该不正确，这个情况有人遇到过么？

你的结果应该和我的现象是一致的，你如果TP设置成8结果是不是就是对的，因为只起了一个server, 设置成TP=4，起了成两个服务结果就不对了，你看是不是这个现象 @gxm651182644 ，我调试了一下，好像两个启动服务后，routed_experts和输入的token_ids就不匹配了
---
### gxm651182644 @ 2025-12-30T07:54:31Z

> > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > 
> > 
> > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> 
> I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022

2node ,vllm TP8 result is error
----
What does the result mean ? The rollout ouuput or the training entropy？

---
### ForeverDJ-ux @ 2025-12-30T08:04:18Z

> > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > 
> > > 
> > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > 
> > 
> > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> 
> ## 2node ,vllm TP8 result is error
> What does the result mean ? The rollout ouuput or the training entropy？

training entropy
---
### ForeverDJ-ux @ 2025-12-30T08:08:21Z

> > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > 
> > > 
> > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > 
> > 
> > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> 
> ## 2node ,vllm TP8 result is error
> What does the result mean ? The rollout ouuput or the training entropy？

<img width="1218" height="745" alt="image" src="https://github.com/user-attachments/assets/e399d62d-5312-4d76-b766-e6f30f9a3744" />
different result like this
---
### gxm651182644 @ 2025-12-30T08:32:21Z

> > 我把vllm这个分支，结合verl框架来做moe的RL训练。 模型qwen3-VL-30B-A3B 使用dapo的策略， 数据集用的eurus-2-RL- Data-Code 训练环境 单机8卡H200机器 推理Tp=1 训练Ep=8,Tp=1,pp=1 遇到的问题是，对比使用R2训练，R3训练的entropy是很大，差不多是 R2:R3=1:10的样子，这个应该不正确，这个情况有人遇到过么？
> 
> 你的结果应该和我的现象是一致的，你如果TP设置成8结果是不是就是对的，因为只起了一个server, 设置成TP=4，起了成两个服务结果就不对了，你看是不是这个现象 @gxm651182644 ，我调试了一下，好像两个启动服务后，routed_experts和输入的token_ids就不匹配了

复现了，我在单机8卡上，其他配置都不变，只把推理TP=8, 这时候训练的entropy就正常了。
---
### gxm651182644 @ 2025-12-30T08:33:12Z

> > > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > > 
> > > > 
> > > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > > 
> > > 
> > > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> > 
> > 
> > ## 2node ,vllm TP8 result is error
> > What does the result mean ? The rollout ouuput or the training entropy？
> 
> <img alt="image" width="1218" height="745" src="https://private-user-images.githubusercontent.com/25948894/530920082-e399d62d-5312-4d76-b766-e6f30f9a3744.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwODM2ODIsIm5iZiI6MTc2NzA4MzM4MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA5MjAwODItZTM5OWQ2MmQtNTMxMi00ZDc2LWI3NjYtZTZmMzBmOWEzNzQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjMwVDA4Mjk0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzZDRiZTY0MTNjOTMzODE1MzUxOTM2ZWNhOTRlYzU4ZDIwNTZhZTkwN2QyOTc2ZjM5MmMyOTYyNDYzMTEyOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ldPmb_sJ_XqZ4wT5TEW9H2qgDl9ju8Q_VPdQBF1h1UE"> different result like this

I have reproduced it
---
### ForeverDJ-ux @ 2025-12-30T09:04:18Z

> > > > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > > > 
> > > > > 
> > > > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > > > 
> > > > 
> > > > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> > > 
> > > 
> > > ## 2node ,vllm TP8 result is error
> > > What does the result mean ? The rollout ouuput or the training entropy？
> > 
> > 
> > <img alt="image" width="1218" height="745" src="https://private-user-images.githubusercontent.com/25948894/530920082-e399d62d-5312-4d76-b766-e6f30f9a3744.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwODM2ODIsIm5iZiI6MTc2NzA4MzM4MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA5MjAwODItZTM5OWQ2MmQtNTMxMi00ZDc2LWI3NjYtZTZmMzBmOWEzNzQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjMwVDA4Mjk0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzZDRiZTY0MTNjOTMzODE1MzUxOTM2ZWNhOTRlYzU4ZDIwNTZhZTkwN2QyOTc2ZjM5MmMyOTYyNDYzMTEyOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ldPmb_sJ_XqZ4wT5TEW9H2qgDl9ju8Q_VPdQBF1h1UE"> different result like this
> 
> I have reproduced it

Yeah, I debug the vllm routed result with 2 node tp8, the routed id is different with same input token id, I think there is a bug in vllm router replay when launch >=2 vllm server, maybe router id memory influenced together
---
### gxm651182644 @ 2025-12-30T09:45:38Z

> > > > > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > > > > 
> > > > > > 
> > > > > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > > > > 
> > > > > 
> > > > > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> > > > 
> > > > 
> > > > ## 2node ,vllm TP8 result is error
> > > > What does the result mean ? The rollout ouuput or the training entropy？
> > > 
> > > 
> > > <img alt="image" width="1218" height="745" src="https://private-user-images.githubusercontent.com/25948894/530920082-e399d62d-5312-4d76-b766-e6f30f9a3744.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwODM2ODIsIm5iZiI6MTc2NzA4MzM4MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA5MjAwODItZTM5OWQ2MmQtNTMxMi00ZDc2LWI3NjYtZTZmMzBmOWEzNzQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjMwVDA4Mjk0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzZDRiZTY0MTNjOTMzODE1MzUxOTM2ZWNhOTRlYzU4ZDIwNTZhZTkwN2QyOTc2ZjM5MmMyOTYyNDYzMTEyOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ldPmb_sJ_XqZ4wT5TEW9H2qgDl9ju8Q_VPdQBF1h1UE"> different result like this
> > 
> > 
> > I have reproduced it
> 
> Yeah, I debug the vllm routed result with 2 node tp8, the routed id is different with same input token id, I think there is a bug in vllm router replay when launch >=2 vllm server, maybe router id memory influenced together

I have fixed it as the attached file and it works for m e
---
### ForeverDJ-ux @ 2025-12-30T09:47:30Z

> > > > > > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > > > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > > > > > 
> > > > > > > 
> > > > > > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > > > > > 
> > > > > > 
> > > > > > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> > > > > 
> > > > > 
> > > > > ## 2node ,vllm TP8 result is error
> > > > > What does the result mean ? The rollout ouuput or the training entropy？
> > > > 
> > > > 
> > > > <img alt="image" width="1218" height="745" src="https://private-user-images.githubusercontent.com/25948894/530920082-e399d62d-5312-4d76-b766-e6f30f9a3744.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwODM2ODIsIm5iZiI6MTc2NzA4MzM4MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA5MjAwODItZTM5OWQ2MmQtNTMxMi00ZDc2LWI3NjYtZTZmMzBmOWEzNzQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjMwVDA4Mjk0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzZDRiZTY0MTNjOTMzODE1MzUxOTM2ZWNhOTRlYzU4ZDIwNTZhZTkwN2QyOTc2ZjM5MmMyOTYyNDYzMTEyOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ldPmb_sJ_XqZ4wT5TEW9H2qgDl9ju8Q_VPdQBF1h1UE"> different result like this
> > > 
> > > 
> > > I have reproduced it
> > 
> > 
> > Yeah, I debug the vllm routed result with 2 node tp8, the routed id is different with same input token id, I think there is a bug in vllm router replay when launch >=2 vllm server, maybe router id memory influenced together
> 
> I have fixed it as the attached file and it works for m e

How to fix it
---
### gxm651182644 @ 2025-12-30T11:31:17Z

> > > > > > > > > I use 2 node to run verl router replay with VLLM==0.12.0(change python code to standand version) tp=8 server async node， the result is error, while with 1 node, output loss is okay 2-node result: <img alt="image" width="1312" height="318" src="https://private-user-images.githubusercontent.com/25948894/530697005-0cadab4c-6875-4475-800d-27cdccaa0136.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcwMDUtMGNhZGFiNGMtNjg3NS00NDc1LTgwMGQtMjdjZGNjYWEwMTM2LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQwMjU0NjdhNWQ1NGQ3ZGE5N2E2NmZkZThiZmZkY2QwOTcxMmRhZmI0MTc3Y2QyM2QxZDc1ZDRmY2Y4N2IyYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.mwLxLzUqE09l_dsMHyLf6O0hC9JUZjbDYu3nwPbDz3o">
> > > > > > > > > 1-node result: <img alt="image" width="1305" height="319" src="https://private-user-images.githubusercontent.com/25948894/530697140-5a395d68-27ec-42b8-b177-583db1dacd99.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwMTE0OTIsIm5iZiI6MTc2NzAxMTE5MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA2OTcxNDAtNWEzOTVkNjgtMjdlYy00MmI4LWIxNzctNTgzZGIxZGFjZDk5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMjklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI5VDEyMjYzMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM1MDljMTA3MzJiNzY4Mzk2MjUxYmIzOTJhMTlkM2M1YmZkYzAzYWIzZmZjZmQ0NjVmZDExYjQxZmJkYTVmZjAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.j2VvwJm3aJjFq5bDHzo8IY3eradY1f512uEcakGn5E4"> It looks like two server hava influenced togather， Is anyone met the same error? @xhx1022
> > > > > > > > 
> > > > > > > > 
> > > > > > > > @ForeverDJ-ux Could you please share a bit more detail about your setup and reproduction steps？
> > > > > > > 
> > > > > > > 
> > > > > > > I test router replay with verl/examples/[run_qwen30_a3b_megatron_vllm.sh(https://github.com/volcengine/verl/blob/main/examples/router_replay/run_qwen30_a3b_megatron_vllm.sh) ， you can try with this script to reproduce with 2node, vllm TP8 the result is error, while use 1node, TP8 or 2node TP 16， the result entropy loss is good @xhx1022
> > > > > > 
> > > > > > 
> > > > > > ## 2node ,vllm TP8 result is error
> > > > > > What does the result mean ? The rollout ouuput or the training entropy？
> > > > > 
> > > > > 
> > > > > <img alt="image" width="1218" height="745" src="https://private-user-images.githubusercontent.com/25948894/530920082-e399d62d-5312-4d76-b766-e6f30f9a3744.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjcwODM2ODIsIm5iZiI6MTc2NzA4MzM4MiwicGF0aCI6Ii8yNTk0ODg5NC81MzA5MjAwODItZTM5OWQ2MmQtNTMxMi00ZDc2LWI3NjYtZTZmMzBmOWEzNzQ0LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEyMzAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjMwVDA4Mjk0MlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTUzZDRiZTY0MTNjOTMzODE1MzUxOTM2ZWNhOTRlYzU4ZDIwNTZhZTkwN2QyOTc2ZjM5MmMyOTYyNDYzMTEyOTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.ldPmb_sJ_XqZ4wT5TEW9H2qgDl9ju8Q_VPdQBF1h1UE"> different result like this
> > > > 
> > > > 
> > > > I have reproduced it
> > > 
> > > 
> > > Yeah, I debug the vllm routed result with 2 node tp8, the routed id is different with same input token id, I think there is a bug in vllm router replay when launch >=2 vllm server, maybe router id memory influenced together
> > 
> > 
> > I have fixed it as the attached file and it works for m e
> 
> How to fix it

pls refer my comments.

https://github.com/vllm-project/vllm/pull/28284/files#r2652808705
https://github.com/vllm-project/vllm/pull/28284/files#r2652811661
---
### xhx1022 @ 2025-12-30T11:37:23Z

@gxm651182644 Thank you for the warm and helpful response.
The issue occurred because I am using verl v0.5, while in the latest version of verl the code that modifies the vLLM instance_id has been removed, which leads to this problem when running on the latest version.
Sorry for the confusion this may have caused, and thank you all for your understanding.
@ForeverDJ-ux
---
### mergify @ 2026-01-06T18:52:47Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @xhx1022.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-01-07T02:04:43Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-01-07T02:17:00Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### ann-qin-lu @ 2026-01-08T19:14:44Z

Hi, thanks for the great work! 
I am wondering if there are any latency profiling result? (for the context, I cherry-picked this PR into vllm.0.12.0, and observed that the latency is around 10X. Not sure if I accidentally introduced some bug during cherry-pick, but would like to check if anyone has observed similar issue? 
---
### 22quinn @ 2026-01-08T23:35:41Z

> Hi, thanks for the great work! I am wondering if there are any latency profiling result? (for the context, I cherry-picked this PR into vllm.0.12.0, and observed that the latency is around 10X. Not sure if I accidentally introduced some bug during cherry-pick, but would like to check if anyone has observed similar issue?

Is the latency observed only when you turn on enable_return_routed_experts?
---
### ann-qin-lu @ 2026-01-09T00:21:20Z

> Is the latency observed only when you turn on enable_return_routed_experts?

yeah, I tried to enable and disable the flag as comparison, and observed this big latency jump. (I haven't profiled the standalone inference yet, the metric is tracked in the RL training, and the inference speed is somehow much slower after enabling this flag)
---
### xhx1022 @ 2026-01-09T01:52:04Z

> > Is the latency observed only when you turn on enable_return_routed_experts?
> 
> yeah, I tried to enable and disable the flag as comparison, and observed this big latency jump. (I haven't profiled the standalone inference yet, the metric is tracked in the RL training, and the inference speed is somehow much slower after enabling this flag)



> > Is the latency observed only when you turn on enable_return_routed_experts?
> 
> yeah, I tried to enable and disable the flag as comparison, and observed this big latency jump. (I haven't profiled the standalone inference yet, the metric is tracked in the RL training, and the inference speed is somehow much slower after enabling this flag)

You can directly run the ShareGPT benchmark script that comes with vLLM to test and compare the pure inference throughput/latency. From my own experiments, the overhead is very small, within ~3%.

In RL training runs, the rollout time observed in wandb does not only include the vLLM inference time. It likely also includes other overheads. Those parts should be attributed to verl and optimized there.

I suggest first running the vLLM server standalone and benchmarking it in isolation to verify whether there is any additional latency introduced by vLLM itself. This should help determine whether the observed slowdown comes from vLLM or from the rl framework.
@ann-qin-lu 
---
### mergify @ 2026-01-09T02:03:44Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-01-12T06:21:26Z

Hi @xhx1022, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---