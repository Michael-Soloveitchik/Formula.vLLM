# PR #13167: [Model] Deepseek GGUF support 

- field: quantization  | area: model_executor/layers
- author: SzymonOzog
- state: MERGED  created: 2025-02-12T16:49:18Z  merged: 2025-02-27T10:08:35Z
- url: https://github.com/vllm-project/vllm/pull/13167
- labels: documentation, structured-output, frontend, speculative-decoding, ready, ci/build, v1

## Body

This adds support for quantized deepseek versions from Unsloth:

Currently Huggingface does not support deepseek so I added an option to add an override path where we can read the correct config from. 

To run at the moment one needs to:

- download the tokenizer, configuration and modeling files from the [original deepseek repo](https://huggingface.co/deepseek-ai/DeepSeek-R1/tree/main) and the config.json from [Unsloth GGUF repo](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main).
- Change the torch_dtype in config to float16
- Merge the weights as instructed in the [vLLM docs](https://docs.vllm.ai/en/latest/features/quantization/gguf.html)

When initializing our deepseek model we need to pass the paths to our huggingface config and tokenizer:

```py
    from vllm import LLM, SamplingParams
    llm = LLM(model="/YOUR_PATH/DeepSeek_Unsloth/DeepSeek-R1-Q2_K/DeepSeek-R1-Q2_K.gguf",
              tokenizer="/YOUR_PATH/DeepSeek_Unsloth",
              hf_config_path="/YOUR_PATH/DeepSeek_Unsloth",
              enforce_eager=True, tensor_parallel_size=4, trust_remote_code=True,
              max_model_len=10000)
    sampling_params = SamplingParams(temperature=0.5, max_tokens=2000)


    def print_outputs(outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text\n: {generated_text}")
        print("-" * 80)
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "Why did the Roman Empire fall?",
        },
    ]
    outputs = llm.chat(conversation,
                       sampling_params=sampling_params,
                       use_tqdm=False)
    print_outputs(outputs)

```

Current issues:
~~Model loading is very slow as we load experts one by one~~ Fixed
GGUF MoE is a very naive implementation and is very slow

I plan to continue working on solving the aforementioned issues, can do this in this PR or future ones, sharing already because there seem to be a demand for running this.

Closes #12436

## Comments

### github-actions @ 2025-02-12T16:49:30Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### junuMoon @ 2025-02-13T05:55:01Z

I followed your instruction but I got an error
Tho I'm reading your Pr and it's great job 👍 

```
(VllmWorkerProcess pid=379357) /home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/parameter.py:167: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3683.)
(VllmWorkerProcess pid=379357)   return super().__torch_function__(func, types, args, kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks.
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] Traceback (most recent call last):
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/executor/multiproc_worker_utils.py", line 236, in _run_worker_process
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     output = run_method(worker, method, args, kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/utils.py", line 2224, in run_method
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/worker.py", line 229, in determine_num_available_blocks
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self.model_runner.profile_run()
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1234, in profile_run
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self._dummy_run(max_num_batched_tokens, max_num_seqs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1345, in _dummy_run
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self.execute_model(model_input, kv_caches, intermediate_tensors)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1718, in execute_model
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_or_intermediate_states = model_executable(
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                                     ^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 677, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states = self.model(input_ids, positions, kv_caches,
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/compilation/decorators.py", line 172, in __call__
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self.forward(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 633, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states, residual = layer(positions, hidden_states,
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 560, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states = self.mlp(hidden_states)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 159, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     shared_output = self.shared_experts(hidden_states)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 90, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     gate_up, _ = self.gate_up_proj(x)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                  ^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/linear.py", line 400, in forward
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     output_parallel = self.quant_method.apply(self, input_, bias)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/quantization/gguf.py", line 185, in apply
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     out = _fuse_mul_mat(x, qweight, qweight_type)
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/quantization/gguf.py", line 98, in _fuse_mul_mat
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return x @ qweight.T
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ~~^~~~~~~~~~~
(VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] RuntimeError: size mismatch, got input (10000), mat (10000x7168), vec (0)
```
---
### SzymonOzog @ 2025-02-13T08:45:58Z

> I followed your instruction but I got an error Tho I'm reading your Pr and it's great job 👍
> 
> ```
> (VllmWorkerProcess pid=379357) /home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/parameter.py:167: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3683.)
> (VllmWorkerProcess pid=379357)   return super().__torch_function__(func, types, args, kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] Exception in worker VllmWorkerProcess while processing method determine_num_available_blocks.
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] Traceback (most recent call last):
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/executor/multiproc_worker_utils.py", line 236, in _run_worker_process
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     output = run_method(worker, method, args, kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/utils.py", line 2224, in run_method
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/worker.py", line 229, in determine_num_available_blocks
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self.model_runner.profile_run()
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1234, in profile_run
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self._dummy_run(max_num_batched_tokens, max_num_seqs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1345, in _dummy_run
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     self.execute_model(model_input, kv_caches, intermediate_tensors)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return func(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/worker/model_runner.py", line 1718, in execute_model
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_or_intermediate_states = model_executable(
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                                     ^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 677, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states = self.model(input_ids, positions, kv_caches,
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/compilation/decorators.py", line 172, in __call__
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self.forward(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 633, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states, residual = layer(positions, hidden_states,
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 560, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     hidden_states = self.mlp(hidden_states)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 159, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     shared_output = self.shared_experts(hidden_states)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/models/deepseek_v2.py", line 90, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     gate_up, _ = self.gate_up_proj(x)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                  ^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return self._call_impl(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return forward_call(*args, **kwargs)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/linear.py", line 400, in forward
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     output_parallel = self.quant_method.apply(self, input_, bias)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/quantization/gguf.py", line 185, in apply
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     out = _fuse_mul_mat(x, qweight, qweight_type)
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]   File "/home/fran/vllm/vllm/model_executor/layers/quantization/gguf.py", line 98, in _fuse_mul_mat
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]     return x @ qweight.T
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242]            ~~^~~~~~~~~~~
> (VllmWorkerProcess pid=379357) ERROR 02-13 14:23:10 multiproc_worker_utils.py:242] RuntimeError: size mismatch, got input (10000), mat (10000x7168), vec (0)
> ```

Which of the quantized models are you trying to load?
---
### junuMoon @ 2025-02-13T09:19:46Z

https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S
this one
---
### SzymonOzog @ 2025-02-13T12:06:30Z

> https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S this one

Just tested and it works for me in a freshly checked out repo, are you sure that you merged gguf weights into one file? Could you share the scripts you are testing with?
---
### chuangzhidan @ 2025-02-14T02:51:43Z

**met an error：**
(base) ubuntu@localhost:/media/data/scripts$ python start_gguf.py
INFO 02-14 10:44:20 __init__.py:190] Automatically detected platform cuda.
Traceback (most recent call last):
  File "/media/data/xgp/scripts/start_gguf.py", line 3, in <module>
    llm = LLM(  
          ^^^^
  File "/home/ubuntu/.local/lib/python3.12/site-packages/vllm/utils.py", line 1051, in inner
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/.local/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 212, in __init__
    engine_args = EngineArgs(
                  ^^^^^^^^^^^
**TypeError: EngineArgs.__init__() got an unexpected keyword argument 'hf_config_path'
not sure where went wrong**

(base) ubuntu@localhost:/media/data/xgp/scripts$ pip show vllm
Name: vllm
Version: 0.7.2

base_dir="/data/llm/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S_MERGE"
from vllm import LLM, SamplingParams
llm = LLM(  
        # model="/YOUR_PATH/DeepSeek_Unsloth/DeepSeek-R1-Q2_K/DeepSeek-R1-Q2_K.gguf",
        model="/data/llm/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S_MERGE/DeepSeek-R1-UD-IQ1_S-merge.gguf",
        tokenizer=base_dir,
        hf_config_path=base_dir,
        enforce_eager=True,
        tensor_parallel_size=2,
        trust_remote_code=True,
        max_model_len=10000
        )
 
/data/llm/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S_MERGE/
-rw-rw-r-- 1 ubuntu ubuntu 1.6K Feb 14 10:38 config.json
-rw-rw-r-- 1 ubuntu ubuntu  11K Feb 14 10:37 configuration_deepseek.py
-rwxrwxrwx 1 root   root   131G Feb 12 17:47 DeepSeek-R1-UD-IQ1_S-merge.gguf*
-rw-rw-r-- 1 ubuntu ubuntu  171 Feb 14 10:37 generation_config.json
-rw-rw-r-- 1 ubuntu ubuntu  74K Feb 14 10:37 modeling_deepseek.py
-rw-rw-r-- 1 ubuntu ubuntu 3.6K Feb 14 10:37 tokenizer_config.json
-rw-rw-r-- 1 ubuntu ubuntu 7.5M Feb 14 10:37 tokenizer.json
---
### SzymonOzog @ 2025-02-14T04:20:25Z

@chuangzhidan Did you check out and install vllm from this PR? You seem to have a different version:

```
root@8b74d742fc51:~/vllm# pip show vllm
Name: vllm
Version: 0.7.3.dev4+ge152f295.precompiled
```
---
### zlh1992 @ 2025-02-15T07:29:15Z

> @chuangzhidan Did you check out and install vllm from this PR? You seem to have a different version:
> 
> ```
> root@8b74d742fc51:~/vllm# pip show vllm
> Name: vllm
> Version: 0.7.3.dev4+ge152f295.precompiled
> ```

Please show your detail environment?
---
### deoxynet @ 2025-02-16T20:52:56Z

Is this faster than llama.cpp for the unsloth quants? The llama.cpp version is also very unoptimized-- the GPUs sit mostly idle. Very eager to see it running on VLLM.
---
### seven1122 @ 2025-02-17T07:47:05Z

when will be merged?
---
### chuangzhidan @ 2025-02-17T07:59:10Z

> > @chuangzhidan Did you check out and install vllm from this PR? You seem to have a different version:
> > ```
> > root@8b74d742fc51:~/vllm# pip show vllm
> > Name: vllm
> > Version: 0.7.3.dev4+ge152f295.precompiled
> > ```
> 
> Please show your detail environment?

u are right ,it has something to do with vllm version and this pr environment.thank u 
---
### seven1122 @ 2025-02-18T03:17:27Z

met a KeyError: 'model.embed_tokens.qweight_type'
![IMG_014](https://github.com/user-attachments/assets/1fc00878-cfbd-4112-b679-4f8a03b50095)

Exception in worker VllmWorkerProcess while processing method load_model.
Traceback (most recent call last):
   File "/usr/local/lib/python3.12/dist - packages/vllm/executor/multiproc_worker_utils.py", line
     output = run_method(worker, method, args, kwargs)

   File "/usr/local/lib/python3.12/dist - packages/vllm/utils.py", line 2220, in run_method
     return func(*args, **kwargs)

   File "/usr/local/lib/python3.12/dist - packages/vllm/worker/worker.py", line 183, in load_model
     self.model_runner.load_model()
   File "/usr/local/lib/python3.12/dist - packages/vllm/worker/model_runner.py", line 1112, in
     self.model = get_model(vllm_config=self.vllm_config)

   File "/usr/local/lib/python3.12/dist - packages/vllm/model_executor/model_loader/init.py",
     return loader.load_model(vllm_config=vllm_config)

   File "/usr/local/lib/python3.12/dist - packages/vllm/model_executor/model_loader/loader.py",
     model.load_weights()
   File "/usr/local/lib/python3.12/dist - packages/vllm/model_executor/models/deepseek_v2.py",
     param = params_dict[name]

KeyError: 'model.embed_tokens.qweight_type'
---
### z7d1 @ 2025-02-18T08:22:46Z

I try to reproduce this PR and raise same error like @seven1122 .
```txt
[rank0]:   File "/home/X/new-vllm/vllm/worker/worker.py", line 183, in load_model
[rank0]:     self.model_runner.load_model()
[rank0]:   File "/home/X/new-vllm/vllm/worker/model_runner.py", line 1112, in load_model
[rank0]:     self.model = get_model(vllm_config=self.vllm_config)
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/X/new-vllm/vllm/model_executor/model_loader/__init__.py", line 14, in get_model
[rank0]:     return loader.load_model(vllm_config=vllm_config)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/home/X/new-vllm/vllm/model_executor/model_loader/loader.py", line 1320, in load_model
[rank0]:     model.load_weights(
[rank0]:   File "/home/X/new-vllm/vllm/model_executor/models/deepseek_v2.py", line 808, in load_weights
[rank0]:     param = params_dict[name]
[rank0]:             ~~~~~~~~~~~^^^^^^
[rank0]: KeyError: 'model.embed_tokens.qweight_type'
```
The checkpoints I used is [DeepSeek-R1-UD-IQ1_S](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S)

I merged multi .gguf files to single by:
```bash
./llama-gguf-split --merge ~/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf single.gguf
```
File path ~/DeepSeek-R1-UD-IQ1_S includes :
```txt
DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf  DeepSeek-R1-UD-IQ1_S-00002-of-00003.gguf  DeepSeek-R1-UD-IQ1_S-00003-of-00003.gguf
```

---
### leolmj @ 2025-02-18T09:39:54Z

i raise the same error as @seven1122 
---
### SzymonOzog @ 2025-02-18T09:49:15Z

@leolmj @seven1122 @zh-jp 

I'm having trouble reproducing the issue, could you share:

- your config.json
- your vllm version + the commit hash that you checked out
- the script that you are runnning 
---
### z7d1 @ 2025-02-18T14:14:41Z

Hello @SzymonOzog .
* [config.json](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/blob/main/config.json)
* The version of vllm is 0.7.2, and I have replaced the files that were specified in this PR.
* Regarding the script, I used the demo you provided and only changed the parameter `model` that passed in the .gguf file path.
---
### SzymonOzog @ 2025-02-18T16:27:26Z

@zh-jp 
You also need to change your dtype in config from `bfloat16` to `float16`. Also could you check out this PR and run it through it? There have been changes in vllm since 0.7.2 and I cannot promise backwards compatibility 
---
### davidsyoung @ 2025-02-18T17:43:28Z

I want to run this, but unfortunately I only have 14x3090 GPU's, so for tensor parallelism I need another 2 GPU's to get to 16. It would be great to see any kind of benchmarks on this compared to llama.cpp. Thank you!
---
### z7d1 @ 2025-02-19T09:06:10Z

@SzymonOzog thanks for your valuable suggestions. I build the vllm from `deepseek-gguf` branch from [your vllm repo](https://github.com/SzymonOzog/vllm). And successfully execute DeepSeek-R1-UD-IQ1_S in 4 NVIDIA A800-SXM4-80GB
---
### davidsyoung @ 2025-02-19T09:32:58Z

> @SzymonOzog thanks for your valuable suggestions. I install the vllm from `deepseek-gguf` branch from [your vllm repo](https://github.com/SzymonOzog/vllm). And successfully execute DeepSeek-R1-UD-IQ1_S in 4 NVIDIA A800-SXM4-80GB

Do you have a benchmark of performance?
---
### slr1997 @ 2025-02-19T09:42:43Z

@zh-jp Did you test the speed compared with the llama.cpp? And how much memory does it need at least?
---
### junuMoon @ 2025-02-19T13:10:28Z

> @zh-jp Did you test the speed compared with the llama.cpp? And how much memory does it need at least?

INFO 02-19 22:08:59 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 6.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.

Based on 8 x A100 GPUs, it's showing around 7 tokens/s
---
### leolmj @ 2025-02-19T15:06:26Z

@zh-jp  @SzymonOzog  how did install  the vllm from deepseek-gguf branch. it seems there is no wheel files to install. you rebuild it all? can you share the whl file?
---
### SzymonOzog @ 2025-02-19T16:30:10Z

@leolmj 
There were no changes to compiled files so you shoud be able to run with the wheel from the main branch 

`VLLM_USE_PRECOMPILED=1 pip install --editable .`
---
### priscilla-pan @ 2025-02-20T07:05:43Z

hi, @SzymonOzog, fantastic work!
I followed your instruction but I got an error at last.
The output of your script is only "\<think\>", something seems wrong.
By the way, the only difference with your script is I set max_model_len=2048, cos 10000 is too large for  A100 40G * 8

```
WARNING 02-20 06:20:54 config.py:1000] MLA is not supported with gguf quantization. Disabling MLA.
WARNING 02-20 06:20:54 config.py:1000] MLA is not supported with gguf quantization. Disabling MLA.
INFO 02-20 06:21:09 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 180.18 seconds
INFO 02-20 06:21:09 chat_utils.py:332] Detected the chat template content format to be 'string'. You can set `--chat-template-content-format` to override this.
Prompt: '<｜begin▁of▁sentence｜>You are a helpful assistant<｜User｜>Why did the Roman Empire fall?<｜Assistant｜>', Generated text
: <think>

--------------------------------------------------------------------------------

```

---
### leolmj @ 2025-02-20T13:46:39Z

@priscilla-pan  which gguf file are you using? DeepSeek-R1-UD-IQ1_S?
---
### SzymonOzog @ 2025-02-20T18:16:32Z

@priscilla-pan The model seems to run fine, the script I posted had max tokens set to 2 I believe you might have not changed it
---
### priscilla-pan @ 2025-02-21T01:00:26Z

> @priscilla-pan which gguf file are you using? DeepSeek-R1-UD-IQ1_S?

DeepSeek-R1-UD-Q2_K_XL
---
### priscilla-pan @ 2025-02-21T01:10:24Z

> @priscilla-pan The model seems to run fine, the script I posted had max tokens set to 2 I believe you might have not changed it

You are right. Thank you!
---
### priscilla-pan @ 2025-02-21T06:56:04Z

> > @priscilla-pan The model seems to run fine, the script I posted had max tokens set to 2 I believe you might have not changed it
> 
> You are right. Thank you!

By the way，although the model runs fine , the following error message appears，which is not friendly for users 
```
python test_deepseek_r1_gguf.py
INFO 02-20 05:42:32 __init__.py:190] Automatically detected platform cuda.
INFO 02-20 05:42:32 config.py:196] Replacing legacy 'type' key with 'rope_type'
ERROR 02-20 05:42:32 config.py:102] Error retrieving file list: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/deepseek-r1-gguf/DeepSeek-R1-2.51bit-merged/DeepSeek-R1-2.51bit.gguf'. Use `repo_type` argument if needed., retrying 1 of 2
ERROR 02-20 05:42:34 config.py:100] Error retrieving file list: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/deepseek-r1-gguf/DeepSeek-R1-2.51bit-merged/DeepSeek-R1-2.51bit.gguf'. Use `repo_type` argument if needed.
```

---
### vip-china @ 2025-02-21T11:09:10Z

> 感谢您的宝贵建议。我从您的 [vllm 存储库](https://github.com/SzymonOzog/vllm)的分支构建 VLLM。并在 4 个 NVIDIA A800-SXM4-80GB 中成功执行 DeepSeek-R1-UD-IQ1_S`deepseek-gguf`

 @zh-jp   Encounter an error,Have you encountered it
ImportError: cannot import name 'PoolingParams' from 'vllm' (unknown location)
---
### SzymonOzog @ 2025-02-21T14:45:31Z

@Isotr0py No worries, just incorporated the feedback and also added some asserts and removed the need to change dtype manually. I tried running `lm_eval` but it seems like there are some issues with reasoning models evaluation, they don't tend to follow guidlines (https://github.com/EleutherAI/lm-evaluation-harness/issues/2682) So I got a score of around 50% for the first 10 examples
---
### SzymonOzog @ 2025-02-21T14:48:11Z

Also FYI got a working version of fused moe kernel for gguf. I'll write necessary test and clean this up so expect an PR soon. Improvement goes to 32 tok/s now for 4xA100 TP4
---
### Isotr0py @ 2025-02-21T14:52:49Z

@SzymonOzog No problem, we can also test on [DeepSeek-V2-Lite](https://huggingface.co/legraphista/DeepSeek-V2-Lite-Chat-IMat-GGUF?show_file_info=DeepSeek-V2-Lite-Chat.IQ4_XS.gguf) gguf model. Just want to double check the model's output is fine, otherwise is LGTM :)
---
### SzymonOzog @ 2025-02-21T15:59:58Z

@Isotr0py Okay just tested and it looks like V2 doesn't work atm, it has a moe_intermediate_size of 1408 and current kernels require sizes to be padded to 512. It runs but the results are very flaky:
```
Prompt: '<｜begin▁of▁sentence｜>You are a helpful assistant\n\nUser: James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?\n\nAssistant:', Generated text
:  To solve this problem, we need to consider the nature of the problem. It's about the squareness of the question.
The sprint James did have 100m.

In this question, we can see three types of characters: characters, elements, and words.                                                                                                                                                                                                                                                                    
In these sentences, such as "How many kilor lines can James run down?” is an OK first, because it’
Exercise in these sentences is well-suited to James’ running
```
compared to deepseek-R1-GGUF-Q2_K:
```
Prompt: '<｜begin▁of▁sentence｜>You are a helpful assistant<｜User｜>James decides to run 3 sprints 3 times a week.  He runs 60 meters each sprint.  How many total meters does he run a week?<｜Assistant｜>', Generated text
: <think>
Okay, so James is going to run sprints three times a week, and each time he does three sprints. Each sprint is 60 meters. Hmm, let me break this down. First, I need to figure out how many sprints he does in total each week. Since he does three sprints per session and he does that three times a week, I think I multiply those numbers together. So 3 sprints per session times 3 sessions per week. That would be 3 times 3, which is 9 sprints in a week. Then, each sprint is 60 meters, so to find the total distance, I need to multiply the number of sprints by the distance per sprint. So 9 sprints multiplied by 60 meters each. Let me calculate that. 9 times 60... 9 times 60 is 540. So he runs 540 meters in a week. Let me double-check that. If each time he runs 3 sprints, that's 3 times 60, which is 180 meters per session. Then three times a week would be 180 times 3, which is also 540. Yep, that makes sense. So the total meters he runs a week is 540.
</think>

James runs 3 sprints per session, 3 times a week.

First, calculate the total number of sprints per week:
3 sprints/session × 3 sessions/week = 9 sprints/week

Then, multiply the number of sprints by the distance per sprint:
9 sprints/week × 60 meters/sprint = 540 meters/week

**Answer:** James runs a total of 540 meters each week.
```

I've seen a similar pattern of "looking valid but not really" when I tried running R1 in TP8 when interm moe dim was not a multiple of 512
---
### deoxynet @ 2025-02-21T18:06:07Z

@SzymonOzog 

I'm getting this error with tp=8 on [gguf-deepseek](https://github.com/SzymonOzog/vllm/commit/0ea9252d60ca56d43a3d57aab1a61741305039c1) branch:

```bash
vllm serve ~/unsloth/DeepSeek-R1-UD-IQ1_S.gguf \
 --tensor-parallel 8 \
--trust-remote-code \
--tokenizer ~/deepseek/ \
--hf-config-path ~/deepseek/ \
--enforce-eager \
--dtype float16
```

```
  File "/mnt/workspace1/vllm-test/vllm/vllm/model_executor/models/deepseek_v2.py", line 591, in <lambda>                                                                                                                                           
    lambda prefix: DeepseekV2DecoderLayer(                                                                                                                                                                                                         
                   ^^^^^^^^^^^^^^^^^^^^^^^                                                                                                                                                                                                         
  File "/mnt/workspace1/vllm-test/vllm/vllm/model_executor/models/deepseek_v2.py", line 517, in __init__
    self.mlp = DeepseekV2MoE(
               ^^^^^^^^^^^^^^
  File "/mnt/workspace1/vllm-test/vllm/vllm/model_executor/models/deepseek_v2.py", line 129, in __init__
    self.experts = FusedMoE(
                   ^^^^^^^^^
  File "/mnt/workspace1/vllm-test/vllm/vllm/model_executor/layers/fused_moe/layer.py", line 310, in __init__
    self.quant_method.create_weights(layer=self, **moe_quant_params)
  File "/mnt/workspace1/vllm-test/vllm/vllm/model_executor/layers/quantization/gguf.py", line 206, in create_weights
    assert intermediate_size_per_partition % 512 == 0, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: GGUF requires a block size of 512, you are running with 256, please adjust your tensor parallel size
```

```bash
uv pip show vllm
Name: vllm
Version: 0.7.3.dev107+gc540baf0.precompiled
```

---
### SzymonOzog @ 2025-02-22T10:33:31Z

@accupham As noted by the error GGUF requires a block size of 512, I updated my example code. Could you please try to run with TP=4?
---
### justinjja @ 2025-02-23T05:49:41Z

I tried tensor parallel 4 and pipeline 2 and I get a NotImplementedError error,
Is there no way to use more than 4 gpus?
---
### fclearner @ 2025-02-24T10:39:02Z

Hi @SzymonOzog,

Thank you for your efforts on this branch! I have tested it, and it works well. However, I noticed that the inference speed for DeepSeek-R1-UD-Q2_K_XL is approximately 7 tokens/s when using your example code.

Is there any recommended method or optimization to improve the inference speed?
Looking forward to your suggestions!
Alan
---
### SzymonOzog @ 2025-02-24T14:59:59Z

@justinjja Could you paste the full error message? I think pipeline parallel requires async engine so you could run it via vllm serve

---
### SzymonOzog @ 2025-02-24T15:02:05Z

@fclearner 
Hey Alan, if you're eager to try to get it to run fast you could try checking out the branch where I'm working on a fast MoE kernel https://github.com/SzymonOzog/vllm/tree/gguf-moe and compile it yourself. This is still work in progress and only vibe tested but as of now I can run it in 32 tok/s for TP4
---
### mergify @ 2025-02-24T15:34:11Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @SzymonOzog.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### fclearner @ 2025-02-25T05:46:46Z

> @fclearner Hey Alan, if you're eager to try to get it to run fast you could try checking out the branch where I'm working on a fast MoE kernel https://github.com/SzymonOzog/vllm/tree/gguf-moe and compile it yourself. This is still work in progress and only vibe tested but as of now I can run it in 32 tok/s for TP4

Hi @SzymonOzog 
Thank you for your suggestions and for appreciating the work on GGUF inference! I’ve tried running your gguf-moe branch, but encountered an error: "operator _C::ggml_moe_a8 does not exist". I suspect this issue might be related to torch version mismatch(I'm using torch 2.5.1). Could you please provide guidance or suggest a possible fix?

Looking forward to your reply, and thanks again for your great contributions!
---
### SzymonOzog @ 2025-02-26T09:45:55Z

@Isotr0py Incorporated all feedback, thanks a lot for reviewing this!
---
### SzymonOzog @ 2025-02-26T10:46:39Z

@fclearner You would need to manually compile the vllm repo to run this kernel.
`pip install -e .`
---
### SzymonOzog @ 2025-02-26T18:04:54Z

Sorry for the notification everyone, my mistake when rebasing
---
### ZinonDynn @ 2025-02-27T08:41:17Z

@zh-jp Have you ever encountered a code exception like this:

 `transformers/modeling_gguf_pytorch_utils.py", line 399, in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture deepseek2 is not supported yet.`

What version of "transformers" and "vllm" did you use?
---
### cjackal @ 2025-02-27T09:12:30Z

> @zh-jp Have you ever encountered a code exception like this:
> 
> 
> 
>  `transformers/modeling_gguf_pytorch_utils.py", line 399, in load_gguf_checkpoint
> 
>     raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
> 
> ValueError: GGUF model with architecture deepseek2 is not supported yet.`
> 
> 
> 
> What version of "transformers" and "vllm" did you use?

I think you need to set --trust-remote-code to use local model class.

My env which successfully serves unsloth deepseek-r1 2.51bit:
transformers==4.48.3
vllm==main(0.7.4.dev116)
---
### ZinonDynn @ 2025-02-27T14:39:00Z

@cjackal I'm deploying on 8 * MI250 GPUs, which are AMD GPUs with 64GB memory. The version of rocm-core is 6.3.2.
```
git clone -b main --single-branch https://github.com/vllm-project/vllm && cd vllm && git checkout 7f0be2a
python install develop
vllm Version: 0.7.3.dev0+g0408efc6d.d20250223.rocm631
transformers Version: 4.48.3
```
and run this command:
```
python -m vllm.entrypoints.api_server --model /app/model/DeepSeek-R1-UD-IQ1_S-merged.gguf -tp 4 --port 8000 --enforce-eager --trust-remote-code
```
The code still throws an exception:
```
ValueError: GGUF model with architecture deepseek2 is not supported yet.
```
I'd be very glad if I could get your guidance.
---
### cjackal @ 2025-02-27T15:09:15Z

> @cjackal I'm deploying on 8 * MI250 GPUs, which are AMD GPUs with 64GB memory. The version of rocm-core is 6.3.2.
> 
> ```
> git clone -b main --single-branch https://github.com/vllm-project/vllm && cd vllm && git checkout 7f0be2a
> python install develop
> vllm Version: 0.7.3.dev0+g0408efc6d.d20250223.rocm631
> transformers Version: 4.48.3
> ```
> 
> and run this command:
> 
> ```
> python -m vllm.entrypoints.api_server --model /app/model/DeepSeek-R1-UD-IQ1_S-merged.gguf -tp 4 --port 8000 --enforce-eager --trust-remote-code
> ```
> 
> The code still throws an exception:
> 
> ```
> ValueError: GGUF model with architecture deepseek2 is not supported yet.
> ```
> 
> I'd be very glad if I could get your guidance.

You need to set `--hf-config-path` and `--tokenizer` as in the OP's instruction. If your repo commit is high enough, you also need to set `VLLM_MLA_DISABLE=1` as current Triton MLA kernel is not supported for gguf format.
---
### vip-china @ 2025-02-28T02:20:07Z

I use 8 A6000 cards, and the inference speed is only 3.5 tokens/s.
![image](https://github.com/user-attachments/assets/9552609f-baf8-4e0e-80c9-f121db4711a2)

---
### boywuxu @ 2025-03-01T13:26:52Z

I got an error:  AttributeError: 'TritonMLAImpl' object has no attribute 'W_Q_UK'

My vllm version:
Name: vllm
Version: 0.7.4.dev150+gb3f7aacc.d20250301.precompiled
---
### cjackal @ 2025-03-01T13:28:37Z

> I got an error: AttributeError: 'TritonMLAImpl' object has no attribute 'W_Q_UK'
> 
> My vllm version: Name: vllm Version: 0.7.4.dev150+gb3f7aacc.d20250301.precompiled

If your repo commit is high enough, you also need to set `VLLM_MLA_DISABLE=1` as current Triton MLA kernel is not supported for gguf format.
---
### boywuxu @ 2025-03-02T05:31:02Z

> > I got an error: AttributeError: 'TritonMLAImpl' object has no attribute 'W_Q_UK'
> > My vllm version: Name: vllm Version: 0.7.4.dev150+gb3f7aacc.d20250301.precompiled
> 
> If your repo commit is high enough, you also need to set `VLLM_MLA_DISABLE=1` as current Triton MLA kernel is not supported for gguf format.

Yes, it worked! Thank u!
And successfully execute DeepSeek-R1-UD-IQ1_S in 8 NVIDIA 4090-24GB
---
### zlh1992 @ 2025-03-02T07:47:04Z

> > > I got an error: AttributeError: 'TritonMLAImpl' object has no attribute 'W_Q_UK'
> > > My vllm version: Name: vllm Version: 0.7.4.dev150+gb3f7aacc.d20250301.precompiled
> > 
> > 
> > If your repo commit is high enough, you also need to set `VLLM_MLA_DISABLE=1` as current Triton MLA kernel is not supported for gguf format.
> 
> Yes, it worked! Thank u! And successfully execute DeepSeek-R1-UD-IQ1_S in 8 NVIDIA 4090-24GB

how much tps did you got?
---
### maekawatoshiki @ 2025-03-02T08:59:50Z

With the long context (16000) setting and Q2_K_XL model, I got gibberish output after the normal 1000-3000 tokens.
Anyone else experiencing this?
---
### SzymonOzog @ 2025-03-02T09:08:59Z

@maekawatoshiki Sometimes this happens when float16 overflows. This is because we are loading a bfloat16 model as float16. https://github.com/vllm-project/vllm/pull/14043 will fix this
---
### maekawatoshiki @ 2025-03-02T09:38:17Z

@SzymonOzog Thank you. I'll try your patch.
---
### iehgit @ 2025-03-02T19:24:36Z

With
```bash
python3 -m vllm.entrypoints.api_server --model /media/raid0/DeepSeek-R1-UD-Q2_K_XL-merged.gguf --enforce-eager --trust-remote-code --tokenizer ~/deepseek/ --hf-config-path ~/deepseek/ --dtype float16 --max-model-len 8192
```

and ` VLLM_MLA_DISABLE=1` set, I get:

> [rank0]: ValueError: Head size 192 is not supported by PagedAttention. Supported head sizes are: [32, 64, 80, 96, 112, 128, 256].

Tried various quants, all with the same effect. Adding `-tp 4` also yielded the same error message.
Built according to this: [Installation with CPU](https://docs.vllm.ai/en/v0.6.2/getting_started/cpu-installation.html)
---
### boywuxu @ 2025-03-03T03:39:06Z

> how much tps did you got? @zlh1992 

one request 7 tokens/s, if more requests 10 tokens/s
---
### zlh1992 @ 2025-03-08T06:10:27Z

> @fclearner Hey Alan, if you're eager to try to get it to run fast you could try checking out the branch where I'm working on a fast MoE kernel https://github.com/SzymonOzog/vllm/tree/gguf-moe and compile it yourself. This is still work in progress and only vibe tested but as of now I can run it in 32 tok/s for TP4

I build your repo from the banch:gguf-moe
Then I tested servering unsloth deepseek-R1 1.58bit gguf with the command:

python3 -m vllm.entrypoints.api_server --model /path/DeepSeek-R1-GGUF/merged_file.gguf --enforce-eager --trust-remote-code --tokenizer /path/DeepSeek-R1-GGUF/ --hf-config-path /path/DeepSeek-R1-GGUF/ --max-model-len 3900 -tp 8 --dtype float16 --gpu-memory-utilization 0.99

My device is 2080ti 22gb * 8 cuda12.4

At the begin I found my device has low compute ability to afford bfloat
so I change the tensor dtype in vllm/vllm/model_executor/layers/layernorm.py:

        from vllm import _custom_ops as ops

        if residual is not None:
            x = x.to(dtype=torch.float16)
            residual = residual.to(dtype=torch.float16)
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        x = x.to(dtype=torch.float16)
        out = out.to(dtype=torch.float16)
        ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out

Then I get my vllm engine started successfully, with the performance as:

INFO 03-08 05:57:48 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 11.3%, CPU KV cache usage: 0.0%.
INFO 03-08 05:57:53 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 12.1%, CPU KV cache usage: 0.0%.
INFO 03-08 05:57:58 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 12.6%, CPU KV cache usage: 0.0%.
INFO 03-08 05:58:03 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.6 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 13.4%, CPU KV cache usage: 0.0%.
INFO 03-08 05:58:08 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 14.2%, CPU KV cache usage: 0.0%.
INFO 03-08 05:58:13 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 15.0%, CPU KV cache usage: 0.0%.
INFO 03-08 05:58:19 metrics.py:455] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 5.7 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 15.8%, CPU KV cache usage: 0.0%.

---
### SzymonOzog @ 2025-03-08T07:46:12Z

@zlh1992 On my branch you can disable --enforce-eagedr, this should give you a boost in performance

---
### richjjj @ 2025-03-10T02:27:04Z

@SzymonOzog I build your banch:gguf-moe from source. My device is 4xA100 and command is ``` vllm serve /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S_merged.gguf \
--tensor-parallel 4 \
--trust-remote-code \
--tokenizer /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1 \
--hf-config-path /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1 \
--gpu-memory-utilization 0.98 \
--max-model-len 1200```.
but it "Error while capturing CUDA graph".
Could you give me some suggest about how to fix.
here is the log:
[log.txt](https://github.com/user-attachments/files/19154381/log.txt)

---
### richjjj @ 2025-03-10T02:29:17Z

> @SzymonOzog I build your banch:gguf-moe from source. My device is 4xA100 and command is ` vllm serve /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S_merged.gguf \ --tensor-parallel 4 \ --trust-remote-code \ --tokenizer /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1 \ --hf-config-path /data/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1 \ --gpu-memory-utilization 0.98 \ --max-model-len 1200`. but it "Error while capturing CUDA graph". Could you give me some suggest about how to fix. here is the log: [log.txt](https://github.com/user-attachments/files/19154381/log.txt)

By the way,it works when I set ``` CUDA_LAUNCH_BLOCKING=1```,but only 3.9tokens/s
---
### SzymonOzog @ 2025-03-10T07:31:53Z

@richjjj Yeah there is no support for fast kernels for I quants, if you want to run this faster I'd suggest switching to one of the Q_K quants, they should be much faster
---
### justinjja @ 2025-03-10T21:04:48Z

It looks like TP16 isn't supported on the gguf-moe branch,
Would it just be a matter of adding a "16" somewhere and recompiling, or more complicated than that?

That said I'm able to get 36T/s on 3090's with TP8 PP2 on the Q2_K_XL quant.
Close to 2x faster than I get on llama.cpp
Thanks!
---
### SzymonOzog @ 2025-03-11T07:44:00Z

@justinjja I think TP 16 is impossible at the moment without a serious rewrite, this is because of the property of GGUF quantization. 

How it works is that it quantizes in blocks, one block of Q2 quants contains 256 values and is structured like:

[SCALES|VALUES]

Deepseek has MoE intermediate size of 2048 so 8 blocks. This means that if we were to split it in 16 parts we would start cutting blocks in half
---
### davidsyoung @ 2025-03-15T13:53:52Z

Seeing gibberish produced at the moment with: 

```
python -m vllm.entrypoints.openai.api_server --model /models/deepseek-config/DeepSeek-R1-Q3_K_M.gguf --seed 3407 --served-model-name deepseek-r1 --hf-config-path /models/deepseek-config --tokenizer /models/deepseek-config --gpu-memory-utilization 0.98 --max-model-len 10240 --trust-remote-code --tensor-parallel-size 8 --pipeline-parallel-size 2 --host 192.168.10.225 --port 8000
```

Model:

https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/tree/main/DeepSeek-R1-Q3_K_M


Output:

```
.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state

�<｜place▁holder▁no▁791｜>

�

**

(

Thought for 1 seconds
<｜place▁holder▁no▁16｜> 」h1superscript

.M

:，1,init<｜place▁holder▁no▁793｜>

�aintunciation,”)<｜place▁holder▁no▁11｜> 

.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state

�<｜place▁holder▁no▁791｜>

�

**

(</think>
<｜place▁holder▁no▁795｜></think>

Thought for 0 seconds
.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state

�<｜place▁holder▁no▁791｜>

�

**

(</think>
<｜place▁holder▁no▁795｜></think>

Thought for 0 seconds
.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state

�<｜place▁holder▁no▁791｜>

�

**

(</think>
<｜place▁holder▁no▁795｜></think>

Thought for 0 seconds
.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state

�<｜place▁holder▁no▁791｜>

�

**

(</think>
<｜place▁holder▁no▁795｜></think>

Thought for 0 seconds
.,

The ō�的:，在
.Psalm�

◦

.

指

{documentg

�
」

<｜place▁holder▁no▁791｜>.

havage

leB state
```

I believe I have everything set up right @SzymonOzog but clearly not, any ideas?
---
### cjackal @ 2025-03-15T13:58:27Z

> Seeing gibberish produced at the moment with:
> 
> ```
> python -m vllm.entrypoints.openai.api_server --model /models/deepseek-config/DeepSeek-R1-Q3_K_M.gguf --seed 3407 --served-model-name deepseek-r1 --hf-config-path /models/deepseek-config --tokenizer /models/deepseek-config --gpu-memory-utilization 0.98 --max-model-len 10240 --trust-remote-code --tensor-parallel-size 8 --pipeline-parallel-size 2 --host 192.168.10.225 --port 8000
> ```
> 
> Model:
> 
> https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/tree/main/DeepSeek-R1-Q3_K_M
> 
> Output:
> 
> ```
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> 
> �<｜place▁holder▁no▁791｜>
> 
> �
> 
> **
> 
> (
> 
> Thought for 1 seconds
> <｜place▁holder▁no▁16｜> 」h1superscript
> 
> .M
> 
> :，1,init<｜place▁holder▁no▁793｜>
> 
> �aintunciation,”)<｜place▁holder▁no▁11｜> 
> 
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> 
> �<｜place▁holder▁no▁791｜>
> 
> �
> 
> **
> 
> (</think>
> <｜place▁holder▁no▁795｜></think>
> 
> Thought for 0 seconds
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> 
> �<｜place▁holder▁no▁791｜>
> 
> �
> 
> **
> 
> (</think>
> <｜place▁holder▁no▁795｜></think>
> 
> Thought for 0 seconds
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> 
> �<｜place▁holder▁no▁791｜>
> 
> �
> 
> **
> 
> (</think>
> <｜place▁holder▁no▁795｜></think>
> 
> Thought for 0 seconds
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> 
> �<｜place▁holder▁no▁791｜>
> 
> �
> 
> **
> 
> (</think>
> <｜place▁holder▁no▁795｜></think>
> 
> Thought for 0 seconds
> .,
> 
> The ō�的:，在
> .Psalm�
> 
> ◦
> 
> .
> 
> 指
> 
> {documentg
> 
> �
> 」
> 
> <｜place▁holder▁no▁791｜>.
> 
> havage
> 
> leB state
> ```
> 
> I believe I have everything set up right @SzymonOzog but clearly not, any ideas?

Which version of vLLM have you used? If you are using current main, try bfloat16 dtype instead.
---
### davidsyoung @ 2025-03-15T13:59:20Z

> > Seeing gibberish produced at the moment with:
> > ```
> > python -m vllm.entrypoints.openai.api_server --model /models/deepseek-config/DeepSeek-R1-Q3_K_M.gguf --seed 3407 --served-model-name deepseek-r1 --hf-config-path /models/deepseek-config --tokenizer /models/deepseek-config --gpu-memory-utilization 0.98 --max-model-len 10240 --trust-remote-code --tensor-parallel-size 8 --pipeline-parallel-size 2 --host 192.168.10.225 --port 8000
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > Model:
> > https://huggingface.co/daydream-org/DeepSeek-R1-GGUF-11446/tree/main/DeepSeek-R1-Q3_K_M
> > Output:
> > ```
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > 
> > �<｜place▁holder▁no▁791｜>
> > 
> > �
> > 
> > **
> > 
> > (
> > 
> > Thought for 1 seconds
> > <｜place▁holder▁no▁16｜> 」h1superscript
> > 
> > .M
> > 
> > :，1,init<｜place▁holder▁no▁793｜>
> > 
> > �aintunciation,”)<｜place▁holder▁no▁11｜> 
> > 
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > 
> > �<｜place▁holder▁no▁791｜>
> > 
> > �
> > 
> > **
> > 
> > (</think>
> > <｜place▁holder▁no▁795｜></think>
> > 
> > Thought for 0 seconds
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > 
> > �<｜place▁holder▁no▁791｜>
> > 
> > �
> > 
> > **
> > 
> > (</think>
> > <｜place▁holder▁no▁795｜></think>
> > 
> > Thought for 0 seconds
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > 
> > �<｜place▁holder▁no▁791｜>
> > 
> > �
> > 
> > **
> > 
> > (</think>
> > <｜place▁holder▁no▁795｜></think>
> > 
> > Thought for 0 seconds
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > 
> > �<｜place▁holder▁no▁791｜>
> > 
> > �
> > 
> > **
> > 
> > (</think>
> > <｜place▁holder▁no▁795｜></think>
> > 
> > Thought for 0 seconds
> > .,
> > 
> > The ō�的:，在
> > .Psalm�
> > 
> > ◦
> > 
> > .
> > 
> > 指
> > 
> > {documentg
> > 
> > �
> > 」
> > 
> > <｜place▁holder▁no▁791｜>.
> > 
> > havage
> > 
> > leB state
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > I believe I have everything set up right @SzymonOzog but clearly not, any ideas?
> 
> Which version of vLLM have you used? If you are using current main, try bfloat16 dtype instead.

Yes this is current main. Will try now. Thank you. Just to confirm, do you mean `--dtype bfloat16` argument, or changing `config.json` back?
---
### cjackal @ 2025-03-15T14:04:45Z

> Yes this is current main. Will try now. Thank you. Just to confirm, do you mean `--dtype bfloat16` argument, or changing `config.json` back?

AFAIK both are equal. I'm using the original `config.json` from huggingface though.
---
### davidsyoung @ 2025-03-15T14:28:50Z

That fixed it, thank you!
---
### davidsyoung @ 2025-03-15T18:30:14Z

@SzymonOzog Would it be possible to support i-matrix quants? It would be really useful to squeeze as much performance as possible! Thank you
---
### davidsyoung @ 2025-03-15T21:37:23Z

> With the long context (16000) setting and Q2_K_XL model, I got gibberish output after the normal 1000-3000 tokens. Anyone else experiencing this?

I seem to be seeing this same issue as this after longer context. Not too sure what the issue is, the model is set to `bfloat16` now as above.
---
### davidsyoung @ 2025-03-16T10:38:56Z

When I say experiencing the same issue, the model just seems to start talking about something completely irrelevant - like outputting code. 

I don’t have this issue in ik_llama.cpp inference for example. 

Has anyone experienced this? It’s a Q3 or mixed Q3/4 quant. 
---
### SzymonOzog @ 2025-03-17T04:41:43Z

I also had issues with long context but that got resolved after switching to bfloat16. This was caused by model outputing NaN after some accumulation and causing a token 0 to get emited(begin of sequence)
---
### SzymonOzog @ 2025-03-17T04:43:21Z

> @SzymonOzog Would it be possible to support i-matrix quants? It would be really useful to squeeze as much performance as possible! Thank you

No plans at the moment, I'm using Q_4_K and plan to invest time mostly in improving Q_K quants
---
### davidsyoung @ 2025-03-17T09:50:37Z

> I also had issues with long context but that got resolved after switching to bfloat16. This was caused by model outputing NaN after some accumulation and causing a token 0 to get emited(begin of sequence)

Seems to be happening still to me with bfloat16. 

Could you give me your run command, and what commit you’re on? 

Seems to happen later in context. Have tried a couple of PR’s as well, with no luck. Have tried to redownload the quant, my own quant, etc. 
---
### davidsyoung @ 2025-03-18T13:45:58Z

Hey @SzymonOzog - I'm still having the same problem unfortunately without being any closer to resolving exactly why. Would it be possible to get a gentle nudge in the right direction on what I could look for, or what commit I could run to test?

It's exactly the same as you're saying, it seems like a 0 token is emitted after a while and the topic changes to something completely different towards the end of a response.
---
### SzymonOzog @ 2025-03-18T19:50:41Z

@davidsyoung I'm running on https://github.com/vllm-project/vllm/pull/14666 with no changes to the default except `tensor_parallel=8` and `max_model_len=20000` but that's just my memory limitations 
---
### joshuakoh1 @ 2025-03-19T21:05:59Z

What do I need to run this with native 0.8.0 version?
```
2025-03-19T20:59:41+00:00 - gpustack.worker.backends.vllm - ERROR - Failed to derive max model length: 
2025-03-19T20:59:41+00:00 - gpustack.worker.backends.vllm - INFO - Starting vllm server
INFO 03-19 20:59:46 [__init__.py:256] Automatically detected platform cuda.
INFO 03-19 20:59:47 [api_server.py:977] vLLM API server version 0.8.0
INFO 03-19 20:59:47 [api_server.py:978] args: Namespace(subparser='serve', model_tag='/mnt/shared/merged_models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M.gguf', config='', host='0.0.0.0', port=40600, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, enable_ssl_refresh=False, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='/mnt/shared/merged_models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M.gguf', task='auto', tokenizer=None, hf_config_path=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='auto', kv_cache_dtype='auto', max_model_len=15000, guided_decoding_backend='xgrammar', logits_processor_pattern=None, model_impl='auto', distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=12, enable_expert_parallel=False, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=None, swap_space=4, cpu_offload_gb=5.0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_partial_prefills=1, max_long_partial_prefills=1, long_prefill_token_threshold=0, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization='gguf', rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, use_tqdm_on_load=True, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=['deepseek-r1-iq1m-merged'], qlora_adapter_name_or_path=None, show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', scheduler_cls='vllm.core.scheduler.Scheduler', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', worker_extension_cls='', generation_config='auto', override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, additional_config=None, enable_reasoning=False, reasoning_parser=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, enable_server_load_tracking=False, dispatch_function=<function ServeSubcommand.cmd at 0x7e0df3a7e480>)
Traceback (most recent call last):
  File "/var/lib/gpustack/bin/vllm_v0.8.0", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/cli/main.py", line 75, in main
    args.dispatch_function(args)
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/cli/serve.py", line 33, in cmd
    uvloop.run(run_server(args))
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/uvloop/__init__.py", line 109, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/uvloop/__init__.py", line 61, in wrapper
    return await main
           ^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 1012, in run_server
    async with build_async_engine_client(args) as engine_client:
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 141, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 161, in build_async_engine_client_from_engine_args
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1206, in create_engine_config
    model_config = self.create_model_config()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1121, in create_model_config
    return ModelConfig(
           ^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/config.py", line 333, in __init__
    hf_config = get_config(self.hf_config_path or self.model,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 280, in get_config
    config_dict, _ = PretrainedConfig.get_config_dict(
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/configuration_utils.py", line 594, in get_config_dict
    config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/configuration_utils.py", line 685, in _get_config_dict
    config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py", line 399, in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture deepseek2 is not supported yet.
```
---
### joshuakoh1 @ 2025-04-02T06:06:55Z

> What do I need to run this with native 0.8.0 version?
> 
> ```
> 2025-03-19T20:59:41+00:00 - gpustack.worker.backends.vllm - ERROR - Failed to derive max model length: 
> 2025-03-19T20:59:41+00:00 - gpustack.worker.backends.vllm - INFO - Starting vllm server
> INFO 03-19 20:59:46 [__init__.py:256] Automatically detected platform cuda.
> INFO 03-19 20:59:47 [api_server.py:977] vLLM API server version 0.8.0
> INFO 03-19 20:59:47 [api_server.py:978] args: Namespace(subparser='serve', model_tag='/mnt/shared/merged_models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M.gguf', config='', host='0.0.0.0', port=40600, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, chat_template_content_format='auto', response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, enable_ssl_refresh=False, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_request_id_headers=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='/mnt/shared/merged_models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M.gguf', task='auto', tokenizer=None, hf_config_path=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', trust_remote_code=False, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='auto', kv_cache_dtype='auto', max_model_len=15000, guided_decoding_backend='xgrammar', logits_processor_pattern=None, model_impl='auto', distributed_executor_backend=None, pipeline_parallel_size=1, tensor_parallel_size=12, enable_expert_parallel=False, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=None, enable_prefix_caching=None, disable_sliding_window=False, use_v2_block_manager=True, num_lookahead_slots=0, seed=None, swap_space=4, cpu_offload_gb=5.0, gpu_memory_utilization=0.9, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_partial_prefills=1, max_long_partial_prefills=1, long_prefill_token_threshold=0, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization='gguf', rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, use_tqdm_on_load=True, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=['deepseek-r1-iq1m-merged'], qlora_adapter_name_or_path=None, show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', scheduler_cls='vllm.core.scheduler.Scheduler', override_neuron_config=None, override_pooler_config=None, compilation_config=None, kv_transfer_config=None, worker_cls='auto', worker_extension_cls='', generation_config='auto', override_generation_config=None, enable_sleep_mode=False, calculate_kv_scales=False, additional_config=None, enable_reasoning=False, reasoning_parser=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, enable_server_load_tracking=False, dispatch_function=<function ServeSubcommand.cmd at 0x7e0df3a7e480>)
> Traceback (most recent call last):
>   File "/var/lib/gpustack/bin/vllm_v0.8.0", line 8, in <module>
>     sys.exit(main())
>              ^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/cli/main.py", line 75, in main
>     args.dispatch_function(args)
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/cli/serve.py", line 33, in cmd
>     uvloop.run(run_server(args))
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/uvloop/__init__.py", line 109, in run
>     return __asyncio.run(
>            ^^^^^^^^^^^^^^
>   File "/usr/lib/python3.12/asyncio/runners.py", line 194, in run
>     return runner.run(main)
>            ^^^^^^^^^^^^^^^^
>   File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
>     return self._loop.run_until_complete(task)
>            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/uvloop/__init__.py", line 61, in wrapper
>     return await main
>            ^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 1012, in run_server
>     async with build_async_engine_client(args) as engine_client:
>   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
>     return await anext(self.gen)
>            ^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 141, in build_async_engine_client
>     async with build_async_engine_client_from_engine_args(
>   File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
>     return await anext(self.gen)
>            ^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/entrypoints/openai/api_server.py", line 161, in build_async_engine_client_from_engine_args
>     vllm_config = engine_args.create_engine_config(usage_context=usage_context)
>                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1206, in create_engine_config
>     model_config = self.create_model_config()
>                    ^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1121, in create_model_config
>     return ModelConfig(
>            ^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/config.py", line 333, in __init__
>     hf_config = get_config(self.hf_config_path or self.model,
>                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/vllm/transformers_utils/config.py", line 280, in get_config
>     config_dict, _ = PretrainedConfig.get_config_dict(
>                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/configuration_utils.py", line 594, in get_config_dict
>     config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
>                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/configuration_utils.py", line 685, in _get_config_dict
>     config_dict = load_gguf_checkpoint(resolved_config_file, return_tensors=False)["config"]
>                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
>   File "/root/.local/share/pipx/venvs/vllm-v0-8-0/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py", line 399, in load_gguf_checkpoint
>     raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
> ValueError: GGUF model with architecture deepseek2 is not supported yet.
> ```

@SzymonOzog any ideas? Already passing the HF config files
---
### SzymonOzog @ 2025-04-02T07:30:20Z

@joshuakoh1 
How are you passing the hf_config_path variable? It's set to none in your logs
---
### lv03 @ 2025-04-16T06:31:06Z

@SzymonOzog hello， I encountered some issues while loading DeepSeeker R1-UD-IQ1_S

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ./merged_file.gguf --tokenizer ../config_file/ --hf-config-path ../config_file/ --tensor-parallel-size 8 --max-model-len 102400 --gpu-memory-utilization 0.5 --port 8000 --dtype auto

It seems to be stuck here 
![image](https://github.com/user-attachments/assets/4eec465d-fc66-4fd0-8e2d-ba15b835a7ff)


![image](https://github.com/user-attachments/assets/efe384c4-82ca-4662-ab9d-b67af92d2600)

+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.04             Driver Version: 570.124.04     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        Off |   00000000:17:00.0 Off |                  Off |
| 49%   37C    P0             78W /  450W |   33746MiB /  49140MiB |     36%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 4090        Off |   00000000:3D:00.0 Off |                  Off |
| 62%   36C    P0             84W /  450W |   33746MiB /  49140MiB |     36%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA GeForce RTX 4090        Off |   00000000:63:00.0 Off |                  Off |
| 64%   36C    P0             81W /  450W |   33746MiB /  49140MiB |     34%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA GeForce RTX 4090        Off |   00000000:99:00.0 Off |                  Off |
| 62%   38C    P0             94W /  450W |   33746MiB /  49140MiB |     33%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA A100 80GB PCIe          Off |   00000000:AB:00.0 Off |                    0 |
| N/A   43C    P0             79W /  300W |   33795MiB /  81920MiB |     48%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA GeForce RTX 4090        Off |   00000000:BD:00.0 Off |                  Off |
| 62%   36C    P0             89W /  450W |   33746MiB /  49140MiB |     37%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA GeForce RTX 4090        Off |   00000000:CF:00.0 Off |                  Off |
| 65%   40C    P0             79W /  450W |   33746MiB /  49140MiB |     37%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA A100 80GB PCIe          Off |   00000000:E1:00.0 Off |                    0 |
| N/A   44C    P0             85W /  300W |   33795MiB /  81920MiB |     49%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          124582      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    1   N/A  N/A          124955      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    2   N/A  N/A          124956      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    3   N/A  N/A          124957      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    4   N/A  N/A          124958      C   ...niconda3/envs/vLLM/bin/python      33786MiB |
|    5   N/A  N/A          124959      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    6   N/A  N/A          124960      C   ...niconda3/envs/vLLM/bin/python      33736MiB |
|    7   N/A  N/A          124961      C   ...niconda3/envs/vLLM/bin/python      33786MiB |
+-----------------------------------------------------------------------------------------+

---
### zhaotyer @ 2025-04-17T10:17:27Z

> @SzymonOzog hello， I encountered some issues while loading DeepSeeker R1-UD-IQ1_S
> 
> CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve ./merged_file.gguf --tokenizer ../config_file/ --hf-config-path ../config_file/ --tensor-parallel-size 8 --max-model-len 102400 --gpu-memory-utilization 0.5 --port 8000 --dtype auto
> 
> It seems to be stuck here ![image](https://private-user-images.githubusercontent.com/104058663/434183249-4eec465d-fc66-4fd0-8e2d-ba15b835a7ff.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ4ODQzNjQsIm5iZiI6MTc0NDg4NDA2NCwicGF0aCI6Ii8xMDQwNTg2NjMvNDM0MTgzMjQ5LTRlZWM0NjVkLWZjNjYtNGZkMC04ZTJkLWJhMTViODM1YTdmZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxN1QxMDAxMDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lMTljYjAwYTZlMjQ3ZjkzNDAyYzJkOWNjY2I4ZDA2ZTNkZDJjYmU2NzE1YzY1OTFjMjA0ZWIyMDVhMmIwYTZiJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.mVkczqFTMLBXuYi-MykCpa1vhU4bwlRleTdYqaLnjYk)
> 
> ![image](https://private-user-images.githubusercontent.com/104058663/434183848-efe384c4-82ca-4662-ab9d-b67af92d2600.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ4ODQzNjQsIm5iZiI6MTc0NDg4NDA2NCwicGF0aCI6Ii8xMDQwNTg2NjMvNDM0MTgzODQ4LWVmZTM4NGM0LTgyY2EtNDY2Mi1hYjlkLWI2N2FmOTJkMjYwMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwNDE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDQxN1QxMDAxMDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zNGNlZWU2OThhYjdlOGI3MDYxOTk1MjQ5NThjOGE4ZGE4ZDhhNDRiYzVlYWU1ODZjZDk3NDNhNWY4MGI5Njg2JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.G5D43yU0NjV0IK3J3zBf-57KLO0q9k1r6gP6EwBc7K0)
> 
> +-----------------------------------------------------------------------------------------+ | NVIDIA-SMI 570.124.04 Driver Version: 570.124.04 CUDA Version: 12.8 | |-----------------------------------------+------------------------+----------------------+ | GPU Name Persistence-M | Bus-Id Disp.A | Volatile Uncorr. ECC | | Fan Temp Perf Pwr:Usage/Cap | Memory-Usage | GPU-Util Compute M. | | | | MIG M. | |=========================================+========================+======================| | 0 NVIDIA GeForce RTX 4090 Off | 00000000:17:00.0 Off | Off | | 49% 37C P0 78W / 450W | 33746MiB / 49140MiB | 36% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 1 NVIDIA GeForce RTX 4090 Off | 00000000:3D:00.0 Off | Off | | 62% 36C P0 84W / 450W | 33746MiB / 49140MiB | 36% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 2 NVIDIA GeForce RTX 4090 Off | 00000000:63:00.0 Off | Off | | 64% 36C P0 81W / 450W | 33746MiB / 49140MiB | 34% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 3 NVIDIA GeForce RTX 4090 Off | 00000000:99:00.0 Off | Off | | 62% 38C P0 94W / 450W | 33746MiB / 49140MiB | 33% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 4 NVIDIA A100 80GB PCIe Off | 00000000:AB:00.0 Off | 0 | | N/A 43C P0 79W / 300W | 33795MiB / 81920MiB | 48% Default | | | | Disabled | +-----------------------------------------+------------------------+----------------------+ | 5 NVIDIA GeForce RTX 4090 Off | 00000000:BD:00.0 Off | Off | | 62% 36C P0 89W / 450W | 33746MiB / 49140MiB | 37% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 6 NVIDIA GeForce RTX 4090 Off | 00000000:CF:00.0 Off | Off | | 65% 40C P0 79W / 450W | 33746MiB / 49140MiB | 37% Default | | | | N/A | +-----------------------------------------+------------------------+----------------------+ | 7 NVIDIA A100 80GB PCIe Off | 00000000:E1:00.0 Off | 0 | | N/A 44C P0 85W / 300W | 33795MiB / 81920MiB | 49% Default | | | | Disabled | +-----------------------------------------+------------------------+----------------------+
> 
> +-----------------------------------------------------------------------------------------+ | Processes: | | GPU GI CI PID Type Process name GPU Memory | | ID ID Usage | |=========================================================================================| | 0 N/A N/A 124582 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 1 N/A N/A 124955 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 2 N/A N/A 124956 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 3 N/A N/A 124957 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 4 N/A N/A 124958 C ...niconda3/envs/vLLM/bin/python 33786MiB | | 5 N/A N/A 124959 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 6 N/A N/A 124960 C ...niconda3/envs/vLLM/bin/python 33736MiB | | 7 N/A N/A 124961 C ...niconda3/envs/vLLM/bin/python 33786MiB | +-----------------------------------------------------------------------------------------+

me too, Have you solved this problem?
---
### zhaotyer @ 2025-04-17T10:20:31Z

> @joshuakoh1 How are you passing the hf_config_path variable? It's set to none in your logs

![1744885087776](https://github.com/user-attachments/assets/86014004-0c7c-4cf2-8d37-e3739aa74d28)
vllm stuck at `There is no support for fast MoE kernel for current quantization method. Falling back to slow implementation.`
vllm version is: 0.8.4
---
### SzymonOzog @ 2025-04-17T10:46:48Z

I think this is happening because I quants use a very slow implementation of moe and for sequence length of 102400 it is processing for a long time, https://github.com/vllm-project/vllm/pull/16780 should add better support for MoE I quants


---
### zhaotyer @ 2025-04-17T11:07:33Z

> > @joshuakoh1 How are you passing the hf_config_path variable? It's set to none in your logs
> 
> ![1744885087776](https://private-user-images.githubusercontent.com/89376832/434752170-86014004-0c7c-4cf2-8d37-e3739aa74d28.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDQ4ODgyNDgsIm5iZiI6MTc0NDg4Nzk0OCwicGF0aCI6Ii84OTM3NjgzMi80MzQ3NTIxNzAtODYwMTQwMDQtMGM3Yy00Y2YyLThkMzctZTM3MzlhYTc0ZDI4LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTA0MTclMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUwNDE3VDExMDU0OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTkzZmE2MTE3NzhhZWQyZGUwOGVhZDIyMTY5YmU4NTBhODI3MTdmZWI3M2NkNGU3Y2RkYWYzYTMyZjk4MDc1NTgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.isifMfixrzRFizzdyFZ2VWy6X7Iqt8-XpEqt3gv7joo) vllm stuck at `There is no support for fast MoE kernel for current quantization method. Falling back to slow implementation.` vllm version is: 0.8.4

@SzymonOzog Is there a solution to this problem?
---
### SzymonOzog @ 2025-04-17T11:15:56Z

Yes, the PR I mentioned in the comment above should speed up I quants, that might reslove the issue 
---
### zhaotyer @ 2025-04-17T11:30:08Z

> Yes, the PR I mentioned in the comment above should speed up I quants, that might reslove the issue

Thank you for your reply, looking forward to merging the code
---
### lv03 @ 2025-04-17T12:41:51Z

> I think this is happening because I quants use a very slow implementation of moe and for sequence length of 102400 it is processing for a long time, #16780 should add better support for MoE I quants我认为发生这种情况是因为 I 量化使用非常慢的 moe 实现，并且对于序列长度为 102400，它需要处理很长时间， #16780 应该增加对 MoE I 量化的更好支持

After loading, the following problem occurred，
![image](https://github.com/user-attachments/assets/5e88f577-582f-4037-a780-5731ffd69262)

I saw someone reported this issue before. If don't change I quants，How should I deal with this problem?
<img width="939" alt="image" src="https://github.com/user-attachments/assets/5f8955aa-b9a4-486a-a30f-210547334a32" />

---
### SzymonOzog @ 2025-04-17T12:52:58Z

For now you can run with `enforce_eager=True` although it will be slow, the PR I mentioned above should also fix this issue

---
### lv03 @ 2025-04-17T12:57:52Z

> For now you can run with `enforce_eager=True` although it will be slow, the PR I mentioned above should also fix this issue现在你可以用 `enforce_eager=True` 运行，虽然它会很慢，但我上面提到的 PR 也应该可以解决这个问题

thank you，Will the next version of vllm solve this problem?
---
### SzymonOzog @ 2025-04-17T18:10:15Z

That depends on when the PR will get merged onto main
---
### zhaotyer @ 2025-04-22T05:03:28Z

> Yes, the PR I mentioned in the comment above should speed up I quants, that might reslove the issue

Hello, could you provide a docker images url? The network here is not good and docker build always fails
---
### zhaotyer @ 2025-04-22T11:23:10Z

> I think this is happening because I quants use a very slow implementation of moe and for sequence length of 102400 it is processing for a long time, #16780 should add better support for MoE I quants

when I set max_model_len to 8192, The service will crash when it start
I tested it on 2xA100x80GB and 8xL40Sx45GB, and both showed errors.
```
vllm serve /models/DeepSeek-R1-UD-IQ1_S/merged_file.gguf -tp 2 --trust-remote-code --enforce-eager --trust-remote-code --tokenizer /models/DeepSeek-R1-UD-IQ1_S/ --hf-config-path /models/DeepSeek-R1-UD-IQ1_S/ --dtype bfloat16 --max-model-len 8192 --served-model-name atom --port 8160 --gpu-memory-utilization 0.95
```
error log
```
INFO 04-22 04:13:35 [model_runner.py:1146] Model loading took 66.5477 GiB and 216.481170 seconds
ERROR 04-22 04:13:40 [engine.py:448] CUDA error: invalid configuration argument
ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
ERROR 04-22 04:13:40 [engine.py:448] Traceback (most recent call last):
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 436, in run_mp_engine
ERROR 04-22 04:13:40 [engine.py:448]     engine = MQLLMEngine.from_vllm_config(
ERROR 04-22 04:13:40 [engine.py:448]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 128, in from_vllm_config
ERROR 04-22 04:13:40 [engine.py:448]     return cls(
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 82, in __init__
ERROR 04-22 04:13:40 [engine.py:448]     self.engine = LLMEngine(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 285, in __init__
ERROR 04-22 04:13:40 [engine.py:448]     self._initialize_kv_caches()
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 434, in _initialize_kv_caches
ERROR 04-22 04:13:40 [engine.py:448]     self.model_executor.determine_num_available_blocks())
ERROR 04-22 04:13:40 [engine.py:448]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 103, in determine_num_available_blocks
ERROR 04-22 04:13:40 [engine.py:448]     results = self.collective_rpc("determine_num_available_blocks")
ERROR 04-22 04:13:40 [engine.py:448]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 331, in collective_rpc
ERROR 04-22 04:13:40 [engine.py:448]     return self._run_workers(method, *args, **(kwargs or {}))
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/mp_distributed_executor.py", line 185, in _run_workers
ERROR 04-22 04:13:40 [engine.py:448]     driver_worker_output = run_method(self.driver_worker, sent_method,
ERROR 04-22 04:13:40 [engine.py:448]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2428, in run_method
ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 229, in determine_num_available_blocks
ERROR 04-22 04:13:40 [engine.py:448]     self.model_runner.profile_run()
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1243, in profile_run
ERROR 04-22 04:13:40 [engine.py:448]     self._dummy_run(max_num_batched_tokens, max_num_seqs)
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1369, in _dummy_run
ERROR 04-22 04:13:40 [engine.py:448]     self.execute_model(model_input, kv_caches, intermediate_tensors)
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1770, in execute_model
ERROR 04-22 04:13:40 [engine.py:448]     hidden_or_intermediate_states = model_executable(
ERROR 04-22 04:13:40 [engine.py:448]                                     ^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 703, in forward
ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.model(input_ids, positions, intermediate_tensors,
ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 172, in __call__
ERROR 04-22 04:13:40 [engine.py:448]     return self.forward(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 660, in forward
ERROR 04-22 04:13:40 [engine.py:448]     hidden_states, residual = layer(positions, hidden_states, residual)
ERROR 04-22 04:13:40 [engine.py:448]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 580, in forward
ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.mlp(hidden_states)
ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 159, in forward
ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.experts(
ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 842, in forward
ERROR 04-22 04:13:40 [engine.py:448]     return self.forward_impl(hidden_states, router_logits)
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 861, in forward_impl
ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.quant_method.apply(
ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 377, in apply
ERROR 04-22 04:13:40 [engine.py:448]     return _fused_moe_gguf(x, layer.w13_qweight, layer.w2_qweight,
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 176, in _fused_moe_gguf
ERROR 04-22 04:13:40 [engine.py:448]     out = ops.ggml_moe_a8_vec(out, w2, topk_ids, 1, qweight_type2,
ERROR 04-22 04:13:40 [engine.py:448]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py", line 1179, in ggml_moe_a8_vec
ERROR 04-22 04:13:40 [engine.py:448]     return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1123, in __call__
ERROR 04-22 04:13:40 [engine.py:448]     return self._op(*args, **(kwargs or {}))
ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 04-22 04:13:40 [engine.py:448] RuntimeError: CUDA error: invalid configuration argument
ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

```
---
### zhaotyer @ 2025-04-22T12:26:34Z

> > I think this is happening because I quants use a very slow implementation of moe and for sequence length of 102400 it is processing for a long time, #16780 should add better support for MoE I quants
> 
> when I set max_model_len to 8192, The service will crash when it start I tested it on 2xA100x80GB and 8xL40Sx45GB, and both showed errors.
> 
> ```
> vllm serve /models/DeepSeek-R1-UD-IQ1_S/merged_file.gguf -tp 2 --trust-remote-code --enforce-eager --trust-remote-code --tokenizer /models/DeepSeek-R1-UD-IQ1_S/ --hf-config-path /models/DeepSeek-R1-UD-IQ1_S/ --dtype bfloat16 --max-model-len 8192 --served-model-name atom --port 8160 --gpu-memory-utilization 0.95
> ```
> 
> error log
> 
> ```
> INFO 04-22 04:13:35 [model_runner.py:1146] Model loading took 66.5477 GiB and 216.481170 seconds
> ERROR 04-22 04:13:40 [engine.py:448] CUDA error: invalid configuration argument
> ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> ERROR 04-22 04:13:40 [engine.py:448] Traceback (most recent call last):
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 436, in run_mp_engine
> ERROR 04-22 04:13:40 [engine.py:448]     engine = MQLLMEngine.from_vllm_config(
> ERROR 04-22 04:13:40 [engine.py:448]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 128, in from_vllm_config
> ERROR 04-22 04:13:40 [engine.py:448]     return cls(
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 82, in __init__
> ERROR 04-22 04:13:40 [engine.py:448]     self.engine = LLMEngine(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 285, in __init__
> ERROR 04-22 04:13:40 [engine.py:448]     self._initialize_kv_caches()
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 434, in _initialize_kv_caches
> ERROR 04-22 04:13:40 [engine.py:448]     self.model_executor.determine_num_available_blocks())
> ERROR 04-22 04:13:40 [engine.py:448]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 103, in determine_num_available_blocks
> ERROR 04-22 04:13:40 [engine.py:448]     results = self.collective_rpc("determine_num_available_blocks")
> ERROR 04-22 04:13:40 [engine.py:448]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 331, in collective_rpc
> ERROR 04-22 04:13:40 [engine.py:448]     return self._run_workers(method, *args, **(kwargs or {}))
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/mp_distributed_executor.py", line 185, in _run_workers
> ERROR 04-22 04:13:40 [engine.py:448]     driver_worker_output = run_method(self.driver_worker, sent_method,
> ERROR 04-22 04:13:40 [engine.py:448]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2428, in run_method
> ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 229, in determine_num_available_blocks
> ERROR 04-22 04:13:40 [engine.py:448]     self.model_runner.profile_run()
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1243, in profile_run
> ERROR 04-22 04:13:40 [engine.py:448]     self._dummy_run(max_num_batched_tokens, max_num_seqs)
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1369, in _dummy_run
> ERROR 04-22 04:13:40 [engine.py:448]     self.execute_model(model_input, kv_caches, intermediate_tensors)
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1770, in execute_model
> ERROR 04-22 04:13:40 [engine.py:448]     hidden_or_intermediate_states = model_executable(
> ERROR 04-22 04:13:40 [engine.py:448]                                     ^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 703, in forward
> ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.model(input_ids, positions, intermediate_tensors,
> ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 172, in __call__
> ERROR 04-22 04:13:40 [engine.py:448]     return self.forward(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 660, in forward
> ERROR 04-22 04:13:40 [engine.py:448]     hidden_states, residual = layer(positions, hidden_states, residual)
> ERROR 04-22 04:13:40 [engine.py:448]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 580, in forward
> ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.mlp(hidden_states)
> ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 159, in forward
> ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.experts(
> ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 842, in forward
> ERROR 04-22 04:13:40 [engine.py:448]     return self.forward_impl(hidden_states, router_logits)
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 861, in forward_impl
> ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.quant_method.apply(
> ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 377, in apply
> ERROR 04-22 04:13:40 [engine.py:448]     return _fused_moe_gguf(x, layer.w13_qweight, layer.w2_qweight,
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 176, in _fused_moe_gguf
> ERROR 04-22 04:13:40 [engine.py:448]     out = ops.ggml_moe_a8_vec(out, w2, topk_ids, 1, qweight_type2,
> ERROR 04-22 04:13:40 [engine.py:448]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py", line 1179, in ggml_moe_a8_vec
> ERROR 04-22 04:13:40 [engine.py:448]     return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1123, in __call__
> ERROR 04-22 04:13:40 [engine.py:448]     return self._op(*args, **(kwargs or {}))
> ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> ERROR 04-22 04:13:40 [engine.py:448] RuntimeError: CUDA error: invalid configuration argument
> ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> ```

when I set max_model_len to 8192,The specific parameter information when the following command reports an error is as follows
```
    return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
                                        tokens)
```
```
(VllmWorkerProcess pid=3264) INFO 04-22 05:19:17 [model_runner.py:1146] Model loading took 66.5477 GiB and 241.765538 seconds
INFO 04-22 05:19:32 [model_runner.py:1146] Model loading took 66.5477 GiB and 257.481310 seconds
ERROR 04-22 05:19:37 [_custom_ops.py:1179] x is:torch.Size([8192, 7168]), w is:torch.Size([256, 2048, 1400]), topk_ids is:torch.Size([8192, 8]),top_k is:8, quant_type is:19, row is:<class 'torch.SymInt'>, tokens is:8192
ERROR 04-22 05:19:37 [_custom_ops.py:1179] x is:torch.Size([65536, 1024]), w is:torch.Size([256, 7168, 264]), topk_ids is:torch.Size([8192, 8]),top_k is:1, quant_type is:16, row is:<class 'torch.SymInt'>, tokens is:65536
ERROR 04-22 05:19:37 [engine.py:448] CUDA error: invalid configuration argument
ERROR 04-22 05:19:37 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
ERROR 04-22 05:19:37 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
ERROR 04-22 05:19:37 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
ERROR 04-22 05:19:37 [engine.py:448] Traceback (most recent call last):
ERROR 04-22 0
```
@SzymonOzog 

---
### hahmad2008 @ 2025-05-05T15:01:45Z

@SzymonOzog 
can we serve this model using vllm?
`NoelJacob/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF `
---
### ChuanhongLi @ 2025-06-03T09:49:15Z

@SzymonOzog  The new DeepSeek-R1-0528-UD-Q2_K_XL gguf files have removed blk.0.attn_kv_b.weight and added blk.0.attn_k_b.weight and blk.0.attn_v_b.weight. This change prevents us from loading the model correctly. How can we address this issue?
---
### SzymonOzog @ 2025-06-03T09:58:12Z

@ChuanhongLi I think you should be able to get around it by modifying https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/vllm/model_executor/model_loader/gguf_loader.py#L37 to point the layers to correct places
---
### ChuanhongLi @ 2025-06-06T02:42:54Z

> @ChuanhongLi I think you should be able to get around it by modifying
> 
> https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/vllm/model_executor/model_loader/gguf_loader.py#L37
> 
> to point the layers to correct places

Thanks for your reply, but the problem may come from kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T (line 703, vllm/v1/attention/backends/mla/common.py) https://github.com/vllm-project/vllm/issues/19050#issuecomment-2938613467
---
### ChuanhongLi @ 2025-06-10T12:05:02Z

> > @ChuanhongLi I think you should be able to get around it by modifying
> > https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/vllm/model_executor/model_loader/gguf_loader.py#L37
> > 
> > to point the layers to correct places
> 
> Thanks for your reply, but the problem may come from kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T (line 703, vllm/v1/attention/backends/mla/common.py) [#19050 (comment)](https://github.com/vllm-project/vllm/issues/19050#issuecomment-2938613467)

@SzymonOzog  Hi, do you have any idea about how to fix it?  Should we  load the attn_k_b and attn_v_b to produce the kv_b weight?  Approcaite for your help, thanks! 
---
### maekawatoshiki @ 2025-06-12T01:41:37Z

I’m facing the same problem. I made a few changes and finally got vLLM to start, but the output is gibberish. Has anyone figured out a solution?
---
### SzymonOzog @ 2025-06-12T09:17:09Z

As of now I don't have a ready solution for ths. I'll try to find some time to debug the issue over the weekend
---
### hyunwen @ 2025-06-23T06:43:38Z

Any progress here? We are also stuck here.
---
### kechengcode @ 2025-07-01T06:05:57Z

> > > I think this is happening because I quants use a very slow implementation of moe and for sequence length of 102400 it is processing for a long time, #16780 should add better support for MoE I quants
> > 
> > 
> > when I set max_model_len to 8192, The service will crash when it start I tested it on 2xA100x80GB and 8xL40Sx45GB, and both showed errors.
> > ```
> > vllm serve /models/DeepSeek-R1-UD-IQ1_S/merged_file.gguf -tp 2 --trust-remote-code --enforce-eager --trust-remote-code --tokenizer /models/DeepSeek-R1-UD-IQ1_S/ --hf-config-path /models/DeepSeek-R1-UD-IQ1_S/ --dtype bfloat16 --max-model-len 8192 --served-model-name atom --port 8160 --gpu-memory-utilization 0.95
> > ```
> > 
> > 
> >     
> >       
> >     
> > 
> >       
> >     
> > 
> >     
> >   
> > error log
> > ```
> > INFO 04-22 04:13:35 [model_runner.py:1146] Model loading took 66.5477 GiB and 216.481170 seconds
> > ERROR 04-22 04:13:40 [engine.py:448] CUDA error: invalid configuration argument
> > ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> > ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> > ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> > ERROR 04-22 04:13:40 [engine.py:448] Traceback (most recent call last):
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 436, in run_mp_engine
> > ERROR 04-22 04:13:40 [engine.py:448]     engine = MQLLMEngine.from_vllm_config(
> > ERROR 04-22 04:13:40 [engine.py:448]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 128, in from_vllm_config
> > ERROR 04-22 04:13:40 [engine.py:448]     return cls(
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 82, in __init__
> > ERROR 04-22 04:13:40 [engine.py:448]     self.engine = LLMEngine(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 285, in __init__
> > ERROR 04-22 04:13:40 [engine.py:448]     self._initialize_kv_caches()
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 434, in _initialize_kv_caches
> > ERROR 04-22 04:13:40 [engine.py:448]     self.model_executor.determine_num_available_blocks())
> > ERROR 04-22 04:13:40 [engine.py:448]     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 103, in determine_num_available_blocks
> > ERROR 04-22 04:13:40 [engine.py:448]     results = self.collective_rpc("determine_num_available_blocks")
> > ERROR 04-22 04:13:40 [engine.py:448]               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 331, in collective_rpc
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._run_workers(method, *args, **(kwargs or {}))
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/mp_distributed_executor.py", line 185, in _run_workers
> > ERROR 04-22 04:13:40 [engine.py:448]     driver_worker_output = run_method(self.driver_worker, sent_method,
> > ERROR 04-22 04:13:40 [engine.py:448]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils.py", line 2428, in run_method
> > ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> > ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 229, in determine_num_available_blocks
> > ERROR 04-22 04:13:40 [engine.py:448]     self.model_runner.profile_run()
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> > ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1243, in profile_run
> > ERROR 04-22 04:13:40 [engine.py:448]     self._dummy_run(max_num_batched_tokens, max_num_seqs)
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1369, in _dummy_run
> > ERROR 04-22 04:13:40 [engine.py:448]     self.execute_model(model_input, kv_caches, intermediate_tensors)
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 116, in decorate_context
> > ERROR 04-22 04:13:40 [engine.py:448]     return func(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1770, in execute_model
> > ERROR 04-22 04:13:40 [engine.py:448]     hidden_or_intermediate_states = model_executable(
> > ERROR 04-22 04:13:40 [engine.py:448]                                     ^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 703, in forward
> > ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.model(input_ids, positions, intermediate_tensors,
> > ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 172, in __call__
> > ERROR 04-22 04:13:40 [engine.py:448]     return self.forward(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 660, in forward
> > ERROR 04-22 04:13:40 [engine.py:448]     hidden_states, residual = layer(positions, hidden_states, residual)
> > ERROR 04-22 04:13:40 [engine.py:448]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 580, in forward
> > ERROR 04-22 04:13:40 [engine.py:448]     hidden_states = self.mlp(hidden_states)
> > ERROR 04-22 04:13:40 [engine.py:448]                     ^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py", line 159, in forward
> > ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.experts(
> > ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._call_impl(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     return forward_call(*args, **kwargs)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 842, in forward
> > ERROR 04-22 04:13:40 [engine.py:448]     return self.forward_impl(hidden_states, router_logits)
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 861, in forward_impl
> > ERROR 04-22 04:13:40 [engine.py:448]     final_hidden_states = self.quant_method.apply(
> > ERROR 04-22 04:13:40 [engine.py:448]                           ^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 377, in apply
> > ERROR 04-22 04:13:40 [engine.py:448]     return _fused_moe_gguf(x, layer.w13_qweight, layer.w2_qweight,
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py", line 176, in _fused_moe_gguf
> > ERROR 04-22 04:13:40 [engine.py:448]     out = ops.ggml_moe_a8_vec(out, w2, topk_ids, 1, qweight_type2,
> > ERROR 04-22 04:13:40 [engine.py:448]           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/vllm/_custom_ops.py", line 1179, in ggml_moe_a8_vec
> > ERROR 04-22 04:13:40 [engine.py:448]     return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448]   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1123, in __call__
> > ERROR 04-22 04:13:40 [engine.py:448]     return self._op(*args, **(kwargs or {}))
> > ERROR 04-22 04:13:40 [engine.py:448]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
> > ERROR 04-22 04:13:40 [engine.py:448] RuntimeError: CUDA error: invalid configuration argument
> > ERROR 04-22 04:13:40 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> > ERROR 04-22 04:13:40 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> > ERROR 04-22 04:13:40 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> > ```
> 
> when I set max_model_len to 8192,The specific parameter information when the following command reports an error is as follows
> 
> ```
>     return torch.ops._C.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row,
>                                         tokens)
> ```
> 
> ```
> (VllmWorkerProcess pid=3264) INFO 04-22 05:19:17 [model_runner.py:1146] Model loading took 66.5477 GiB and 241.765538 seconds
> INFO 04-22 05:19:32 [model_runner.py:1146] Model loading took 66.5477 GiB and 257.481310 seconds
> ERROR 04-22 05:19:37 [_custom_ops.py:1179] x is:torch.Size([8192, 7168]), w is:torch.Size([256, 2048, 1400]), topk_ids is:torch.Size([8192, 8]),top_k is:8, quant_type is:19, row is:<class 'torch.SymInt'>, tokens is:8192
> ERROR 04-22 05:19:37 [_custom_ops.py:1179] x is:torch.Size([65536, 1024]), w is:torch.Size([256, 7168, 264]), topk_ids is:torch.Size([8192, 8]),top_k is:1, quant_type is:16, row is:<class 'torch.SymInt'>, tokens is:65536
> ERROR 04-22 05:19:37 [engine.py:448] CUDA error: invalid configuration argument
> ERROR 04-22 05:19:37 [engine.py:448] CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
> ERROR 04-22 05:19:37 [engine.py:448] For debugging consider passing CUDA_LAUNCH_BLOCKING=1
> ERROR 04-22 05:19:37 [engine.py:448] Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
> ERROR 04-22 05:19:37 [engine.py:448] Traceback (most recent call last):
> ERROR 04-22 0
> ```
> 
> @SzymonOzog

I meet the same problme
---