# PR #21126: [Perf] Use FlashInfer RoPE for RotaryEmbedding.forward_cuda when available

- field: attention  | area: model_executor/layers
- author: mgoin
- state: MERGED  created: 2025-07-17T16:12:23Z  merged: 2025-09-19T20:06:49Z
- url: https://github.com/vllm-project/vllm/pull/21126
- labels: ready

## Body

## Purpose

Enables FlashInfer RoPE for the cuda forward impl when custom op is set. 
Microbenchmarks show ~2x speedup vs our custom op, but haven't seen noticeable end-to-end results yet.

## Test Plan

lm-eval with llama 8b

## Test Result

I found no noticeable performance difference so leaving this not enabled by default. Looking at the profiles, it seems like we can possibly introduce more  If we want to enable it by default, this is the change to `VllmConfig::__post_init__`
```python
        # Enable rope custom op by default if nothing is specified
        if not any(op in ["-rotary_embedding", "none"]
                   for op in self.compilation_config.custom_ops):
            self.compilation_config.custom_ops.append("+rotary_embedding")
            logger.debug_once("Enabled rotary_embedding custom op by default")
```

lm-eval results:
```
# Torch Native
Processed prompts: 100%|█████| 1319/1319 [00:37<00:00, 35.07it/s, est. speed input: 30589.23 toks/s, output: 3404.12 toks/s]
Running generate_until requests: 100%|█████████| 1319/1319 [00:37<00:00, 34.97it/s]
2025-09-16:23:51:38 INFO     [loggers.evaluation_tracker:280] Output path not provided, skipping saving results aggregated
vllm (pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7854|±  |0.0113|
|     |       |strict-match    |     5|exact_match|↑  |0.7582|±  |0.0118|

# vLLM Custom Op
Processed prompts: 100%|█████| 1319/1319 [00:38<00:00, 34.66it/s, est. speed input: 30228.40 toks/s, output: 3371.77 toks/s]
Running generate_until requests: 100%|█████████| 1319/1319 [00:38<00:00, 34.56it/s]
2025-09-16:23:55:59 INFO     [loggers.evaluation_tracker:280] Output path not provided, skipping saving results aggregated
vllm (pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7786|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7551|±  |0.0118|

# FlashInfer
Processed prompts: 100%|█████| 1319/1319 [00:39<00:00, 33.54it/s, est. speed input: 29252.70 toks/s, output: 3261.28 toks/s]
Running generate_until requests: 100%|█████████| 1319/1319 [00:39<00:00, 33.45it/s]
2025-09-16:23:49:42 INFO     [loggers.evaluation_tracker:280] Output path not provided, skipping saving results aggregated
vllm (pretrained=meta-llama/Llama-3.1-8B-Instruct,trust_remote_code=True), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7817|±  |0.0114|
|     |       |strict-match    |     5|exact_match|↑  |0.7619|±  |0.0117|
```

## (Optional) Documentation Update



## Comments

### github-actions @ 2025-07-17T16:12:41Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### mergify @ 2025-09-15T14:54:41Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @mgoin.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mgoin @ 2025-09-18T15:28:04Z

@ProExpertProg I didn't see any performance improvements, only very small regressions, so I'm punting the enablement. We can still just enable `+rotary_embedding` to use it
---
### mergify @ 2025-09-19T18:11:57Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @mgoin.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### huydhn @ 2025-09-20T06:26:39Z

@mgoin I'm seeing a failure coming out of this change when testing this against PyTorch main https://github.com/pytorch/pytorch/actions/runs/17874374399/job/50835265915#step:25:6075

```
pytest -v -s -x lora/test_llama_tp.py -k test_tp2_serialize_and_deserialize_lora
...
EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/v1/executor/multiproc_executor.py", line 666, in worker_busy_loop
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     output = func(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]              ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return func(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py", line 264, in determine_available_memory
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     self.model_runner.profile_run()
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 3292, in profile_run
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     = self._dummy_run(self.max_num_tokens, is_profile=True)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return func(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 3069, in _dummy_run
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     outputs = self.model(
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]               ^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/models/llama.py", line 597, in forward
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     model_output = self.model(input_ids, positions, intermediate_tensors,
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/compilation/decorators.py", line 223, in __call__
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self.forward(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/models/llama.py", line 405, in forward
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     hidden_states, residual = layer(positions, hidden_states, residual)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/models/llama.py", line 318, in forward
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     hidden_states = self.self_attn(positions=positions,
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/models/llama.py", line 215, in forward
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     q, k = self.rotary_emb(positions, q, k)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1775, in _wrapped_call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._call_impl(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1786, in _call_impl
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return forward_call(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/custom_op.py", line 44, in forward
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._forward_method(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/base.py", line 116, in forward_cuda
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     torch.ops.vllm.flashinfer_rotary_embedding(positions, query, key,
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/torch/_ops.py", line 1255, in __call__
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     return self._op(*args, **kwargs)
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]            ^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding/common.py", line 123, in _flashinfer_rotary_embedding
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     apply_rope_with_cos_sin_cache_inplace(
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]   File "/opt/conda/envs/py_3.12/lib/python3.12/site-packages/flashinfer/rope.py", line 1140, in apply_rope_with_cos_sin_cache_inplace
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671]     raise ValueError("cos_sin_cache should be float32")
(EngineCore_DP0 pid=20473) (Worker_TP0 pid=20479) ERROR 09-20 06:12:54 [multiproc_executor.py:671] ValueError: cos_sin_cache should be float32
```

The full log is at https://paste.sh/t1uRBOGf#sdNREw2PZgI1Xy1nJArnpFBr

Any thoughts?

cc @zou3519 @yeqcharlotte 


---