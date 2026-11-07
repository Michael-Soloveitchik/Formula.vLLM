# PR #30647: [Perf] Eliminate padding and slicing op for GPT-OSS with Flashinfer MXFP4 MXFP8 MoE

- field: quantization  | area: model_executor/models
- author: elvischenv
- state: CLOSED  created: 2025-12-14T13:40:05Z  merged: 2026-03-18T15:01:27Z
- url: https://github.com/vllm-project/vllm/pull/30647
- labels: ready, ci/build, gpt-oss, nvidia

## Body

<!-- markdownlint-disable -->

## Purpose

- Depends on Flashinfer update #30993
- Eliminated padding op before the MoE: by setting the alignment in flashinfer mxfp8 quant, the output quantized tensor will be padded.
- Eliminated slicing op after the MoE: by passing the output tensor with unpadded hidden size to MoE kernel, this depends on a Flashinfer PR:
  - https://github.com/flashinfer-ai/flashinfer/pull/2217
  - This will also resolve the previous AR+Norm fusion broken by slice op issue: #28841
- ~~Cleaned up the padding logic: for mxfp4 quant, the padded hidden size is calculated in `create_weights()`, the `maybe_roundup_hidden_size()` in `vllm/model_executor/layers/fused_moe/layer.py` seems like a dup.~~


## Test Plan && Test Result(GPT-OSS-120b TP8)

### Accuracy

PR:
```
[{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20251213_233136', 'metric': 0.7803030303030303}]
[{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20251213_234320', 'metric': 0.8875}]
```

main:
```
[{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20251214_002509', 'metric': 0.7891414141414141}]
[{'eval_name': 'aime25', 'model_name': 'gpt-oss-120b-high_temp1.0_20251214_001505', 'metric': 0.8875}]
```

### Kernel

PR:
```
void cublasLt::splitKreduce_kernel                          2.400 μs
void tensorrt_llm::kernels::quantize_with_block_size        2.944 μs
void moe::dev::routing::routingRenormalize                  5.216 μs
bmm_MxE4m3_MxE2m1MxE4m3_Fp32_t128x16x256u2                 13.216 μs
bmm_Bfloat16_MxE2m1MxE4m3_Fp32_t128x16x256u2                9.504 μs
void moe::dev::finalize::finalizeKernel                     3.168 μs
void flashinfer::trtllm_allreduce_fusion                    7.744 μs (ar+norm)
nvjet_tst_32x64_64x16_4x1_v_bz_splitK_TNN                   4.448 μs
```

main:
```
void cublasLt::splitKreduce_kernel                          2.048 μs
triton_poi_fused_constant_pad_nd_moe_forward_0              1.407 μs (pad)
void tensorrt_llm::kernels::quantize_with_block_size        2.432 μs
void moe::dev::routing::routingRenormalize                  5.056 μs
bmm_MxE4m3_MxE2m1MxE4m3_Fp32_t128x16x256u2                 10.368 μs
bmm_Bfloat16_MxE2m1MxE4m3_Fp32_t128x16x256u2                8.512 μs
void moe::dev::finalize::finalizeKernel                     2.112 μs
void vllm::cross_device_reduce_1stage                       8.320 μs (ar)
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_slice_1    2.336 μs (norm, slice)
nvjet_tst_32x64_64x16_4x1_v_bz_splitK_TNN                   4.448 μs
```

### Perf (GPT-OSS-120b TP8 con8)

PR: **5% E2E improvement**
```
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  28.90
Total input tokens:                      81920
Total generated tokens:                  81920
Request throughput (req/s):              2.77
Output token throughput (tok/s):         2834.13
Peak output token throughput (tok/s):    158.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          5668.25
---------------Time to First Token----------------
Mean TTFT (ms):                          50.37
Median TTFT (ms):                        58.07
P99 TTFT (ms):                           75.11
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.77
Median TPOT (ms):                        2.77
P99 TPOT (ms):                           2.84
---------------Inter-token Latency----------------
Mean ITL (ms):                           54.59
Median ITL (ms):                         55.28
P99 ITL (ms):                            68.89
==================================================
```

main:
```
============ Serving Benchmark Result ============
Successful requests:                     80
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  30.43
Total input tokens:                      81920
Total generated tokens:                  81920
Request throughput (req/s):              2.63
Output token throughput (tok/s):         2692.11
Peak output token throughput (tok/s):    149.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          5384.22
---------------Time to First Token----------------
Mean TTFT (ms):                          52.61
Median TTFT (ms):                        57.20
P99 TTFT (ms):                           88.23
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.92
Median TPOT (ms):                        2.92
P99 TPOT (ms):                           2.99
---------------Inter-token Latency----------------
Mean ITL (ms):                           57.47
Median ITL (ms):                         58.17
P99 ITL (ms):                            70.35
==================================================
```


---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mergify[bot] @ 2025-12-17T09:45:22Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @elvischenv.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

### mergify[bot] @ 2026-02-04T06:14:06Z

Hi @elvischenv, the pre-commit checks have failed. Please run:

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

### mergify[bot] @ 2026-02-05T00:56:09Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @elvischenv.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

### ProExpertProg @ 2026-02-08T14:14:23Z

@elvischenv could you post the new benchmarking numbers once you have them? 

### elvischenv @ 2026-02-09T03:29:00Z

> @elvischenv could you post the new benchmarking numbers once you have them?

This is the perf number based on main ToT:
PR
```
============ Serving Benchmark Result ============
Successful requests:                     40
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  15.67
Total input tokens:                      40960
Total generated tokens:                  40960
Request throughput (req/s):              2.55
Output token throughput (tok/s):         2613.70
Peak output token throughput (tok/s):    141.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          5227.39
---------------Time to First Token----------------
Mean TTFT (ms):                          61.13
Median TTFT (ms):                        52.89
P99 TTFT (ms):                           93.71
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3.00
Median TPOT (ms):                        3.00
P99 TPOT (ms):                           3.05
---------------Inter-token Latency----------------
Mean ITL (ms):                           59.05
Median ITL (ms):                         59.72
P99 ITL (ms):                            83.17
==================================================
```

main
```
============ Serving Benchmark Result ============
Successful requests:                     40
Failed requests:                         0
Maximum request concurrency:             8
Benchmark duration (s):                  16.46
Total input tokens:                      40960
Total generated tokens:                  40960
Request throughput (req/s):              2.43
Output token throughput (tok/s):         2488.50
Peak output token throughput (tok/s):    136.00
Peak concurrent requests:                16.00
Total token throughput (tok/s):          4977.00
---------------Time to First Token----------------
Mean TTFT (ms):                          64.64
Median TTFT (ms):                        64.21
P99 TTFT (ms):                           101.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          3.15
Median TPOT (ms):                        3.15
P99 TPOT (ms):                           3.22
---------------Inter-token Latency----------------
Mean ITL (ms):                           62.04
Median ITL (ms):                         63.02
P99 ITL (ms):                            64.60
==================================================
```

accuracy of PR
```
[{'eval_name': 'gpqa', 'model_name': 'gpt-oss-120b-high_temp1.0_20260208_183703', 'metric': 0.7992424242424242}]
```

### mergify[bot] @ 2026-02-10T17:23:55Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @elvischenv.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

### ProExpertProg @ 2026-03-16T22:30:39Z

Looks good to me but would want @mgoin or @robertgshaw2-redhat or @bnell to check the Moe code

### zyongye @ 2026-03-18T20:37:42Z

I am getting the error that the fusion pass trigger the assertion error when running `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 vllm serve openai/gpt-oss-120b -tp 2` on b200. only `--enforce-eager` works

```
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932] WorkerProc hit an exception.
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932] Traceback (most recent call last):
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/executor/multiproc_executor.py", line 927, in worker_busy_loop
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     output = func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_worker.py", line 388, in determine_available_memory
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.model_runner.profile_run()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_model_runner.py", line 5527, in profile_run
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     hidden_states, last_hidden_states = self._dummy_run(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                                         ^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_model_runner.py", line 5221, in _dummy_run
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     outputs = self.model(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]               ^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/cuda_graph.py", line 241, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.runnable(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._call_impl(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return forward_call(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/model_executor/models/gpt_oss.py", line 1218, in forward
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/decorators.py", line 583, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/wrapper.py", line 206, in aot_compile
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._compiled_callable.aot_compile((args, kwargs))
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 832, in aot_compile
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return aot_compile_fullgraph(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 239, in aot_compile_fullgraph
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     compiled_fn = backend(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                   ^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2509, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.compiler_fn(model_, inputs_, **self.kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/backends.py", line 1063, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.configure_post_pass()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/backends.py", line 847, in configure_post_pass
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.pass_manager.configure(self.vllm_config)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/pass_manager.py", line 125, in configure
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.passes += [AllReduceFusionPass(config)]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 791, in __init__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.register_patterns()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/inductor_pass.py", line 130, in fn_new
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     result = fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]              ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 817, in register_patterns
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     ).register(self.patterns)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]       ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 605, in register
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     pm.register_replacement(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1602, in register_replacement
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     pattern, gm = gen_pattern_and_search_gm(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1811, in gen_pattern_and_search_gm
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     search_gm = trace_fn(search_fn, flat_inputs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 2190, in fwd_only
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     gm = make_fx(fn, decompositions, tracing_mode="real")(*args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2701, in wrapped
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return make_fx_tracer.trace(f, *args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2626, in trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._trace_inner(f, *args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2588, in _trace_inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     t = dispatch_trace(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]         ^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_compile.py", line 54, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return disable_fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1460, in dispatch_trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 879, in trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     (self.create_arg(fn(*args)),),
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                      ^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1526, in wrapped
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     out = f(*tensors)  # type:ignore[call-arg]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]           ^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 563, in pattern
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     quant_out_tuple = auto_functionalized(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                       ^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/auto_functionalize.py", line 357, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     assert can_auto_functionalize(_mutable_op)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932] AssertionError
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932] Traceback (most recent call last):
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/executor/multiproc_executor.py", line 927, in worker_busy_loop
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     output = func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]              ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_worker.py", line 388, in determine_available_memory
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.model_runner.profile_run()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_model_runner.py", line 5527, in profile_run
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     hidden_states, last_hidden_states = self._dummy_run(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                                         ^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/v1/worker/gpu_model_runner.py", line 5221, in _dummy_run
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     outputs = self.model(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]               ^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/cuda_graph.py", line 241, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.runnable(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._call_impl(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _call_impl
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return forward_call(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/model_executor/models/gpt_oss.py", line 1218, in forward
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/decorators.py", line 583, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.aot_compiled_fn = self.aot_compile(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/wrapper.py", line 206, in aot_compile
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._compiled_callable.aot_compile((args, kwargs))
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 832, in aot_compile
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return aot_compile_fullgraph(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 239, in aot_compile_fullgraph
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     compiled_fn = backend(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                   ^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/__init__.py", line 2509, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self.compiler_fn(model_, inputs_, **self.kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/backends.py", line 1063, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.configure_post_pass()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/backends.py", line 847, in configure_post_pass
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.pass_manager.configure(self.vllm_config)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/pass_manager.py", line 125, in configure
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.passes += [AllReduceFusionPass(config)]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 791, in __init__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     self.register_patterns()
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/inductor_pass.py", line 130, in fn_new
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     result = fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]              ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 817, in register_patterns
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     ).register(self.patterns)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]       ^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 605, in register
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     pm.register_replacement(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1602, in register_replacement
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     pattern, gm = gen_pattern_and_search_gm(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/usr/lib/python3.12/contextlib.py", line 81, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwds)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 1811, in gen_pattern_and_search_gm
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     search_gm = trace_fn(search_fn, flat_inputs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 124, in decorate_context
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return func(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_inductor/pattern_matcher.py", line 2190, in fwd_only
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     gm = make_fx(fn, decompositions, tracing_mode="real")(*args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2701, in wrapped
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return make_fx_tracer.trace(f, *args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2626, in trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return self._trace_inner(f, *args)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2588, in _trace_inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     t = dispatch_trace(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]         ^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_compile.py", line 54, in inner
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return disable_fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1460, in dispatch_trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1181, in _fn
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     return fn(*args, **kwargs)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 879, in trace
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     (self.create_arg(fn(*args)),),
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                      ^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1526, in wrapped
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     out = f(*tensors)  # type:ignore[call-arg]
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]           ^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/vllm/compilation/passes/fusion/allreduce_rms_fusion.py", line 563, in pattern
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     quant_out_tuple = auto_functionalized(
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]                       ^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]   File "/home/yongye/vllm/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/auto_functionalize.py", line 357, in __call__
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]     assert can_auto_functionalize(_mutable_op)
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Worker_TP1 pid=2452919) ERROR 03-18 20:35:19 [multiproc_executor.py:932] AssertionError
```
