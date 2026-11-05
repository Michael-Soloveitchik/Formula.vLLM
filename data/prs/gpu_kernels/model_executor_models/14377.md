# PR #14377: [Perf]:Optimize qwen2-vl to reduce cudaMemcpyAsync

- field: gpu_kernels  | area: model_executor/models
- author: cynthieye
- state: MERGED  created: 2025-03-06T17:51:47Z  merged: 2025-03-11T07:39:56Z
- url: https://github.com/vllm-project/vllm/pull/14377
- labels: ready

## Body

qwen2-vl logic optimization: During each forward propagation, the xformer branch of Qwen2VisionTransformer will execute multiple tensor tolist methods (flash attn branch will execute multiple tensor items) to force the GPU tensor to be copied to the CPU, triggering CUDAMemcpyAsync to increase time consumption. Since the input and output are the same multiple times, it will be executed once, and the remaining will reuse the first result. After optimization, the online environment xformer branch QPS can be improved by 15%, and the flash attn branch QPS can be improved by 7%

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### ywang96 @ 2025-03-11T04:27:44Z

@cynthieye Thank you for making this PR! Can you update this branch with our main branch? I think thr CI error should be fixed on main a while ago.
---