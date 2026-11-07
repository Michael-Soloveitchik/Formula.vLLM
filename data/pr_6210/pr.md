# PR #6210: [Kernel] reloading fused_moe config on the last chunk

- field: gpu_kernels  | area: model_executor/layers
- author: avshalomman
- state: MERGED  created: 2024-07-08T09:38:23Z  merged: 2024-07-08T15:00:39Z
- url: https://github.com/vllm-project/vllm/pull/6210
- labels: 

## Body

This is a follow-up PR for the fused_moe chunking mechanism introduced [here](https://github.com/vllm-project/vllm/pull/6029). We've noticed that when the last chunk is smaller then the chunk size, it helps to reload the kernel configuration according to the last chunk size.
In our experiments, this can yield a ~5% speedup on average (see plot for a general idea of the diff in performance).

<img width="400" alt="Screenshot 2024-07-07 at 16 21 28" src="https://github.com/vllm-project/vllm/assets/12231371/37096b5e-7f5d-48c6-b547-9350b1f8aae2">


## Comments

(no discussion comments)