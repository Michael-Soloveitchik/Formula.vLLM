# PR #28449: [Feat][Perf] Enable deepep-low-latency with round-robin expert placement.

- field: quantization  | area: model_executor/layers
- author: cboss6
- state: MERGED  created: 2025-11-11T06:49:07Z  merged: 2025-11-19T12:46:24Z
- url: https://github.com/vllm-project/vllm/pull/28449
- labels: ready

## Body

## Purpose
Enable deepep low latency all2all backend with round-robin expert plamement strategy, which produces significant performance improvement.

## Performance
Test Platform: CUDA 12.8, drivier 550.144.03
Model: DeepSeek-R1-671B
GPU: H20 * 8 * 2 nodes
Vllm config: dp=16, tp=1, enable_expert_parallel=1, all2all_backend=deepep_low_latency, use_deep_gemm=1

The current functionality has been fully implemented with correct accuracy and significant performance improvements.
In the benchmark with parameters times=10, num_prompts=512, dataset=sharegpt, input_len=1024, output_len=512, max_concurrency=8, and req_rate=8,
the results show that compared to the default linear placement, round-robin's throughput increased by 14.57% and TPOT improved by 13.38%.
<img width="1657" height="911" alt="Clipboard_Screenshot_1762843721" src="https://github.com/user-attachments/assets/cc540e73-66c7-43d9-beaf-87f4e178c455" />


### serving command (head node)
export NCCL_CHECK_DISABLE=1
export NCCL_COLLNET_ENABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SL=3
export NCCL_IB_TC=160
export NCCL_LL_THRESHOLD=16384
export NCCL_NET_GDR_LEVEL=2
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_PXN_DISABLE=0
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond1
export NVSHMEM_HCA_LIST=mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1
export NVSHMEM_IB_TRAFFIC_CLASS=160
export NVSHMEM_DIR=/usr/local/nvshmem
export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
export PATH=${NVSHMEM_DIR}/bin:$PATH
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION=1
export VLLM_ATTENTION_BACKEND=FLASHMLA
export VLLM_USE_DEEP_GEMM=1
export DG_JIT_CACHE_DIR=/root/.cache/vllm/deep_gemm/

vllm serve /path/to/DeepSeek-R1 \
    --host 0.0.0.0 \
    -tp 1  \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --expert-placement-strategy round_robin \
    --enable-chunked-prefill \
    --gpu-memory-utilization 0.8 \
    --load-format "auto" \
    --enable-expert-parallel \
    --data-parallel-size 16 \
    --data-parallel-size-local 8 \
    --data-parallel-address ${HOST_IP} \
    --data-parallel-rpc-port 12345 \
    --api-server-count=8

### benchmark command
vllm bench serve \    --backend vllm \    --base-url "http://127.0.0.1:8500" \    --port 8500 \    --endpoint '/v1/completions' \    --model ${model} \    --dataset-name sharegpt \  --num-prompts 512 \    --max-concurrency 8 \
    --request-rate 8 \    --random-input-len 1024 \    --random-output-len 512

## Accuracy
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.97|±  |0.0171|
|     |       |strict-match    |     5|exact_match|↑  | 0.97|±  |0.0171|
<img width="1210" height="225" alt="Clipboard_Screenshot_1762842773" src="https://github.com/user-attachments/assets/37b88238-fca4-4276-90a8-801b543e895a" />



## Comments

### mergify @ 2025-11-11T08:16:11Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @cboss6.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-11-11T18:23:46Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @cboss6.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### bnellnm @ 2025-11-11T18:46:38Z

Nice perf numbers!  I just had a few minor comments.
---
### cboss6 @ 2025-11-14T11:19:07Z

Hi @bnellnm, I’ve updated the code based on your comments. Could you take another look when you have a moment? Thanks a lot!
---