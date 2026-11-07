# PR #18242: [V1][P/D] An native implementation of xPyD based on P2P NCCL

- field: scheduling  | area: distributed/kv_transfer
- author: Abatom
- state: MERGED  created: 2025-05-16T04:07:35Z  merged: 2025-06-18T06:32:36Z
- url: https://github.com/vllm-project/vllm/pull/18242
- labels: documentation, ready, ci/build, v1

## Body

An implementation of xPyD with dynamic scaling based on point-to-point communication, partly inspired by Dynamo.

> Follow-up on PR #15806
> [中文设计文档](https://zhuanlan.zhihu.com/p/1917559316807923245)

# Detailed Design

## Overall Process
As shown in Figure 1, the overall process of this **PD disaggregation** solution is described through a request flow:  

1. The client sends an HTTP request to the Proxy/Router's `/v1/completions` interface.  
2. The Proxy/Router selects a **1P1D (1 Prefill instance + 1 Decode instance)** through either through round-robin or random selection, generates a `request_id` (rules to be introduced later), modifies the `max_tokens` in the HTTP request message to **1**, and then forwards the request to the **P instance**.  
3. Immediately afterward, the Proxy/Router forwards the **original HTTP request** to the **D instance**.  
4. The **P instance** performs **Prefill** and then **actively sends the generated KV cache** to the D instance (using **PUT_ASYNC** mode). The D instance's `zmq_addr` can be resolved through the `request_id`.  
5. The **D instance** has a **dedicated thread** for receiving the KV cache (to avoid blocking the main process). The received KV cache is saved into the **GPU memory buffer**, the size of which is determined by the vLLM startup parameter `kv_buffer_size`. When the GPU buffer is full, the KV cache is stored in the **local Tensor memory pool**.  
6. During the **Decode**, the D instance's main process retrieves the KV cache (transmitted by the P instance) from either the **GPU buffer** or the **memory pool**, thereby **skipping Prefill**.  
7. After completing **Decode**, the D instance returns the result to the **Proxy/Router**, which then forwards it to the **client**.

![image1](https://github.com/user-attachments/assets/fb01bde6-755b-49f7-ad45-48a94b1e10a7)

## Proxy/Router (Demo)

A simple HTTP service acts as the entry point for client requests and starts a background thread to listen for P/D instances reporting their HTTP IP and PORT, as well as ZMQ IP and PORT. It maintains a dictionary of `http_addr -> zmq_addr`. The `http_addr` is the IP:PORT for the vLLM instance's request, while the `zmq_addr` is the address for KV cache handshake and metadata reception.

The Proxy/Router is responsible for selecting 1P1D based on the characteristics of the client request, such as the prompt, and generating a corresponding `request_id`, for example:

```
cmpl-___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_93923d63113b4b338973f24d19d4bf11-0
```

Currently, to quickly verify whether xPyD can work, a round-robin selection of 1P1D is used. In the future, it is planned to use a trie combined with the load status of instances to select appropriate P and D.

Each P/D instance periodically sends a heartbeat packet to the Proxy/Router (currently every 3 seconds) to register (i.e., report `http_addr -> zmq_addr`) and keep the connection alive. If an instance crashes and fails to send a ping for a certain period of time, the Proxy/Router will remove the timed-out instance (this feature has not yet been developed).

## KV Cache Transfer Methods

There are three methods for KVcache transfer: PUT, GET, and PUT_ASYNC. These methods can be specified using the `--kv-transfer-config` and `kv_connector_extra_config` parameters, specifically through the `send_type` field. Both PUT and PUT_ASYNC involve the P instance actively sending KVcache to the D instance. The difference is that PUT is a synchronous transfer method that blocks the main process, while PUT_ASYNC is an asynchronous transfer method. PUT_ASYNC uses a dedicated thread for sending KVcache, which means it does not block the main process. In contrast, the GET method involves the P instance saving the KVcache to the memory buffer after computing the prefill. The D instance then actively retrieves the computed KVcache from the P instance once it has allocated space for the KVcache.

Experimental results have shown that the performance of these methods, from highest to lowest, is as follows: PUT_ASYNC → GET → PUT.

## P2P Communication via ZMQ & NCCL

As long as the address of the counterpart is known, point-to-point KV cache transfer (using NCCL) can be performed, without being constrained by rank and world size. To support dynamic scaling (expansion and contraction) of instances with PD disaggregation. This means that adding or removing P/D instances does not require a full system restart.

Each P/D instance only needs to create a single `P2pNcclEngine` instance. This instance maintains a ZMQ Server, which runs a dedicated thread to listen on the `zmq_addr` address and receive control flow requests from other instances. These requests include requests to establish an NCCL connection and requests to send KVcache metadata (such as tensor shapes and data types). However, it does not actually transmit the KVcache data itself.

When a P instance and a D instance transmit KVcache for the first time, they need to establish a ZMQ connection and an NCCL group. For subsequent KVcache transmissions, this ZMQ connection and NCCL group are reused. The NCCL group consists of only two ranks, meaning the world size is equal to 2. This design is intended to support dynamic scaling, which means that adding or removing P/D instances does not require a full system restart. As long as the address of the counterpart is known, point-to-point KVcache transmission can be performed, without being restricted by rank or world size.

## NCCL Group Topology

Currently, only symmetric TP (Tensor Parallelism) methods are supported for KVcache transmission. Asymmetric TP and PP (Pipeline Parallelism) methods will be supported in the future. Figure 2 illustrates the 1P2D setup, where each instance has a TP (Tensor Parallelism) degree of 2. There are a total of 7 NCCL groups: three vLLM instances each have one NCCL group with TP=2. Additionally, the 0th GPU card of the P instance establishes an NCCL group with the 0th GPU card of each D instance. Similarly, the 1st GPU card of the P instance establishes an NCCL group with the 1st GPU card of each D instance.

![image2](https://github.com/user-attachments/assets/837e61d6-365e-4cbf-8640-6dd7ab295b36)

Each NCCL group occupies a certain amount of GPU memory buffer for communication, the size of which is primarily influenced by the `NCCL_MAX_NCHANNELS` environment variable. When `NCCL_MAX_NCHANNELS=16`, an NCCL group typically occupies 100MB, while when `NCCL_MAX_NCHANNELS=8`, it usually takes up 52MB. For large-scale xPyD configurations—such as DeepSeek's 96P144D—this implementation is currently not feasible. Moving forward, we are considering using RDMA for point-to-point communication and are also keeping an eye on UCCL.

## GPU Memory Buffer and Tensor Memory Pool

The trade-off in the size of the memory buffer is as follows: For P instances, the memory buffer is not required in PUT and PUT_ASYNC modes, but it is necessary in GET mode. For D instances, a memory buffer is needed in all three modes. The memory buffer for D instances should not be too large. Similarly, for P instances in GET mode, the memory buffer should also not be too large. The memory buffer of D instances is used to temporarily store KVcache sent by P instances. If it is too large, it will reduce the KVcache space available for normal inference by D instances, thereby decreasing the inference batch size and ultimately leading to a reduction in output throughput. The size of the memory buffer is configured by the parameter `kv_buffer_size`, measured in bytes, and is typically set to 5%～10% of the memory size.

If the `--max-num-seqs` parameter for P instances is set to a large value, due to the large batch size, P instances will generate a large amount of KVcache simultaneously. This may exceed the capacity of the memory buffer of D instances, resulting in KVcache loss. Once KVcache is lost, D instances need to recompute Prefill, which is equivalent to performing Prefill twice. Consequently, the time-to-first-token (TTFT) will significantly increase, leading to degraded performance.

To address the above issues, I have designed and developed a local Tensor memory pool for storing KVcache, inspired by the buddy system used in Linux memory modules. Since the memory is sufficiently large, typically in the TB range on servers, there is no need to consider prefix caching or using block-based designs to reuse memory, thereby saving space. When the memory buffer is insufficient, KVcache can be directly stored in the Tensor memory pool, and D instances can subsequently retrieve KVcache from it. The read and write speed is that of PCIe, with PCIe 4.0 having a speed of approximately 21 GB/s, which is usually faster than the Prefill speed. Otherwise, solutions like Mooncake and lmcache would not be necessary. The Tensor memory pool acts as a flood diversion area, typically unused except during sudden traffic surges. In the worst-case scenario, my solution performs no worse than the normal situation with a Cache store.


# Install vLLM

```shell
# Enter the home directory or your working directory.
cd /home

# Download the installation package, and I will update the commit-id in time. You can directly copy the command.
wget https://vllm-wheels.s3.us-west-2.amazonaws.com/9112b443a042d8d815880b8780633882ad32b183/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# Download the code repository.
git clone -b xpyd-v1 https://github.com/Abatom/vllm.git
cd vllm

# Set the installation package path.
export VLLM_PRECOMPILED_WHEEL_LOCATION=/home/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# installation
pip install -e . -v
```

# Run xPyD

## Instructions
- The following examples are run on an A800 (80GB) device, using the Meta-Llama-3.1-8B-Instruct model.
- Pay attention to the setting of the `kv_buffer_size` (in bytes). The empirical value is 10% of the GPU memory size. This is related to the kvcache size. If it is too small, the GPU memory buffer for temporarily storing the received kvcache will overflow, causing the kvcache to be stored in the tensor memory pool, which increases latency. If it is too large, the kvcache available for inference will be reduced, leading to a smaller batch size and decreased throughput.
- For Prefill instances, when using non-GET mode, the `kv_buffer_size` can be set to 1, as Prefill currently does not need to receive kvcache. However, when using GET mode, a larger `kv_buffer_size` is required because it needs to store the kvcache sent to the D instance.
- You may need to modify the `kv_buffer_size` and `port` in the following commands (if there is a conflict).
- `PUT_ASYNC` offers the best performance and should be prioritized.
- The `--port` must be consistent with the `http_port` in the `--kv-transfer-config`.
- The `disagg_prefill_proxy_xpyd.py` script will use port 10001 (for receiving client requests) and port 30001 (for receiving service discovery from P and D instances).
- The node running the proxy must have `quart` installed.
- Supports multiple nodes; you just need to modify the `proxy_ip` and `proxy_port` in `--kv-transfer-config`.
- In the following examples, it is assumed that **the proxy's IP is 10.0.1.1**.

## Run 1P3D

### Proxy (e.g. 10.0.1.1)

```shell
cd {your vllm directory}/examples/online_serving/disagg_xpyd/
python3 disagg_prefill_proxy_xpyd.py &
```

### Prefill1 (e.g. 10.0.1.2 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20005 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Decode1 (e.g. 10.0.1.3 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20009 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Decode2 (e.g. 10.0.1.4 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20003 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Decode3 (e.g. 10.0.1.5 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20008 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

## Run 3P1D

### Proxy (e.g. 10.0.1.1)

```shell
cd {your vllm directory}/examples/online_serving/disagg_xpyd/
python3 disagg_prefill_proxy_xpyd.py &
```

### Prefill1 (e.g. 10.0.1.2 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20005 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Prefill2 (e.g. 10.0.1.3 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=1 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20009 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20009","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Prefill3 (e.g. 10.0.1.4 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=2 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20003 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"23001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20003","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

### Decode1 (e.g. 10.0.1.5 or 10.0.1.1)

```shell
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=3 vllm serve {your model directory} \
    --host 0.0.0.0 \
    --port 20008 \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --served-model-name base_model \
    --dtype float16 \
    --max-model-len 10000 \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"24001","kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20008","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}' > /var/vllm.log 2>&1 &
```

# Single request

```shell
curl -X POST -s http://10.0.1.1:10001/v1/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "base_model",
    "prompt": "San Francisco is a",
    "max_tokens": 10,
    "temperature": 0
}'
```

# Benchmark

```shell
python3 benchmark_serving.py \
    --backend vllm \
    --model base_model \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name "random" \
    --host 10.0.1.1 \
    --port 10001 \
    --random-input-len 1024 \
    --random-output-len 1024 \
    --ignore-eos \
    --burstiness 100 \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "90,95,99" \
    --seed $(date +%s) \
    --trust-remote-code \
    --request-rate 3 \
    --num-prompts 1000
```

# Shut down

```shell
pgrep python | xargs kill -9 && pkill -f python
```

# Test data

## **Scenario 1**: 1K input & 1K output tokens, E2E P99 latency ~20s  
- **1P5D (6×A800) vs vLLM (1×A800)**:  
  - Throughput ↑7.2% (1085 → 6979/6)  
  - ITL (P99) ↓81.3% (120ms → 22.9ms) 
  - TTFT (P99) ↑26.8% (175ms → 222ms) 
  - TPOT: No change  

- **1P6D (7×A800) vs vLLM (1×A800)**:  
  - Throughput ↑9.6% (1085 → 8329/7)  
  - ITL (P99) ↓81.0% (120ms → 22.7ms) 
  - TTFT (P99) ↑210% (175ms →543ms) 
  - TPOT: No change  

## **Scenario 2**: 1K input & 200 output tokens, E2E P99 latency ~4s  
- **1P1D (2×A800) vs vLLM (1×A800)**:  
  - Throughput ↑37.4% (537 → 1476/2)  
  - ITL (P99) ↓81.8% (127ms → 23.1ms) 
  - TTFT (P99) ↑41.8% (160ms → 227ms) 
  - TPOT: No change

![testdata](https://github.com/user-attachments/assets/f791bfc7-9f3d-4e5c-9171-a42f9f4da627)


# TODO

## In this PR

- [x] Fix the issue of inaccurate results caused by Chunked prefill.
- [x] Fix the issue of inaccurate results caused by preemption.
- [x] Clear the buffer upon request completion.

## In the following PRs

- [ ] Support automatic instance removal after crash
- [ ] Support asynchronous loading of KV cache
- [ ] Release KV cache after sending is completed
- [ ] Support PP and EP, etc.
- [ ] Support asymmetric TP, PP, and EP, etc.


## Comments

### github-actions @ 2025-05-16T04:07:44Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### WXFMAV @ 2025-05-19T09:03:57Z

![image](https://github.com/user-attachments/assets/16c761bd-c11d-4b9e-b538-d04ed33b834a)
![image](https://github.com/user-attachments/assets/f12704e1-870b-443d-8eca-77af9d89288e)
你好，我继续基于您的这个pd版本的v1版本往下进行了尝试，遇到一个trace profile的很奇怪的问题。 我的程序中一个主计算流，和一个ncclRecv接收数据的流，按道理应该是各自独立的。 但是运行的时候，发现接收数据的效率比较慢，Profiler看了一下之后，发现经常会有接收数据的kernel在wait计算流的操作完成，但是代码上看起来接收数据的stream与计算流是没有关系的，请问是不是我观察trace的方式不对啊，还是ncclRecv内部自动会做全局cuda.synchronize()?
---
### Abatom @ 2025-05-19T16:09:27Z

@WXFMAV Thank you very much for your continuous attention.My understanding is that only one operator can be executed at the same time,even if it is in different streams.It still has to wait for the operator in another stream to finish before executing the operator in this stream.By the way,is the version you are running based on the xpyd-v1 branch and built using a Dockerfile to create the image?
---
### WXFMAV @ 2025-05-23T00:30:09Z

> @WXFMAV Thank you very much for your continuous attention.My understanding is that only one operator can be executed at the same time,even if it is in different streams.It still has to wait for the operator in another stream to finish before executing the operator in this stream.By the way,is the version you are running based on the xpyd-v1 branch and built using a Dockerfile to create the image?

是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。 
---
### Abatom @ 2025-05-23T09:04:33Z

> 是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。

@WXFMAV Can you submit the main changes to my branch 'xpyd-v1'? You're welcome to be a co-author.
---
### mmdbhs @ 2025-05-27T13:26:34Z

> > @WXFMAV Thank you very much for your continuous attention.My understanding is that only one operator can be executed at the same time,even if it is in different streams.It still has to wait for the operator in another stream to finish before executing the operator in this stream.By the way,is the version you are running based on the xpyd-v1 branch and built using a Dockerfile to create the image?
> 
> 是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。

请问您是在什么模型，什么卡配置下获得了30%的吞吐提升呢
---
### WXFMAV @ 2025-05-28T00:36:26Z

> > 是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。
> 
> @WXFMAV Can you submit the main changes to my branch 'xpyd-v1'? You're welcome to be a co-author.



> > 是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。
> 
> @WXFMAV Can you submit the main changes to my branch 'xpyd-v1'? You're welcome to be a co-author.

可以的，我把一些业务的逻辑剥离一下，然后提交一个版本出来。 
---
### WXFMAV @ 2025-05-28T00:43:37Z

> > > @WXFMAV Thank you very much for your continuous attention.My understanding is that only one operator can be executed at the same time,even if it is in different streams.It still has to wait for the operator in another stream to finish before executing the operator in this stream.By the way,is the version you are running based on the xpyd-v1 branch and built using a Dockerfile to create the image?
> > 
> > 
> > 是的，我就是基于您这边的V1版本往下尝试的，出现这个问题的原因，是vllm的Mainbranch中加了好几个全局的cuda.Synchronize操作，而且增加的位置和接收流发生了冲突，导致时序混乱了。 这两天修改了一些代码，终于搞通了，在我们的2K prompts 长度，2~3KTOKEN生成的任务中，有30%左右的吞吐量提升。主函数执行torch.cuda.Synchronize()后，全局所有的流都要处于wait状态，所以我把主函数中的torch.cuda.Synchronize()改成了只wait 自己的流。 建议我们的这个xpyd实现中，也要同时把Main Thread中的全部同步操作都改成细粒度的同步操作。
> 
> 请问您是在什么模型，什么卡配置下获得了30%的吞吐提升呢

Qwen2-7b, A100, 2k prompts, 2k token genration. 1P5D
---
### ekagra-ranjan @ 2025-05-28T14:49:53Z

Thanks for the PR! Could you share more on why this was an issue and how setting low value helps D nodes?

> For the P instance, the --max-num-seqs should be set to a smaller value. In my scenario, I set it to 5 to avoid filling up the buffer of the D instance, which would otherwise cause the D instance to recompute the prefill. To address the issue of setting --max-num-seqs to a small value, I have implemented a local memory pool to handle the sudden increase in kvcache to avoid recompute prefill.
---
### Abatom @ 2025-05-28T15:56:13Z

> Thanks for the PR! Could you share more on why this was an issue and how setting low value helps D nodes?
> 
> > For the P instance, the --max-num-seqs should be set to a smaller value. In my scenario, I set it to 5 to avoid filling up the buffer of the D instance, which would otherwise cause the D instance to recompute the prefill. To address the issue of setting --max-num-seqs to a small value, I have implemented a local memory pool to handle the sudden increase in kvcache to avoid recompute prefill.

The buffer of D instance's GPU memory cannot be too large. This buffer is used to temporarily store the kvcache sent by P. If the buffer is too large, it will reduce the size of D instance's kvcache, which in turn will decrease the inference batch size and consequently reduce the output throughput. If the --max-num-seqs of the P instance is set to a relatively large value, it will result in a large amount of kvcache generated by the P instance at the same time because of the large batch size. This size is likely to exceed the buffer of the D instance, leading to the loss of kvcache. When loss occurs, the D instance needs to recompute the prefill, which is equivalent to computing prefill twice. Therefore, the TTFT (Time to First Token) will be significantly increased and become worse.
---
### Abatom @ 2025-05-31T02:35:08Z

> Impressive work! Left some comments. Also, I requested the design doc access, could you please check?

@KuntaiDu, Thank you for your review. I will modify the code later. The access permission for the design doc has been granted.
---
### cyber-pioneer @ 2025-06-02T06:53:06Z

Could you please show the benchmark data for the Meta-Llama-3.1-8B-Instruct model in both v0 and v1 versions?

---
### Abatom @ 2025-06-02T08:20:50Z

> Could you please show the benchmark data for the Meta-Llama-3.1-8B-Instruct model in both v0 and v1 versions?
> 

I had tested it once before, but there have been many changes since then, so I need to test it again. Also, Version 0 is no longer supported.
---
### mergify @ 2025-06-06T03:48:57Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Abatom.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### gemini-code-assist @ 2025-06-13T06:09:06Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-13T06:25:37Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-13T06:27:21Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-13T06:41:05Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-13T06:41:10Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-13T06:41:40Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### KuntaiDu @ 2025-06-16T04:01:25Z

I did initial performance benchmarking. It seems to be functional.
(BTW in the doc, maybe change all the IPs to `0.0.0.0` instead of using the local ip address `10.0.1.1`, and `--tokenizer auto` does not work for me --- I changed to `--tokenizer meta-llama/Llama-3.1-8B-Instruct`. Also add `--disable-log-request`)

Please clean up the log in `p2p_nccl_engine.py`, it's a bit bothering.

Perf TL;DR: still have room for optimization (~30ms overhead for 10000 tokens)

Some performance number:
Device: 8xH100.
1P1D, workload:
`python3 benchmark_serving.py --port 10001 --seed $(date +%s) \
        --tokenizer meta-llama/Llama-3.1-8B-Instruct --model base_model \
        --dataset-name random --random-input-len 8000 --random-output-len 200 \
        --num-prompts 200 --burstiness 100 --request-rate 3.6 --metric-percentiles 95  --ignore-eos`

Perf of this PR:
---------------Time to First Token----------------
Mean TTFT (ms):                          309.84    
Median TTFT (ms):                        298.10    
P95 TTFT (ms):                           359.73    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.45     
Median TPOT (ms):                        10.37     
P95 TPOT (ms):                           11.34   

Perf of Dynamo (an unofficial version in old vLLM PR)
---------------Time to First Token----------------
Mean TTFT (ms):                          278.96    
Median TTFT (ms):                        278.54    
P95 TTFT (ms):                           292.48    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.16     
Median TPOT (ms):                        13.22     
P95 TPOT (ms):                           13.41   



2P1D, workload:
`python3 benchmark_serving.py --tokenizer meta-llama/Llama-3.1-8B-Instruct --port 10001 --host 0.0.0.0 --seed $(date +%s)         --model base_model         --dataset-name random --random-input-len 9900 --random-output-len 100         --num-prompts 250 --burstiness 100 --request-rate 5.5 --metric-percentiles 95`

Perf of this PR
---------------Time to First Token----------------
Mean TTFT (ms):                          386.52    
Median TTFT (ms):                        363.16    
P95 TTFT (ms):                           572.50    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          11.31     
Median TPOT (ms):                        11.33     
P95 TPOT (ms):                           12.30     

Perf of Dynamo (unofficial version):
---------------Time to First Token----------------
Mean TTFT (ms):                          356.09    
Median TTFT (ms):                        342.67    
P95 TTFT (ms):                           359.73    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.09     
Median TPOT (ms):                        13.12     
P95 TPOT (ms):                           13.42   

---
### cyber-pioneer @ 2025-06-16T08:25:35Z

Why to set such request rate in the benchmark table? It seems that the baseline request rate does not reach the performance bottleneck. 
<img width="378" alt="截屏2025-06-16 16 20 04" src="https://github.com/user-attachments/assets/c929f3c0-3442-47c2-a777-501aa6867c8a" />

---
### Abatom @ 2025-06-16T08:44:48Z

@cyber-pioneer I didn't quite understand your question. In your table, when the request rate is equal to 4, the P99 TTFT has suddenly increased, indicating that the service has reached its bottleneck. The longer the pressure test runs, the larger the TTFT will become.
---
### cyber-pioneer @ 2025-06-16T08:57:53Z

> @cyber-pioneer I didn't quite understand your question. In your table, when the request rate is equal to 4, the P99 TTFT has suddenly increased, indicating that the service has reached its bottleneck. The longer the pressure test runs, the larger the TTFT will become.

Thanks for you response. You are right. When request rate is set 4, all latency metrics have  severely degraded.  2 might be a more balanced choice, especially for throughput performance.
---
### simon-mo @ 2025-06-16T19:10:49Z

/gemini review
---
### gemini-code-assist @ 2025-06-17T00:19:10Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:24Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:29Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:35Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:43Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:50Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:19:55Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### gemini-code-assist @ 2025-06-17T00:20:01Z

> [!WARNING]
> You have reached your daily quota limit. Please wait up to 24 hours and I will start processing your requests again!
---
### zovonoir @ 2025-07-18T16:25:41Z

Hi Abatom, have you ever encountered hang or deadlock bug in with this patch? In my 2node 8xMI300X/node environment, I encountered deadlock problem, but it disappeared in single node test.
Here are my testing scripts

P instance

```
#!/bin/bash

LOG_FILE="logs/vllm_serve_prefill_$(date +'%Y%m%d_%H-%M-%S').log"

set -ex
export GLOO_SOCKET_IFNAME=ens50f0
export NCCL_SOCKET_IFNAME=ens50f0

export CUDA_VISIBLE_DEVICES=0,1
export HIP_VISIBLE_DEVICES=0,1
export NCCL_NCHANNELS_PER_NET_PEER=1
export VLLM_LOGGING_CONFIG_PATH=log.conf.json
 {
 NCCL_DEBUG=INFO VLLM_RINGBUFFER_WARNING_INTERVAL=500 VLLM_RPC_TIMEOUT=1800000 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 \
vllm serve /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 \
    -tp 2 \
    --block-size 16 \
    --max_seq_len_to_capture 6144 \
    --max-num-batched-tokens 6144 \
    --host 0.0.0.0 \
    --port 20005 \
    --trust-remote-code \
    --gpu-memory-utilization 0.7 \
    --disable-log-request \
    --enforce-eager \
    --disable-custom-all-reduce  \
    --served-model-name deepseek-ai/DeepSeek-R1 \
    --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e10","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"10.200.190.55","proxy_port":"30001","http_port":"20005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'
 } 2>&1 | tee -a "$LOG_FILE" &
```
D instance

```
#!/bin/bash
set -ex
LOG_FILE="logs/vllm_serve_decode_$(date +'%Y%m%d_%H-%M-%S').log"
export GLOO_SOCKET_IFNAME=ens50f0
export NCCL_SOCKET_IFNAME=ens50f0
#export NCCL_DEBUG=INFO

export NCCL_NCHANNELS_PER_NET_PEER=1
export VLLM_LOGGING_CONFIG_PATH=log.conf.json
export CUDA_VISIBLE_DEVICES=6,7
export HIP_VISIBLE_DEVICES=6,7
{
 NCCL_DEBUG=INFO  VLLM_RINGBUFFER_WARNING_INTERVAL=500 VLLM_RPC_TIMEOUT=1800000 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1\
    vllm serve /apps/data/models/models--Qwen--Qwen3-0.6B/snapshots/e6de91484c29aa9480d55605af694f39b081c455 \
        -tp 2   \
        --block-size 16  \
        --max_seq_len_to_capture 6144 \
        --max-num-batched-tokens 6144 \
        --host 0.0.0.0 \
        --port 40005 \
        --trust-remote-code \
        --gpu-memory-utilization 0.7 \
        --disable-log-request \
        --enforce-eager \
	--disable-custom-all-reduce  \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"1e10","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"10.200.190.55","proxy_port":"30001","http_port":"40005","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'
} 2>&1 | tee -a "$LOG_FILE" &

```
My benchmark script

```
ISL=1024
OSL=1
RATIO=1.0
PORT=10001
 
CONCURRENCY=2   # I found if this value > 1, P instance will hang or deadlock
#CONCURRENCY="32"
for CON in $CONCURRENCY; do
    if [ $CON -le 32 ]; then
      PROMPTS=128
    elif [ $CON -le 128 ]; then
      PROMPTS=128
    else
      PROMPTS=$((CON * 2))
    fi
    python3 -m sglang.bench_serving \
        --warmup-requests 0 \
        --dataset-name random \
        --dataset-path /apps/zovtest/vllm_pd/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
        --model  deepseek-ai/DeepSeek-R1 \
        --random-input-len $ISL \
        --random-output-len $OSL \
        --num-prompt $PROMPTS \
        --random-range-ratio $RATIO \
        --max-concurrency $CON \
        --base-url http://10.200.190.55:10001 \
        --backend vllm
done
```

I found P instance will hang at synchronize next to ncclSend.





---
### Abatom @ 2025-07-19T03:44:50Z

@zovonoir Let's communicate on Slack.
---
### zovonoir @ 2025-07-19T11:38:00Z

> @zovonoir Let's communicate on Slack.

Cool.Thanks for the invitation! I've just submitted my request to join the vLLM Slack workspace (https://inviter.co/vllm-slack). Will contact you directly ASAP regarding this question. Really appreciate your guidance!
---