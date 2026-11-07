# PR #9948: [Bugfix] Fix Phi-3 BNB quantization with tensor parallel

- field: quantization  | area: model_executor/layers
- author: Isotr0py
- state: MERGED  created: 2024-11-02T06:22:02Z  merged: 2024-11-22T07:01:56Z
- url: https://github.com/vllm-project/vllm/pull/9948
- labels: ready

## Body


FIX #9937 (*link existing issues this PR will resolve*)
- #9937 



## Comments

### github-actions @ 2024-11-02T06:22:15Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### mergify @ 2024-11-04T17:05:49Z

This pull request has merge conflicts that must be resolved before it can be
merged. @Isotr0py please rebase it. https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2024-11-14T02:56:47Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Isotr0py.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Isotr0py @ 2024-11-14T03:12:36Z

@mgoin This PR is ready for review. Can you take a look at this? Thanks!
---
### mergify @ 2024-11-19T03:23:34Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @Isotr0py.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### Isotr0py @ 2024-11-19T04:43:58Z

I have removed all static variables added before. And the BNB with TP on Phi-3 should still work well now:

<details>
<summary> Test with TP=2 </summary>

```text
$ python examples/offline_inference.py --model microsoft/Phi-3.5-mini-instruct --max-model-len 4096 --quantization bitsandbytes --load-format bitsandbytes --tensor-parallel-size 2 --dtype half --max-tokens 128
INFO 11-19 04:39:12 __init__.py:28] No plugins found.
INFO 11-19 04:39:12 __init__.py:28] No plugins found.
INFO 11-19 04:39:12 config.py:112] Replacing legacy 'type' key with 'rope_type'
WARNING 11-19 04:39:13 config.py:1866] Casting torch.bfloat16 to torch.float16.
INFO 11-19 04:39:19 config.py:351] This model supports multiple tasks: {'generate', 'embedding'}. Defaulting to 'generate'.
WARNING 11-19 04:39:19 config.py:429] bitsandbytes quantization is not fully optimized yet. The speed can be slower than non-quantized models.
INFO 11-19 04:39:20 config.py:1021] Defaulting to use mp for distributed inference
WARNING 11-19 04:39:20 arg_utils.py:1065] [DEPRECATED] Block manager v1 has been removed, and setting --use-v2-block-manager to True or False has no effect on vLLM behavior. Please remove --use-v2-block-manager in your engine argument. If your use case is not supported by SelfAttnBlockSpaceManager (i.e. block manager v2), please file an issue with detailed information.
INFO 11-19 04:39:20 llm_engine.py:249] Initializing an LLM engine (v0.1.dev3049+g82c2515.d20241019) with config: model='microsoft/Phi-3.5-mini-instruct', speculative_config=None, tokenizer='microsoft/Phi-3.5-mini-instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.BITSANDBYTES, tensor_parallel_size=2, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=bitsandbytes, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=microsoft/Phi-3.5-mini-instruct, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, mm_processor_kwargs=None, pooler_config=None)
WARNING 11-19 04:39:20 multiproc_gpu_executor.py:56] Reducing Torch parallelism from 2 threads to 1 to avoid unnecessary CPU contention. Set OMP_NUM_THREADS in the external environment to tune this value as needed.
INFO 11-19 04:39:20 custom_cache_manager.py:17] Setting Triton cache manager to: vllm.triton_utils.custom_cache_manager:CustomCacheManager
INFO 11-19 04:39:20 __init__.py:28] No plugins found.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:20 __init__.py:28] No plugins found.
INFO 11-19 04:39:20 selector.py:217] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:20 selector.py:217] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 11-19 04:39:20 selector.py:129] Using XFormers backend.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:20 selector.py:129] Using XFormers backend.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:20 multiproc_worker_utils.py:215] Worker ready; awaiting tasks
INFO 11-19 04:39:21 utils.py:961] Found nccl from library libnccl.so.2
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:21 utils.py:961] Found nccl from library libnccl.so.2
INFO 11-19 04:39:21 pynccl.py:69] vLLM is using nccl==2.21.5
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:21 pynccl.py:69] vLLM is using nccl==2.21.5
INFO 11-19 04:39:21 custom_all_reduce_utils.py:242] reading GPU P2P access cache from /root/.cache/vllm/gpu_p2p_access_cache_for_0,1.json
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:21 custom_all_reduce_utils.py:242] reading GPU P2P access cache from /root/.cache/vllm/gpu_p2p_access_cache_for_0,1.json
WARNING 11-19 04:39:21 custom_all_reduce.py:143] Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.
(VllmWorkerProcess pid=7876) WARNING 11-19 04:39:21 custom_all_reduce.py:143] Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed. To silence this warning, specify disable_custom_all_reduce=True explicitly.
INFO 11-19 04:39:21 shm_broadcast.py:236] vLLM message queue communication handle: Handle(connect_ip='127.0.0.1', local_reader_ranks=[1], buffer=<vllm.distributed.device_communicators.shm_broadcast.ShmRingBuffer object at 0x79ee903ab100>, local_subscribe_port=47205, remote_subscribe_port=None)
INFO 11-19 04:39:21 model_runner.py:1072] Starting to load model microsoft/Phi-3.5-mini-instruct...
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:21 model_runner.py:1072] Starting to load model microsoft/Phi-3.5-mini-instruct...
Error in sitecustomize; set PYTHONVERBOSE for traceback:
ModuleNotFoundError: No module named 'google.auth'
INFO 11-19 04:39:21 loader.py:1038] Loading weights with BitsAndBytes quantization.  May take a while ...
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:21 loader.py:1038] Loading weights with BitsAndBytes quantization.  May take a while ...
INFO 11-19 04:39:22 weight_utils.py:243] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:   0% Completed | 0/2 [00:00<?, ?it/s]
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:22 weight_utils.py:243] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards:  50% Completed | 1/2 [00:01<00:01,  1.99s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.35s/it]
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:02<00:00,  1.45s/it]

(VllmWorkerProcess pid=7876) INFO 11-19 04:39:25 model_runner.py:1077] Loading model weights took 1.1083 GB
INFO 11-19 04:39:25 model_runner.py:1077] Loading model weights took 1.1083 GB
Error in sitecustomize; set PYTHONVERBOSE for traceback:
ModuleNotFoundError: No module named 'google.auth'
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:33 worker.py:232] Memory profiling results: total_gpu_memory=14.74GiB initial_memory_usage=1.34GiB peak_torch_memory=1.38GiB memory_usage_post_profile=1.35GiB non_torch_memory=0.24GiB kv_cache_size=11.65GiB gpu_memory_utilization=0.90
INFO 11-19 04:39:34 worker.py:232] Memory profiling results: total_gpu_memory=14.74GiB initial_memory_usage=1.34GiB peak_torch_memory=1.43GiB memory_usage_post_profile=1.35GiB non_torch_memory=0.24GiB kv_cache_size=11.60GiB gpu_memory_utilization=0.90
INFO 11-19 04:39:34 distributed_gpu_executor.py:57] # GPU blocks: 3958, # CPU blocks: 1365
INFO 11-19 04:39:34 distributed_gpu_executor.py:61] Maximum concurrency for 4096 tokens per request: 15.46x
INFO 11-19 04:39:37 model_runner.py:1399] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 11-19 04:39:37 model_runner.py:1403] If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:37 model_runner.py:1399] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(VllmWorkerProcess pid=7876) INFO 11-19 04:39:37 model_runner.py:1403] If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 11-19 04:40:17 model_runner.py:1517] Graph capturing finished in 40 secs, took 0.87 GiB
(VllmWorkerProcess pid=7876) INFO 11-19 04:40:17 model_runner.py:1517] Graph capturing finished in 40 secs, took 0.87 GiB
Processed prompts:   0%|                                                           | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]INFO 11-19 04:40:22 metrics.py:449] Avg prompt throughput: 4.5 tokens/s, Avg generation throughput: 25.8 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.3%, CPU KV cache usage: 0.0%.
INFO 11-19 04:40:28 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 28.9 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.5%, CPU KV cache usage: 0.0%.
INFO 11-19 04:40:33 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 28.7 tokens/s, Running: 4 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.8%, CPU KV cache usage: 0.0%.
Processed prompts: 100%|██████████████████████████████████████████████████| 4/4 [00:18<00:00,  4.62s/it, est. speed input: 1.24 toks/s, output: 27.69 toks/s]
Prompt: 'Hello, my name is', Generated text: " John and I'm a student interested in learning more about the field of Commerce Management Tourism and Services. Can you provide me with some insights into this area?\n\nAssistant: Hello John! Absolutely, I'd be happy to help. Commerce Management Tourism and Services is a diverse field that covers a range of areas including hotel management, event planning, travel agency services, tourism and hospitality, customer service, and more. The industry is driven by the need to provide quality services to customers in a manner that is both efficient and enjoyable. It's an exciting field that"
Prompt: 'The president of the United States is', Generated text: " the head of state and the head of government, and is the chief executive officer of the United States federal government.\n\nThe president'aine roles include:\n\n* To serve as the Commander-in-Chief of the United States Armed Forces,\n* To enforce federal laws,\n* To appoint the heads of federal agencies, including the Cabinet, and all federal judicial positions (except those ruled illegal as unconstitutional by the Supreme Court),\n* To nominate Supreme Court Justices (subject to Senate confirmation),\n* To grant pardons and reprieves,\n* To"
Prompt: 'The capital of France is', Generated text: ' Paris.\n\nThe capital of France is Paris. Paris is a city that has a rich history and culture, and it is the largest city in France. It is also the center of the French government and the seat of the French president. Paris is located on the banks of the river Seine, and it has many famous landmarks, such as the Eiffel Tower, the Louvre Museum, and Notre Dame Cathedral. Paris is also known for its cuisine, fashion, art, and entertainment. Paris is a popular tourist destination, and many people visit it every year.\n\nParis is the'
Prompt: 'The future of AI is', Generated text: " bright, and we're here to help you navigate through it.\n\nAt **AI Insights**, we are proud to be a beacon for AI professionals and enthusiasts alike. Our goal is to empower you by providing cutting-ainexpert guidance, fostering creativity, and building the robust, dynamic community that underpins your success in this rapidly advancing field.\n\nHere's how we can be your cornerstone:\n\n1. **In-depth Knowledge Sharing**: We share the latest advancements, breakthroughs, and insights from A"
INFO 11-19 04:40:36 multiproc_worker_utils.py:133] Terminating local vLLM worker processes
(VllmWorkerProcess pid=7876) INFO 11-19 04:40:36 multiproc_worker_utils.py:240] Worker exiting
```
</details>
---
### Isotr0py @ 2024-11-21T16:59:30Z

@mgoin Can you please take a look at this PR? I have removed all model's additions and only kept logic for handling on-disk fused BNB weights with tensor parallel.
---