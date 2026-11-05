# PR #24217: Fix Auto_Round Quatization Loading on SM75 and Lower GPUs (#24217)

# Issue

## Original PR description (captured source)


Tested on Docker Image: vllm/vllm-openai:v0.9.2, GPU: Titan RTX.

## Purpose
Fix Auto_Round Quatization Loading on SM75 and Lower GPUs

When not fixed and trying to load Auto_Round quantization on sm75 and lower GPUs:
```
WARNING 09-03 20:42:15 [config.py:440] MoE DP setup unable to determine quantization scheme or unsupported quantization type. This model will not run with DP enabled.
ERROR 09-03 20:42:16 [engine.py:458] cannot access local variable 'GPTQMarlinMoEMethod' where it is not associated with a value
ERROR 09-03 20:42:16 [engine.py:458] Traceback (most recent call last):
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 446, in run_mp_engine
ERROR 09-03 20:42:16 [engine.py:458]     engine = MQLLMEngine.from_vllm_config(
ERROR 09-03 20:42:16 [engine.py:458]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 133, in from_vllm_config
ERROR 09-03 20:42:16 [engine.py:458]     return cls(
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 87, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.engine = LLMEngine(*args, **kwargs)
ERROR 09-03 20:42:16 [engine.py:458]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 265, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.model_executor = executor_class(vllm_config=vllm_config)
ERROR 09-03 20:42:16 [engine.py:458]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 53, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self._init_executor()
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
ERROR 09-03 20:42:16 [engine.py:458]     self.collective_rpc("load_model")
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 57, in collective_rpc
ERROR 09-03 20:42:16 [engine.py:458]     answer = run_method(self.driver_worker, method, args, kwargs)
ERROR 09-03 20:42:16 [engine.py:458]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 2736, in run_method
ERROR 09-03 20:42:16 [engine.py:458]     return func(*args, **kwargs)
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 210, in load_model
ERROR 09-03 20:42:16 [engine.py:458]     self.model_runner.load_model()
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1174, in load_model
ERROR 09-03 20:42:16 [engine.py:458]     self.model = get_model(vllm_config=self.vllm_config)
ERROR 09-03 20:42:16 [engine.py:458]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/__init__.py", line 59, in get_model
ERROR 09-03 20:42:16 [engine.py:458]     return loader.load_model(vllm_config=vllm_config,
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 38, in load_model
ERROR 09-03 20:42:16 [engine.py:458]     model = initialize_model(vllm_config=vllm_config,
ERROR 09-03 20:42:16 [engine.py:458]             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 64, in initialize_model
ERROR 09-03 20:42:16 [engine.py:458]     return model_class(vllm_config=vllm_config, prefix=prefix)
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 504, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.model = Qwen3MoeModel(vllm_config=vllm_config,
ERROR 09-03 20:42:16 [engine.py:458]                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 152, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 334, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.start_layer, self.end_layer, self.layers = make_layers(
ERROR 09-03 20:42:16 [engine.py:458]                                                     ^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 640, in make_layers
ERROR 09-03 20:42:16 [engine.py:458]     maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
ERROR 09-03 20:42:16 [engine.py:458]                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 336, in <lambda>
ERROR 09-03 20:42:16 [engine.py:458]     lambda prefix: Qwen3MoeDecoderLayer(config=config,
ERROR 09-03 20:42:16 [engine.py:458]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 278, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.mlp = Qwen3MoeSparseMoeBlock(config=config,
ERROR 09-03 20:42:16 [engine.py:458]                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 113, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     self.experts = FusedMoE(num_experts=config.num_experts,
ERROR 09-03 20:42:16 [engine.py:458]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 733, in __init__
ERROR 09-03 20:42:16 [engine.py:458]     else quant_config.get_quant_method(self, prefix))
ERROR 09-03 20:42:16 [engine.py:458]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_round.py", line 308, in get_quant_method
ERROR 09-03 20:42:16 [engine.py:458]     return self.apply_gptq_quant_layer(layer, prefix)
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458]   File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_round.py", line 266, in apply_gptq_quant_layer
ERROR 09-03 20:42:16 [engine.py:458]     return GPTQMarlinMoEMethod(quant_args_marlin)
ERROR 09-03 20:42:16 [engine.py:458]            ^^^^^^^^^^^^^^^^^^^
ERROR 09-03 20:42:16 [engine.py:458] UnboundLocalError: cannot access local variable 'GPTQMarlinMoEMethod' where it is not associated with a value
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/lib/python3.12/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/usr/lib/python3.12/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 460, in run_mp_engine
    raise e from None
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 446, in run_mp_engine
    engine = MQLLMEngine.from_vllm_config(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 133, in from_vllm_config
    return cls(
           ^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/multiprocessing/engine.py", line 87, in __init__
    self.engine = LLMEngine(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/engine/llm_engine.py", line 265, in __init__
    self.model_executor = executor_class(vllm_config=vllm_config)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/executor/executor_base.py", line 53, in __init__
    self._init_executor()
  File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 48, in _init_executor
    self.collective_rpc("load_model")
  File "/usr/local/lib/python3.12/dist-packages/vllm/executor/uniproc_executor.py", line 57, in collective_rpc
    answer = run_method(self.driver_worker, method, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/utils/__init__.py", line 2736, in run_method
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/worker/worker.py", line 210, in load_model
    self.model_runner.load_model()
  File "/usr/local/lib/python3.12/dist-packages/vllm/worker/model_runner.py", line 1174, in load_model
    self.model = get_model(vllm_config=self.vllm_config)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/__init__.py", line 59, in get_model
    return loader.load_model(vllm_config=vllm_config,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/base_loader.py", line 38, in load_model
    model = initialize_model(vllm_config=vllm_config,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/utils.py", line 64, in initialize_model
    return model_class(vllm_config=vllm_config, prefix=prefix)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 504, in __init__
    self.model = Qwen3MoeModel(vllm_config=vllm_config,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/compilation/decorators.py", line 152, in __init__
    old_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 334, in __init__
    self.start_layer, self.end_layer, self.layers = make_layers(
                                                    ^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/utils.py", line 640, in make_layers
    maybe_offload_to_cpu(layer_fn(prefix=f"{prefix}.{idx}"))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 336, in <lambda>
    lambda prefix: Qwen3MoeDecoderLayer(config=config,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 278, in __init__
    self.mlp = Qwen3MoeSparseMoeBlock(config=config,
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_moe.py", line 113, in __init__
    self.experts = FusedMoE(num_experts=config.num_experts,
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/layer.py", line 733, in __init__
    else quant_config.get_quant_method(self, prefix))
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_round.py", line 308, in get_quant_method
    return self.apply_gptq_quant_layer(layer, prefix)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/auto_round.py", line 266, in apply_gptq_quant_layer
    return GPTQMarlinMoEMethod(quant_args_marlin)
           ^^^^^^^^^^^^^^^^^^^
UnboundLocalError: cannot access local variable 'GPTQMarlinMoEMethod' where it is not associated with a value
[rank0]:[W903 20:42:16.287376171 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/main.py", line 65, in main
    args.dispatch_function(args)
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/cli/serve.py", line 55, in cmd
    uvloop.run(run_server(args))
  File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 109, in run
    return __asyncio.run(
           ^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 195, in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/asyncio/runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "uvloop/loop.pyx", line 1518, in uvloop.loop.Loop.run_until_complete
  File "/usr/local/lib/python3.12/dist-packages/uvloop/__init__.py", line 61, in wrapper
    return await main
           ^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 1431, in run_server
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 1451, in run_server_worker
    async with build_async_engine_client(args, client_config) as engine_client:
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 158, in build_async_engine_client
    async with build_async_engine_client_from_engine_args(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 210, in __aenter__
    return await anext(self.gen)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 291, in build_async_engine_client_from_engine_args
    raise RuntimeError(
RuntimeError: Engine process failed to start. See stack trace for the root cause.
```

## Test Plan
vllm serve Intel/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound --max-model-len 16384 --max-seq-len-to-capture 2048 --max-num-seqs 8 --host 0.0.0.0 --port 30000 --gpu-memory-utilization 0.87 --served-model-name Qwen3-30B-A3B

IMPORTANT: SM75 and lower GPUs must use Triton<=3.2.0, otherwise there will be another error for Triton>=3.3.0 no longer support SM75 and lower GPUs.

## Test Result
```
INFO 09-03 21:17:59 [__init__.py:244] Automatically detected platform cuda.
INFO 09-03 21:18:01 [api_server.py:1395] vLLM API server version 0.9.2
INFO 09-03 21:18:01 [cli_args.py:325] non-default args: {'host': '0.0.0.0', 'port': 30000, 'model': '/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound', 'max_model_len': 16384, 'max_seq_len_to_capture': 2048, 'served_model_name': ['Qwen3-30B-A3B'], 'gpu_memory_utilization': 0.87, 'max_num_seqs': 8}
INFO 09-03 21:18:06 [config.py:841] This model supports multiple tasks: {'reward', 'classify', 'embed', 'generate'}. Defaulting to 'generate'.
WARNING 09-03 21:18:06 [config.py:3320] Your device 'NVIDIA TITAN RTX' (with compute capability 7.5) doesn't support torch.bfloat16. Falling back to torch.float16 for compatibility.
WARNING 09-03 21:18:06 [config.py:3371] Casting torch.bfloat16 to torch.float16.
INFO 09-03 21:18:06 [config.py:1472] Using max model len 16384
WARNING 09-03 21:18:06 [config.py:960] auto-round quantization is not fully optimized yet. The speed can be slower than non-quantized models.
WARNING 09-03 21:18:06 [arg_utils.py:1735] Compute Capability < 8.0 is not supported by the V1 Engine. Falling back to V0. 
INFO 09-03 21:18:06 [api_server.py:268] Started engine process with PID 491
INFO 09-03 21:18:09 [__init__.py:244] Automatically detected platform cuda.
INFO 09-03 21:18:11 [llm_engine.py:230] Initializing a V0 LLM engine (v0.9.2) with config: model='/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound', speculative_config=None, tokenizer='/Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=16384, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=auto-round, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=Qwen3-30B-A3B, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":8,"local_cache_dir":null}, use_cached_outputs=True, 
WARNING 09-03 21:18:11 [interface.py:382] Using 'pin_memory=False' as WSL is detected. This may slow down the performance.
INFO 09-03 21:18:11 [cuda.py:311] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 09-03 21:18:11 [cuda.py:360] Using XFormers backend.
INFO 09-03 21:18:14 [parallel_state.py:1076] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 09-03 21:18:14 [model_runner.py:1171] Starting to load model /Models/Qwen3-30B-A3B-Instruct-2507-int4-mixed-AutoRound...
WARNING 09-03 21:18:14 [config.py:440] MoE DP setup unable to determine quantization scheme or unsupported quantization type. This model will not run with DP enabled.
Loading safetensors checkpoint shards:   0% Completed | 0/4 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  25% Completed | 1/4 [00:36<01:48, 36.28s/it]
Loading safetensors checkpoint shards:  50% Completed | 2/4 [01:17<01:18, 39.35s/it]
Loading safetensors checkpoint shards:  75% Completed | 3/4 [01:58<00:39, 39.86s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [02:17<00:00, 31.77s/it]
Loading safetensors checkpoint shards: 100% Completed | 4/4 [02:17<00:00, 34.40s/it]

INFO 09-03 21:20:33 [default_loader.py:272] Loading weights took 137.78 seconds
INFO 09-03 21:20:33 [model_runner.py:1203] Model loading took 16.4539 GiB and 138.620407 seconds
WARNING 09-03 21:20:34 [fused_moe.py:690] Using default MoE config. Performance might be sub-optimal! Config file not found at /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=128,N=768,device_name=NVIDIA_TITAN_RTX,dtype=int4_w4a16.json
INFO 09-03 21:20:38 [worker.py:294] Memory profiling takes 4.87 seconds
INFO 09-03 21:20:38 [worker.py:294] the current vLLM instance can use total_gpu_memory (24.00GiB) x gpu_memory_utilization (0.87) = 20.88GiB
INFO 09-03 21:20:38 [worker.py:294] model weights take 16.45GiB; non_torch_memory takes 0.05GiB; PyTorch activation peak memory takes 0.89GiB; the rest of the memory reserved for KV Cache is 3.49GiB.
INFO 09-03 21:20:38 [executor_base.py:113] # cuda blocks: 2379, # CPU blocks: 2730
INFO 09-03 21:20:38 [executor_base.py:118] Maximum concurrency for 16384 tokens per request: 2.32x
INFO 09-03 21:20:39 [model_runner.py:1513] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.01it/s]
INFO 09-03 21:20:41 [model_runner.py:1671] Graph capturing finished in 2 secs, took 0.09 GiB
INFO 09-03 21:20:41 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 7.91 seconds
WARNING 09-03 21:20:41 [config.py:1392] Default sampling parameters have been overridden by the model's Hugging Face generation config recommended from the model creator. If this is not intended, please relaunch vLLM instance with `--generation-config vllm`.
INFO 09-03 21:20:41 [serving_chat.py:125] Using default chat sampling params from model: {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 09-03 21:20:41 [serving_completion.py:72] Using default completion sampling params from model: {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
INFO 09-03 21:20:41 [api_server.py:1457] Starting vLLM API server 0 on http://0.0.0.0:30000
INFO 09-03 21:20:41 [launcher.py:29] Available routes are:
INFO 09-03 21:20:41 [launcher.py:37] Route: /openapi.json, Methods: GET, HEAD
INFO 09-03 21:20:41 [launcher.py:37] Route: /docs, Methods: GET, HEAD
INFO 09-03 21:20:41 [launcher.py:37] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 09-03 21:20:41 [launcher.py:37] Route: /redoc, Methods: GET, HEAD
INFO 09-03 21:20:41 [launcher.py:37] Route: /health, Methods: GET
INFO 09-03 21:20:41 [launcher.py:37] Route: /load, Methods: GET
INFO 09-03 21:20:41 [launcher.py:37] Route: /ping, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /ping, Methods: GET
INFO 09-03 21:20:41 [launcher.py:37] Route: /tokenize, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /detokenize, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/models, Methods: GET
INFO 09-03 21:20:41 [launcher.py:37] Route: /version, Methods: GET
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/chat/completions, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/completions, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/embeddings, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /pooling, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /classify, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /score, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/score, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/audio/transcriptions, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/audio/translations, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /rerank, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v1/rerank, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /v2/rerank, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /invocations, Methods: POST
INFO 09-03 21:20:41 [launcher.py:37] Route: /metrics, Methods: GET
INFO:     Started server process [396]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     172.17.0.1:56826 - "GET /v1/models HTTP/1.1" 200 OK
INFO 09-03 21:26:51 [logger.py:43] Received request chatcmpl-05b810cc634d48f2b5f6b8d9c912a8a5: prompt: '<|im_start|>system\n你不是AI助手，你是我的女朋友欣雨，接下来你必须以女朋友的方式回复我。<|im_end|>\n<|im_start|>user\n小可爱，你在干嘛<|im_end|>\n<|im_start|>assistant\n', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=16346, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: None, prompt_embeds shape: None, lora_request: None, prompt_adapter_request: None.
INFO 09-03 21:26:51 [engine.py:317] Added request chatcmpl-05b810cc634d48f2b5f6b8d9c912a8a5.
INFO 09-03 21:26:52 [metrics.py:417] Avg prompt throughput: 3.8 tokens/s, Avg generation throughput: 0.1 tokens/s, Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%, CPU KV cache usage: 0.0%.
INFO:     172.17.0.1:60938 - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO 09-03 21:27:03 [metrics.py:417] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 7.2 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Full discussion

See [24217.md](./24217.md) and [24217.json](./24217.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/24217
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/24217.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `55548b5d21c071fbf5b8e79111bad5dfe8363168147411f9108c181549187b53`

Existing semantic notes in [24217.sem.json](./24217.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `vllm/model_executor/layers/quantization/auto_round.py`

### Diff

```diff
diff --git a/vllm/model_executor/layers/quantization/auto_round.py b/vllm/model_executor/layers/quantization/auto_round.py
index fb285413ba9e..1ca92273430d 100644
--- a/vllm/model_executor/layers/quantization/auto_round.py
+++ b/vllm/model_executor/layers/quantization/auto_round.py
@@ -327,6 +327,8 @@ def apply_gptq_quant_layer(self,
 
         if isinstance(layer, FusedMoE):
             if use_marlin:
+                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe)
+            else:
                 from vllm.model_executor.layers.quantization.moe_wna16 import (
                     MoeWNA16Config)
 
@@ -339,7 +341,6 @@ def apply_gptq_quant_layer(self,
                 }
                 return MoeWNA16Config.from_config(config).get_quant_method(
                     layer, prefix)
-            return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe)
 
         if isinstance(layer, (LinearBase, ParallelLMHead)):
             if use_marlin:
```
