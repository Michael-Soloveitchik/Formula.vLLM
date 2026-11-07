# PR #42311: [Model] [Perf] Use flatten for Qwen3.5's GDN output projection

- field: attention  | area: model_executor/layers
- author: rishaps
- state: MERGED  created: 2026-05-11T10:51:46Z  merged: 2026-05-18T08:14:37Z
- url: https://github.com/vllm-project/vllm/pull/42311
- labels: ready, qwen

## Body

## Purpose
For Qwen3.5's GDN output projection we currently use einops's `rearrange(core_attn_out, "... h d -> ... (h d)")` to merge the last 2 dimensions, which causes expensive python-level string parsing and validation on every GDN forward call. We can use the pytorch equivalent `core_attn_out.flatten(-2)` to avoid this.

## Test Plan

Profiler repro on `Qwen/Qwen3.5-0.8B`.


<details>
<summary>CUDA</summary>

```python
.venv/bin/vllm bench latency \
  --model Qwen/Qwen3.5-0.8B \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 64 \
  --language-model-only \
  --input-len 16 \
  --output-len 32 \
  --batch-size 1 \
  --num-iters-warmup 1 \
  --profile \
  --disable-detokenize \
  --enforce-eager \
  --kv-cache-memory-bytes 64M \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_PROFILE_DIR}\",\"torch_profiler_use_gzip\":false,\"torch_profiler_with_stack\":true,\"torch_profiler_dump_cuda_time_total\":true}"
```
</details>

<details>
<summary>CPU</summary>

```python
.venv/bin/vllm bench latency \
  --model Qwen/Qwen3.5-0.8B \
  --trust-remote-code \
  --dtype float32 \
  --max-model-len 16 \
  --language-model-only \
  --input-len 8 \
  --output-len 8 \
  --batch-size 1 \
  --num-iters-warmup 0 \
  --disable-detokenize \
  --gpu-memory-utilization 0.3 \
  --max-num-seqs 1 \
  --enforce-eager \
  --profile \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${VLLM_PROFILE_DIR}\",\"torch_profiler_use_gzip\":false,\"torch_profiler_with_stack\":true,\"ignore_frontend\":true}"
```
</details>

Correctness test: `pytest tests/models/language/generation_ppl_test/test_qwen.py` with the following tweak:
```python
 if model_info.name == "Qwen/Qwen3.5-0.8B":
      vllm_extra_kwargs["language_model_only"] = True
      vllm_extra_kwargs["cpu_offload_gb"] = 10
      vllm_extra_kwargs["kv_cache_memory_bytes"] = 256 * 1024 * 1024

```

## Test Result

CUDA, eager mode:

| Event | Calls | Avg | Min | Max |
|---|---:|---:|---:|---:|
| einops/einops.py(561): rearrange| 576 | 23.261 us | 12.949 us | 103.882 us |
| aten::flatten| 576 | 2.828 us | 1.486 us | 94.331 us |

Per-call average speedup: `8.23x`

CPU, eager mode:

| Event | Calls | Avg | Min | Max |
|---|---:|---:|---:|---:|
|einops/einops.py(561): rearrange| 144 | 41.741 us | 29.737 us | 106.243 us |
|aten::flatten| 144 | 1.949 us | 1.203 us | 19.260 us |

Per-call average speedup: `21.41x`

Correctness, CUDA only:

| Method | our PPL | HF expected PPL | Relative difference |
|---|---:|---:|---:|
| einops.rearrange | 19.509801864624023 | 19.38858413696289 | 0.6252% |
| flatten | 19.510005950927734 | 19.38858413696289 | 0.6263% |

## Comments

### vadiklyutiy @ 2026-05-12T01:17:30Z

The description a little bit unclear. 
What do you reference to with CUDA and CPU. 
I don't understand correctness results.
---
### vadiklyutiy @ 2026-05-12T01:22:06Z

Could you please try running the e2e tests with a concurrency value chosen so that the batch size exceeds the default maximum CUDA graph capture size?

---
### rishaps @ 2026-05-12T05:02:19Z

> The description a little bit unclear. What do you reference to with CUDA and CPU. I don't understand correctness results.

CUDA: RTX 3050, 4 GB VRAM, driver 580.142, CUDA 13.0, PyTorch 2.11.0+cu130.

CPU: 11th Gen Intel Core i7-11800H, 8 cores / 16 threads, Linux 6.17.0-23-generic, glibc 2.39, PyTorch 2.11.0+cpu.

Regarding correctness, it verifies that replacing the output-projection shape merge with `flatten(-2)` does not change model behavior. Since we only do a tensor reshape before `out_proj`, if the reshape changed the tensor order incorrectly, the logits would change and the model PPL would move noticeably.

> Could you please try running the e2e tests with a concurrency value chosen so that the batch size exceeds the default maximum CUDA graph capture size?

Results on a RTX 4090 24GB vram

Serve command:
```
VLLM_LOGGING_LEVEL=INFO .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --host 127.0.0.1 --port 1800 --trust-remote-code --dtype float16 --max-model-len 64 --language-model-only --max-num-seqs 600 --max-num-batched-tokens 1200 --gpu-memory-utilization 0.80 --disable-uvicorn-access-log --cudagraph-metrics --enable-logging-iteration-details
```
Bench command:
```
.venv/bin/vllm bench serve --backend openai --base-url http://127.0.0.1:1800 --endpoint /v1/completions --model Qwen/Qwen3.5-0.8B --dataset-name random --random-input-len 8 --random-output-len 8 --random-range-ratio 0 --num-prompts 600 --request-rate inf --max-concurrency 600 --ignore-eos --disable-tqdm --save-result
```
main:
```
============ Serving Benchmark Result ============
Successful requests:                     600
Failed requests:                         0
Maximum request concurrency:             600
Benchmark duration (s):                  2.76
Total input tokens:                      4800
Total generated tokens:                  4800
Request throughput (req/s):              217.38
Output token throughput (tok/s):         1739.06
Peak output token throughput (tok/s):    2882.00
Peak concurrent requests:                600.00
Total token throughput (tok/s):          3478.12
---------------Time to First Token----------------
Mean TTFT (ms):                          1451.10
Median TTFT (ms):                        1421.80
P99 TTFT (ms):                           2607.28
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          168.49
Median TPOT (ms):                        180.94
P99 TPOT (ms):                           186.83
---------------Inter-token Latency----------------
Mean ITL (ms):                           168.77
Median ITL (ms):                         125.35
P99 ITL (ms):                            517.86
==================================================
```

PR:
```
============ Serving Benchmark Result ============
Successful requests:                     600
Failed requests:                         0
Maximum request concurrency:             600
Benchmark duration (s):                  2.65
Total input tokens:                      4800
Total generated tokens:                  4800
Request throughput (req/s):              226.51
Output token throughput (tok/s):         1812.05
Peak output token throughput (tok/s):    2879.00
Peak concurrent requests:                600.00
Total token throughput (tok/s):          3624.09
---------------Time to First Token----------------
Mean TTFT (ms):                          1464.64
Median TTFT (ms):                        1431.07
P99 TTFT (ms):                           2579.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          150.87
Median TPOT (ms):                        163.63
P99 TPOT (ms):                           168.46
---------------Inter-token Latency----------------
Mean ITL (ms):                           151.23
Median ITL (ms):                         112.51
P99 ITL (ms):                            484.87
==================================================

```
---
### ZJY0516 @ 2026-05-12T06:12:35Z

But for decode, we use cudagraph by default, which means we don't have "expensive python-level string parsing"
---
### rishaps @ 2026-05-12T07:09:24Z

> But for decode, we use cudagraph by default, which means we don't have "expensive python-level string parsing"

The per-call overhead is stilll present in eager mode (as shown in pr description).
---
### ZJY0516 @ 2026-05-12T07:10:46Z

> VLLM_LOGGING_LEVEL=INFO .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --host 127.0.0.1 --port 18000 --trust-remote-code --dtype float16 --max-model-len 64 --language-model-only --max-num-seqs 600 --max-num-batched-tokens 1200 --gpu-memory-utilization 0.80 --disable-uvicorn-access-log --cudagraph-metrics --enable-logging-iteration-details

But you didn't test in eager mode(for decode)
---
### ZJY0516 @ 2026-05-12T07:15:19Z

I just want to understand where the tpot improvements come from
---
### rishaps @ 2026-05-12T08:45:03Z

> > VLLM_LOGGING_LEVEL=INFO .venv/bin/vllm serve Qwen/Qwen3.5-0.8B --host 127.0.0.1 --port 18000 --trust-remote-code --dtype float16 --max-model-len 64 --language-model-only --max-num-seqs 600 --max-num-batched-tokens 1200 --gpu-memory-utilization 0.80 --disable-uvicorn-access-log --cudagraph-metrics --enable-logging-iteration-details
> 
> But you didn't test in eager mode(for decode)

Sorry, my mistake. 

main:

```
============ Serving Benchmark Result ============
Successful requests:                     600
Failed requests:                         0
Maximum request concurrency:             600
Benchmark duration (s):                  1.38
Total input tokens:                      4800
Total generated tokens:                  4800
Request throughput (req/s):              435.66
Output token throughput (tok/s):         3485.32
Peak output token throughput (tok/s):    3538.00
Peak concurrent requests:                600.00
Total token throughput (tok/s):          6970.63
---------------Time to First Token----------------
Mean TTFT (ms):                          680.87
Median TTFT (ms):                        634.72
P99 TTFT (ms):                           1159.16
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          88.38
Median TPOT (ms):                        95.15
P99 TPOT (ms):                           115.69
---------------Inter-token Latency----------------
Mean ITL (ms):                           88.48
Median ITL (ms):                         48.54
P99 ITL (ms):                            198.62
==================================================
```

PR:
 ```
============ Serving Benchmark Result ============
Successful requests:                     600
Failed requests:                         0
Maximum request concurrency:             600
Benchmark duration (s):                  1.13
Total input tokens:                      4800
Total generated tokens:                  4800
Request throughput (req/s):              531.50
Output token throughput (tok/s):         4251.99
Peak output token throughput (tok/s):    2878.00
Peak concurrent requests:                600.00
Total token throughput (tok/s):          8503.99
---------------Time to First Token----------------
Mean TTFT (ms):                          568.95
Median TTFT (ms):                        577.61
P99 TTFT (ms):                           884.23
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          67.59
Median TPOT (ms):                        70.61
P99 TPOT (ms):                           92.63
---------------Inter-token Latency----------------
Mean ITL (ms):                           68.35
Median ITL (ms):                         44.84
P99 ITL (ms):                            167.74
==================================================
```

---
### rishaps @ 2026-05-16T04:00:24Z

@ZJY0516, could you PTAL?  Thanks.
---
### ZJY0516 @ 2026-05-17T08:21:52Z

@claude review
---
### rishaps @ 2026-05-18T03:18:18Z

@ZJY0516, should I rebase or can we merge it as is?
---