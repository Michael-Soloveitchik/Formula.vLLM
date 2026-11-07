# PR #28587: [Hybrid][torch.compile] Refactor mamba2 forward to avoid obscuring linear projections under custom op

- field: attention  | area: model_executor/models
- author: tomeras91
- state: MERGED  created: 2025-11-12T20:11:14Z  merged: 2025-11-19T00:49:36Z
- url: https://github.com/vllm-project/vllm/pull/28587
- labels: performance, ready, torch.compile

## Body

## Purpose

Currently, the mamba_mixer2 custom op encapsulates the entire forward pass of the Mamba2 block. This is including: the input projection, the token mixing conv + SSM ops, the gated RMSnorm, and the output proj. 


The reason the custom op exists is to enable piecewise CUDA graphs - the token mixing ops aren't easily CUDA-graphable for a prefill/mixed batch, due input dependent control flow (cu_seqlens for example). This is similar to why we want to break CUDA graphs on Attention ops. Yet, everything except the token mixing ops can be moved out of the custom op since it doesn't have input dependent control flow. This is similar to how the QKV and output projections in Attention are **not** part of the Attention custom op. Rather, the custom op performs just the token mixing Attention op itself. Thus, **this PR makes the Mamba2 forward much more similar to how the Attention forward is implemented.**


The motivation for this change is to allow torch.compile to run on these projections and RMSnorm. These, especially the RMSnorm, are implemented using many torch ops In the default flow (i.e. without activating the `mixer2_gated_rms_norm` custom op). When they are part of the custom op, they are obscured from torch.compile and many simple kernel fusion opportunities are missed. By moving them out of the custom op, we allow Inductor to fuse them.

This is even more important when running quantized models where the RMSnorm can be fused with the quantization op before the projections. I this case, even when enabling the `mixer2_gated_rms_norm` and the `quant` custom ops we don't get the norm + quant fusion since there is no custom pass replacing the `mixer2_gated_rms_norm` -> `quant` pattern with a fused custom kernel.

## Performance effect

This PR boosts overall performance of hybrid models significantly. For [nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8) with ISL/OSL/BS = 1K/1K/256 on a single H100, we observed 10%-15% increase in throughput, and ~22% decrease in TTFT:

```
========================== Serving Benchmark Result Comparison ===========================
Metric                                    Main            PR              Diff (%)
--------------------------------------------------------------------------------------------
Successful requests:                     1280            1280            +0.00%
Failed requests:                         0               0               +0.00%
Maximum request concurrency:             256             256             +0.00%
Benchmark duration (s):                  226.32          201.32          -11.04%
Total input tokens:                      1309444         1309444         +0.00%
Total generated tokens:                  1310720         1310720         +0.00%
Request throughput (req/s):              5.66            6.36            +12.37%
Output token throughput (tok/s):         5791.47         6510.57         +12.42%
Peak output token throughput (tok/s):    7936.00         8704.00         +9.68%
Peak concurrent requests:                288.00          295.00          +2.43%
Total Token throughput (tok/s):          11577.31        13014.81        +12.42%
-------------------------------Time to First Token---------------------------------------
Mean TTFT (ms):                          2021.85         1576.73         -22.02%
Median TTFT (ms):                        1077.14         843.59          -21.69%
P99 TTFT (ms):                           9803.03         7490.96         -23.59%
-------------------------------Time per Output Token (excl. 1st token)-------------------
Mean TPOT (ms):                          42.21           37.77           -10.52%
Median TPOT (ms):                        42.20           37.81           -10.40%
P99 TPOT (ms):                           45.55           40.55           -10.98%
-------------------------------Inter-token Latency---------------------------------------
Mean ITL (ms):                           42.21           37.77           -10.52%
Median ITL (ms):                         33.00           30.74           -6.85%
P99 ITL (ms):                            349.69          273.94          -21.67%
-------------------------------End-to-end Latency----------------------------------------
Mean E2EL (ms):                          45204.97        40210.64        -11.05%
Median E2EL (ms):                        44381.41        39479.33        -11.05%
P99 E2EL (ms):                           55504.54        48002.33        -13.52%
=========================================================================================
``` 

<details>
<summary> Benchmark details </summary>

Server run command:
```
vllm serve <model> --trust-remote-code --no-enable-prefix-caching --async-scheduling --max-num-seqs 2048
```

Bench command:
```
vllm bench serve --backend vllm --model <model> --tokenizer <model> --trust-remote-code \
  --num-prompts 1280 \
  --ignore-eos \
  --num-warmups 20 \
  --max-concurrency 256 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --no-stream \
  --dataset-name random \
  --save-result
```
</details>

The following screenshots, taken from nsys profiles from the benchmark above (decode forward pass), show that the RMS + quant ops after the SSM update and before the output proj are indeed fused now, reducing the kernel count from 15 to 2, and taking ~14.5 microsec instead of ~91.8 microsec:

Profile on `main`:
![main](https://github.com/user-attachments/assets/49256718-4f3a-4560-a2d9-cca45107ad3f)

Profile on PR:
![pr](https://github.com/user-attachments/assets/a6ba0772-2d79-4ebe-8da8-507f3cb852df)


## Test Plan

Validate quality of hybrid models hasn't changed, by running GSM8K eval.

Run command:

```
lm_eval --model vllm --model_args pretrained=<model> --batch_size auto --trust_remote_code --cache_requests true --tasks gsm8k
```

## Test Result

### model: [ibm-granite/granite-4.0-h-tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny)

results on `main`:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7961|±  |0.0111|
|     |       |strict-match    |     5|exact_match|↑  |0.7953|±  |0.0111|

results on PR:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8044|±  |0.0109|
|     |       |strict-match    |     5|exact_match|↑  |0.8036|±  |0.0109|


### model: [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)

results on `main`:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7983|±  |0.0111|
|     |       |strict-match    |     5|exact_match|↑  |0.8650|±  |0.0094|

results on PR:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7961|±  |0.0111|
|     |       |strict-match    |     5|exact_match|↑  |0.8635|±  |0.0095|

**Results are identical up to errorbar**

## Comments

### tdoublep @ 2025-11-12T20:56:49Z

Wow!! This look amazing. Will review carefully asap
---
### ZJY0516 @ 2025-11-14T15:14:28Z

LGTM
---
### mgoin @ 2025-11-17T19:13:40Z

The CI failure seems related, PTAL
```
[2025-11-16T21:42:14Z] (EngineCore_DP0 pid=2149) AssertionError: expected size 4096==4096, stride 1088==1048 at dim=0
[2025-11-16T21:42:14Z] (EngineCore_DP0 pid=2149) This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
```
---
### tomeras91 @ 2025-11-17T19:41:48Z

> The CI failure seems related, PTAL
> 
> ```
> [2025-11-16T21:42:14Z] (EngineCore_DP0 pid=2149) AssertionError: expected size 4096==4096, stride 1088==1048 at dim=0
> [2025-11-16T21:42:14Z] (EngineCore_DP0 pid=2149) This error most often comes from a incorrect fake (aka meta) kernel for a custom op.
> ```

On it
---