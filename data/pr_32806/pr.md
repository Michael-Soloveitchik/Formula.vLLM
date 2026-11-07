# PR #32806: [torch.compile] Compile `CustomOp.forward_native` for `SiluAndMul` and `QuantFP8` to avoid raw torch ops inside opaque custom ops

- field: quantization  | area: model_executor/layers
- author: ProExpertProg
- state: MERGED  created: 2026-01-21T20:11:26Z  merged: 2026-01-23T03:52:26Z
- url: https://github.com/vllm-project/vllm/pull/32806
- labels: ready, torch.compile, ready-run-all-tests

## Body

## Purpose
When `CustomOp.forward_native` is invoked from within another opaque torch custom op (e.g. `fused_moe`, `unified_attention`), it is hidden from the model-level torch.compile. In this case, executing `forward_native` directly executes eager PyTorch ops, which can significantly hurt performance.

In this PR, we compile forward_native manually to avoid this scenario. Existing `CustomOp` invocations visible to model-level compilation are not affected as `torch.compile` is ignored when nested inside another compiled region.

This will specifically help the shared expert in llama4 and DeepSeek that currently use `SiluMul` and `QuantFP8` for activation and quantization inside the `fused_moe` torch custom op. We get 15% latency improvement for llama4 and 3% for DSR1-fp4. Note that there's very little benefit from just using CUDA kernels for the custom ops (they are slower than compiled native).

We only apply the compilation to `SiluAndMul` and `QuantFP8` custom ops to avoid an avalanche of compilation issues. This is a temporary solution until we transition these ops to vLLM IR. Also, once layers using `CustomOp` for OOT pluggability transition to `PluggableLayer`, it might also be easier to apply this for all ops by default.

Profile view before and after for the llama4 expert:

<img width="1363" height="66" alt="Screenshot 2026-01-21 at 2 58 26 PM" src="https://github.com/user-attachments/assets/86d09b17-57ea-4992-92f9-10278d02529f" />

<img width="1062" height="50" alt="Screenshot 2026-01-21 at 2 58 30 PM" src="https://github.com/user-attachments/assets/313b029b-aed5-4dd9-af93-564ddab44430" />

### Llama4

Average latency improvement of 15.8%.

#### This PR
```sh
$ vllm bench latency --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --max_num_seqs=256 --max_model_len=1024

Avg latency: 1.3004933385488888 seconds
10% percentile latency: 1.2667980642523617 seconds
25% percentile latency: 1.2849869856145233 seconds
50% percentile latency: 1.2978113484568894 seconds
75% percentile latency: 1.314628697000444 seconds
90% percentile latency: 1.334548356430605 seconds
99% percentile latency: 1.3555138180823996 seconds

COLD STARTUP:
Avg total startup time: 168.02 seconds
Avg compilation time:   63.54 seconds

WARM STARTUP:
Avg total startup time: 111.25 seconds
Avg compilation time:   10.60 seconds
```

#### Main

```sh
$ vllm bench latency --model nvidia/Llama-4-Scout-17B-16E-Instruct-FP8 --max_num_seqs=256 --max_model_len=1024
Avg latency: 1.5057966923185935 seconds
10% percentile latency: 1.4795922364108265 seconds
25% percentile latency: 1.5009765535360202 seconds
50% percentile latency: 1.5098670485895127 seconds
75% percentile latency: 1.5172874061390758 seconds
90% percentile latency: 1.5234768875408917 seconds
99% percentile latency: 1.528884345544502 seconds

COLD STARTUP:
Avg total startup time: 169.51 seconds
Avg compilation time:   64.47 seconds

WARM STARTUP:
Avg total startup time: 105.80 seconds
Avg compilation time:   9.98 seconds
```

### DSR1-nvfp4

Average latency improvement of 3.7% (only silu is sped up in this case).

#### This PR
```
$ vllm bench throughput --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel
Throughput: 19.84 requests/s, 22860.72 total tokens/s, 2540.08 output tokens/s
Throughput: 19.88 requests/s, 22905.11 total tokens/s, 2545.01 output tokens/s
Throughput: 19.85 requests/s, 22863.03 total tokens/s, 2540.34 output tokens/s

$ vllm bench latency --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel
Avg latency: 2.200703702463458 seconds
10% percentile latency: 2.1833375917747615 seconds
25% percentile latency: 2.1868423947598785 seconds
50% percentile latency: 2.1954199643805623 seconds
75% percentile latency: 2.204060160322115 seconds
90% percentile latency: 2.2155614748597148 seconds
99% percentile latency: 2.2779232540261 seconds

$ lm_eval --model local-completions --model_args pretrained=nvidia/DeepSeek-R1-NVFP4,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=50,max_retries=3 --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 100
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.96|±  |0.0197|
|     |       |strict-match    |     5|exact_match|↑  | 0.96|±  |0.0197|
```

#### Main
```
$ vllm bench throughput --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel
Throughput: 19.64 requests/s, 22623.96 total tokens/s, 2513.77 output tokens/s
Throughput: 19.83 requests/s, 22838.97 total tokens/s, 2537.66 output tokens/s
Throughput: 19.85 requests/s, 22869.88 total tokens/s, 2541.10 output tokens/s

$ vllm bench latency --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel
Avg latency: 2.282624842195461 seconds
10% percentile latency: 2.2656339406967163 seconds
25% percentile latency: 2.277342037530616 seconds
50% percentile latency: 2.2847984745167196 seconds
75% percentile latency: 2.291269538924098 seconds
90% percentile latency: 2.297466269880533 seconds
99% percentile latency: 2.300187219269574 seconds

# Custom ops instead of compiling native
$ vllm bench latency --model=nvidia/DeepSeek-R1-NVFP4 -tp=4 --enable-expert-parallel -cc.custom_ops+=+silu_and_mul -cc.custom_ops+=+quant_fp8
Avg latency: 2.2708569430125256 seconds
10% percentile latency: 2.261239434592426 seconds
25% percentile latency: 2.2650694884359837 seconds
50% percentile latency: 2.2697742921300232 seconds
75% percentile latency: 2.2773161833174527 seconds
90% percentile latency: 2.2821202569641175 seconds
99% percentile latency: 2.2854324592091144 seconds

$ lm_eval --model local-completions --model_args pretrained=nvidia/DeepSeek-R1-NVFP4,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=50,max_retries=3 --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 100
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.96|±  |0.0197|
|     |       |strict-match    |     5|exact_match|↑  | 0.96|±  |0.0197|
```

### llama3.1-fp8
Shouldn't be affected, latency within 1.59400e-3.

#### This PR
```
$ vllm bench latency --model=RedHatAI/Meta-Llama-3.1-8B-FP8
Avg latency: 0.36803547097370026 seconds
10% percentile latency: 0.36711904434487225 seconds
25% percentile latency: 0.36740035680122674 seconds
50% percentile latency: 0.3678689650259912 seconds
75% percentile latency: 0.36835682461969554 seconds
90% percentile latency: 0.36936185443773867 seconds
99% percentile latency: 0.37035379780456423 seconds
```
#### Main
```
$ vllm bench latency --model=RedHatAI/Meta-Llama-3.1-8B-FP8
Avg latency: 0.36802960286537806 seconds
10% percentile latency: 0.3674700011499226 seconds
25% percentile latency: 0.36762360110878944 seconds
50% percentile latency: 0.36778497509658337 seconds
75% percentile latency: 0.36802939465269446 seconds
90% percentile latency: 0.36876016585156324 seconds
99% percentile latency: 0.37217607663944363 seconds
```

## Test Plan
CI, eval

## Test Result
Looks good

## Comments

### ProExpertProg @ 2026-01-21T20:29:08Z

@aprendis543 can you disable these automatic comments please?
---
### ElizaWszola @ 2026-01-22T06:30:35Z

LGTM, nice speedups
---