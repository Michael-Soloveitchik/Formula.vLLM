# PR #42849: [Perf] Add do_not_specialize in fused FP8 RoPE kernel

- field: gpu_kernels  | area: v1/attention
- author: xyang16
- state: MERGED  created: 2026-05-16T20:29:03Z  merged: 2026-05-18T08:32:47Z
- url: https://github.com/vllm-project/vllm/pull/42849
- labels: performance, ready, v1

## Body




## Purpose

This PR add `do_not_specialize` in `_fused_inv_rope_fp8_quant_per_head` kernel, to prevent unnecessary recompilation of the kernel during inference when num_tokens are seen for the first time.

e2e TTFT improves 9% on H200.

## Benchmark

```
rm -rf ~/.triton/cache

vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 8 \
  --max-num-seqs 8 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching
```

```
vllm bench serve \
        --model deepseek-ai/DeepSeek-V4-Flash \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts 480 \
        --max-concurrency 8 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```
Main:

```
============ Serving Benchmark Result ============
Successful requests:                     480       
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  211.37    
Total input tokens:                      106703    
Total generated tokens:                  144000    
Request throughput (req/s):              2.27      
Output token throughput (tok/s):         681.27    
Peak output token throughput (tok/s):    792.00    
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1186.09   
---------------Time to First Token----------------
Mean TTFT (ms):                          370.14    
Median TTFT (ms):                        339.15    
P99 TTFT (ms):                           632.74    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.54     
Median TPOT (ms):                        10.41     
P99 TPOT (ms):                           11.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.54     
Median ITL (ms):                         10.12     
P99 ITL (ms):                            15.20     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     480       
Failed requests:                         0         
Maximum request concurrency:             8         
Benchmark duration (s):                  209.31    
Total input tokens:                      106703    
Total generated tokens:                  144000    
Request throughput (req/s):              2.29      
Output token throughput (tok/s):         687.99    
Peak output token throughput (tok/s):    784.00    
Peak concurrent requests:                16.00     
Total token throughput (tok/s):          1197.78   
---------------Time to First Token----------------
Mean TTFT (ms):                          338.37    
Median TTFT (ms):                        329.73    
P99 TTFT (ms):                           790.74    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.53     
Median TPOT (ms):                        10.40     
P99 TPOT (ms):                           11.50     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.53     
Median ITL (ms):                         10.10     
P99 ITL (ms):                            15.14     
==================================================
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### zyongye @ 2026-05-16T20:52:21Z

The triton will cache the generated kernel right? So re-compile will only happen once when the engine startup? If that's the case I don't think there's need not to specialize. 
---
### zyongye @ 2026-05-16T20:53:49Z

Unless we can prove that not specialized kernel has perf benefit over specialized one (with cache so no recompilation is happening), I am lean towards not adding this flag. 
---
### xyang16 @ 2026-05-16T21:20:05Z

> The triton will cache the generated kernel right? So re-compile will only happen once when the engine startup? If that's the case I don't think there's need not to specialize.

@zyongye Thanks for review!

Yes triton will cache the kernels. But in production, adding `do_not_specialize` will give steady perf. Also `do_not_specialize` is already used in many triton kernels, like https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/sample/rejection_sampler.py#L707
---
### zyongye @ 2026-05-16T21:35:53Z

> > The triton will cache the generated kernel right? So re-compile will only happen once when the engine startup? If that's the case I don't think there's need not to specialize.
> 
> @zyongye Thanks for review!
> 
> Yes triton will cache the kernels. But in production, adding `do_not_specialize` will give steady perf. Also `do_not_specialize` is already used in many triton kernels, like https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/sample/rejection_sampler.py#L707

Thank you! Could you post microbenchmark on the perf comparing specialize/not specialize? Also do you know when triton decide to specialize a kernel given the value?
---
### xyang16 @ 2026-05-16T21:51:07Z

> > > The triton will cache the generated kernel right? So re-compile will only happen once when the engine startup? If that's the case I don't think there's need not to specialize.
> > 
> > 
> > @zyongye Thanks for review!
> > Yes triton will cache the kernels. But in production, adding `do_not_specialize` will give steady perf. Also `do_not_specialize` is already used in many triton kernels, like https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/sample/rejection_sampler.py#L707
> 
> Thank you! Could you post microbenchmark on the perf comparing specialize/not specialize? Also do you know when triton decide to specialize a kernel given the value?

Sure I will add a microbenchmark script for this kernel.

Triton specializes `tl.constexpr` arguments and some scalar arguments etc. In my benchmark, I can see the kernel recompiled when `num_tokens` is seen for the first time, for example num_tokens=62.
---
### xyang16 @ 2026-05-16T22:44:24Z

@zyongye I have added a microbench script. Please take a look. Thanks!

```
python3 benchmarks/kernels/benchmark_fused_inv_rope_fp8_quant.py
```

Main:

```
    num_tokens    vLLM (us)
0          1.0    70.015997
1          2.0    69.359999
2          4.0    69.664001
3          8.0    69.375999
4         16.0    70.079997
5         32.0    69.792002
6         64.0    69.856003
7        128.0    69.728002
8        256.0    70.704002
9        512.0    81.983998
10      1024.0   152.352005
11      2048.0   294.687986
12      4096.0   579.023987
13      8192.0  1144.351959
```

PR:

```
    num_tokens    vLLM (us)
0          1.0    70.400000
1          2.0    69.936004
2          4.0    69.920003
3          8.0    68.768002
4         16.0    67.808002
5         32.0    68.928003
6         64.0    68.896003
7        128.0    69.983996
8        256.0    69.664001
9        512.0    81.759997
10      1024.0   152.416006
11      2048.0   294.880003
12      4096.0   580.160022
13      8192.0  1145.440042
```
---
### xyang16 @ 2026-05-17T17:05:59Z

> Thank you. Given the perf benchmark I think it make sense no to specialize. Could you also remove the benchmark file?

I have removed benchmark file. Thanks!
---