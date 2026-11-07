# PR #36795: [Perf] Enable dual stream execution of input projection for Qwen3

- field: quantization  | area: model_executor/models
- author: xyang16
- state: CLOSED  created: 2026-03-11T14:39:56Z  merged: 2026-03-18T03:13:27Z
- url: https://github.com/vllm-project/vllm/pull/36795
- labels: ready, qwen

## Body


## Purpose

This PR Enable dual stream execution of input projection for Qwen3 Next.
* Parallelize the execution of `in_proj_qkvz` and `in_proj_ba` in 2 streams, because their outputs are independent.
* Wrap the implementation in custom op for torch.compile.

## Profiling

Main:

<img width="1722" height="180" alt="Screenshot 2026-03-10 at 6 08 16 PM" src="https://github.com/user-attachments/assets/58f0249e-c9de-47a6-8f00-218781344132" />

PR:

<img width="1723" height="139" alt="Screenshot 2026-03-10 at 6 08 31 PM" src="https://github.com/user-attachments/assets/10cd4f6b-df85-42b2-b164-92aab942c98b" />

Main: nvjet_tst_64x8_64x16_4x2_h_bz_TNT (in_proj_qkvz) and nvjet_tst_64x8_64x16_1x2_h_bz_TNT (in_proj_ba) kernels launched sequentially.

PR: kernels launched in parallel.

## Benchmarking

Benchmarked on H200.

* Qwen3

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  121.46    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              7.90      
Output token throughput (tok/s):         2371.20   
Peak output token throughput (tok/s):    2640.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4175.46   
---------------Time to First Token----------------
Mean TTFT (ms):                          176.72    
Median TTFT (ms):                        193.47    
P99 TTFT (ms):                           222.77    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.18      
Median TPOT (ms):                        6.15      
P99 TPOT (ms):                           6.48      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.18      
Median ITL (ms):                         6.13      
P99 ITL (ms):                            6.94      
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  120.20    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              7.99      
Output token throughput (tok/s):         2396.09   
Peak output token throughput (tok/s):    2672.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          4219.28   
---------------Time to First Token----------------
Mean TTFT (ms):                          191.90    
Median TTFT (ms):                        214.37    
P99 TTFT (ms):                           249.63    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          6.06      
Median TPOT (ms):                        6.04      
P99 TPOT (ms):                           6.40      
---------------Inter-token Latency----------------
Mean ITL (ms):                           6.06      
Median ITL (ms):                         6.02      
P99 ITL (ms):                            6.72      
==================================================
```

* Qwen3 fp8

```
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  195.14    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              4.92      
Output token throughput (tok/s):         1475.89   
Peak output token throughput (tok/s):    1648.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2641.98   
---------------Time to First Token----------------
Mean TTFT (ms):                          162.16    
Median TTFT (ms):                        156.37    
P99 TTFT (ms):                           234.19    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.33     
Median TPOT (ms):                        10.31     
P99 TPOT (ms):                           10.77     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.33     
Median ITL (ms):                         10.24     
P99 ITL (ms):                            11.43     
==================================================
```

PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  191.31    
Total input tokens:                      219140    
Total generated tokens:                  288000    
Request throughput (req/s):              5.02      
Output token throughput (tok/s):         1505.38   
Peak output token throughput (tok/s):    1712.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2650.83   
---------------Time to First Token----------------
Mean TTFT (ms):                          236.09    
Median TTFT (ms):                        236.95    
P99 TTFT (ms):                           380.30    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.87      
Median TPOT (ms):                        9.83      
P99 TPOT (ms):                           10.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.87      
Median ITL (ms):                         9.81      
P99 ITL (ms):                            10.99     
==================================================
```

* Qwen3.5

```
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 1 \
    --max-num-seqs 16 \
    --no-enable-prefix-caching
```

```
vllm bench serve \
        --model Qwen/Qwen3.5-35B-A3B \
        --dataset-name sharegpt \
        --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json \
        --sharegpt-output-len 300 \
        --num-prompts ${num_prompts} \
        --max-concurrency 16 \
        --num-warmups 50 \
        --ignore-eos \
        --temperature 0
```

Main:
```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  197.96    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              4.85      
Output token throughput (tok/s):         1454.81   
Peak output token throughput (tok/s):    1648.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2604.25   
---------------Time to First Token----------------
Mean TTFT (ms):                          142.74    
Median TTFT (ms):                        152.15    
P99 TTFT (ms):                           199.72    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.55     
Median TPOT (ms):                        10.59     
P99 TPOT (ms):                           11.10     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.55     
Median ITL (ms):                         10.24     
P99 ITL (ms):                            12.05     
==================================================
```


PR:

```
============ Serving Benchmark Result ============
Successful requests:                     960       
Failed requests:                         0         
Maximum request concurrency:             16        
Benchmark duration (s):                  190.61    
Total input tokens:                      227546    
Total generated tokens:                  288000    
Request throughput (req/s):              5.04      
Output token throughput (tok/s):         1510.93   
Peak output token throughput (tok/s):    1715.00   
Peak concurrent requests:                32.00     
Total token throughput (tok/s):          2704.71   
---------------Time to First Token----------------
Mean TTFT (ms):                          169.85    
Median TTFT (ms):                        173.71    
P99 TTFT (ms):                           254.99    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          10.05     
Median TPOT (ms):                        10.06     
P99 TPOT (ms):                           10.28     
---------------Inter-token Latency----------------
Mean ITL (ms):                           10.05     
Median ITL (ms):                         10.06     
P99 ITL (ms):                            11.37     
==================================================
```

## Accuracy Testing

* Qwen3

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8575|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8150|±  |0.0107|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8552|±  |0.0097|
|     |       |strict-match    |     5|exact_match|↑  |0.8082|±  |0.0108|
```

* Qwen3 fp8

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3-Next-80B-A3B-Instruct-FP8,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8491|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

PR:

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8575|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8127|±  |0.0107|
```

* Qwen3.5

```
python3 -m lm_eval --model local-completions \
  --model_args model=Qwen/Qwen3.5-35B-A3B,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16 \
  --tasks gsm8k
```

Main:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8476|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8370|±  |0.0102|
```

PR:
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8499|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8332|±  |0.0103|
```

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mergify[bot] @ 2026-03-12T08:08:51Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @xyang16.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

### ZJY0516 @ 2026-03-12T15:57:17Z

Could you please also apply this to qwen 3.5?

### robertgshaw2-redhat @ 2026-03-12T18:37:01Z

I would avoid passing the `aux_stream` through the class constructors

### benchislett @ 2026-03-13T01:39:12Z

consider leveraging a maybe_execute_in_parallel primitive as in https://github.com/vllm-project/vllm/pull/35968

### xyang16 @ 2026-03-13T01:46:34Z

> Could you please also apply this to qwen 3.5?

@ZJY0516 Thanks for review! I have put the benchmark and accuracy testing result in PR description.

### xyang16 @ 2026-03-13T03:15:22Z

@robertgshaw2-redhat Thanks for review! I have removed passing `aux_stream` the class constructors.



### ZJY0516 @ 2026-03-13T03:16:45Z

> > Could you please also apply this to qwen 3.5?
> 
> @ZJY0516 Thanks for review! I have put the benchmark and accuracy testing result in PR description.

I don't see qwen 3.5 related code change

### mergify[bot] @ 2026-03-13T03:50:49Z

Hi @xyang16, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>

### xyang16 @ 2026-03-13T04:17:40Z

> > > Could you please also apply this to qwen 3.5?
> > 
> > 
> > @ZJY0516 Thanks for review! I have put the benchmark and accuracy testing result in PR description.
> 
> I don't see qwen 3.5 related code change

I have pushed the change in qwen 3.5. Thanks!

### xyang16 @ 2026-03-13T04:20:01Z

> consider leveraging a maybe_execute_in_parallel primitive as in #35968

@benchislett I have added `maybe_execute_in_parallel`. Thanks!

### xyang16 @ 2026-03-13T18:20:45Z

@robertgshaw2-redhat Could you please review again? Thanks!

### jhaotingc @ 2026-03-13T18:23:11Z

Thanks for the implementation! https://github.com/vllm-project/vllm/issues/32828
