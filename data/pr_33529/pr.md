# PR #33529: Triton MLA perf fixes

- field: attention  | area: v1/attention
- author: koush
- state: MERGED  created: 2026-02-02T01:40:05Z  merged: 2026-04-02T13:40:01Z
- url: https://github.com/vllm-project/vllm/pull/33529
- labels: performance, ready, v1, deepseek

## Body

## Purpose

Triton MLA on sm120 performance degrades on batch size 1 as context length increases.

This perf issue has been bugging me for a while as it made deepseek and Kimi k2 unusable, and when Kimi k2.5 was released I finally got around to digging into it. I'm familiar with w/ cuda but this is my first foray into triton. 

The primary issues are suboptimal kv splitting during low batch count resulting in underutilized SM and unnecessary load of Q vector, since c_kv contains both.

## Test Plan

Test Kimi 2.5, Deepseek v2, Qwen 235B.

## Test Result

Models load and generate correctly, Kimi 2.5 shows notable improvement as seen below.

Testing with -dcp 8 and 80k context


| Concurrent Requests | OLD Peak Throughput (tok/s) | NEW Peak Throughput (tok/s) | Peak Throughput Increase (%) |
|---------------------|----------------------------|----------------------------|------------------------------|
| 1 (128 tokens)      | 54.00                      | 55.00                      | 1.85%                        |
| 8 (128 tokens)      | 272.00                     | 272.00                     | 0.00%                        |
| 1 (80K tokens)      | 34.00                      | 52.00                      | 52.94%                       |
| 8 (80K tokens)      | 176.00                     | 184.00                     | 4.55%                        |


---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [X] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [X] The test plan, such as providing test command.
- [X] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### dosubot @ 2026-02-02T01:40:14Z

<!-- Answer -->
**Related Documentation**

No published documentation to review for changes on this repository.

[Write your first living document](https://app.dosu.dev/documents?utm_campaign=pr-doc-check-nudge)


<!-- Dosu Comment Footer -->
<sup>How did I do? Any feedback?</sup>&nbsp;&nbsp;[![Join Discord](https://img.shields.io/badge/join-5865F2?logo=discord&logoColor=white&label=)](https://go.dosu.dev/discord-bot)
---
### mergify @ 2026-02-02T01:43:42Z

Hi @koush, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-02-02T01:52:53Z

Hi @koush, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### koush @ 2026-02-02T01:59:45Z

benchmarks on Kimi k2.5 

Summary

| Context | OLD (tok/s) | NEW (tok/s) | Improvement |
|---------|-------------|-------------|-------------|
| Short (100 in / 400 out) | 69.14 | 72.37 | 1.05x |
| Long (80K in / 400 out) | 5.61 | 24.47 | 4.36x |
```
OLD

============ Serving Benchmark Result ============
Successful requests:                     2         
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  11.57     
Total input tokens:                      200       
Total generated tokens:                  800       
Request throughput (req/s):              0.17      
Output token throughput (tok/s):         69.14     
Peak output token throughput (tok/s):    72.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          86.42     
---------------Time to First Token----------------
Mean TTFT (ms):                          76.94     
Median TTFT (ms):                        76.94     
P99 TTFT (ms):                           81.87     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          14.31     
Median TPOT (ms):                        14.31     
P99 TPOT (ms):                           14.45     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.31     
Median ITL (ms):                         14.15     
P99 ITL (ms):                            20.66     
==================================================



============ Serving Benchmark Result ============
Successful requests:                     2         
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  142.69    
Total input tokens:                      160000    
Total generated tokens:                  800       
Request throughput (req/s):              0.01      
Output token throughput (tok/s):         5.61      
Peak output token throughput (tok/s):    9.00      
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          1126.90   
---------------Time to First Token----------------
Mean TTFT (ms):                          25966.21  
Median TTFT (ms):                        25966.21  
P99 TTFT (ms):                           25980.01  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          113.73    
Median TPOT (ms):                        113.73    
P99 TPOT (ms):                           113.86    
---------------Inter-token Latency----------------
Mean ITL (ms):                           113.73    
Median ITL (ms):                         113.74    
P99 ITL (ms):                            114.33    
==================================================

NEW

============ Serving Benchmark Result ============
Successful requests:                     2         
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  11.06     
Total input tokens:                      200       
Total generated tokens:                  800       
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         72.37     
Peak output token throughput (tok/s):    74.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          90.46     
---------------Time to First Token----------------
Mean TTFT (ms):                          74.98     
Median TTFT (ms):                        74.98     
P99 TTFT (ms):                           78.82     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          13.66     
Median TPOT (ms):                        13.66     
P99 TPOT (ms):                           13.68     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.66     
Median ITL (ms):                         13.66     
P99 ITL (ms):                            14.08     
==================================================

============ Serving Benchmark Result ============
Successful requests:                     2         
Failed requests:                         0         
Maximum request concurrency:             1         
Benchmark duration (s):                  32.69     
Total input tokens:                      160000    
Total generated tokens:                  800       
Request throughput (req/s):              0.06      
Output token throughput (tok/s):         24.47     
Peak output token throughput (tok/s):    26.00     
Peak concurrent requests:                2.00      
Total token throughput (tok/s):          4918.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          490.87    
Median TTFT (ms):                        490.87    
P99 TTFT (ms):                           491.32    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.73     
Median TPOT (ms):                        39.73     
P99 TPOT (ms):                           39.76     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.73     
Median ITL (ms):                         39.72     
P99 ITL (ms):                            40.34     
==================================================


```
---
### koush @ 2026-02-03T19:51:02Z

@tjtanaa I tested -dcp with cuda graph and the model worked fine, but I did notice performance degradation compared to not using cuda graphs. I'm not sure what the expected behavior is supposed to be.

Regardless, since dcp is the recommended way to run it, as it saves kv cache space by preventing data duplication, and cuda graphs reduced performance, I backed out that part of my change.
---
### koush @ 2026-02-03T20:15:45Z

Testing with -dcp 8 and 80k context

| Concurrent Requests | OLD Output Throughput (tok/s) | NEW Output Throughput (tok/s) | OLD Peak Throughput (tok/s) | NEW Peak Throughput (tok/s) | Output Throughput Increase (%) | Peak Throughput Increase (%) |
|--------------------:|------------------------------:|------------------------------:|----------------------------:|----------------------------:|-------------------------------:|-----------------------------:|
| 1                   | 27.32                         | 36.66                         | 34.00                       | 49.00                       | **+34.2%**                     | **+44.1%**                   |
| 8                   | 43.47                         | 50.60                         | 60.00                       | 72.00                       | **+16.4%**                     | **+20.0%**                   |

On main branch, I'm seeing that dcp doesn't degrade as catastrophically on long context where tp alone does. In any case the patch still provides notable performance improvements, particularly when dcp is lower or unset.
---
### koush @ 2026-02-20T16:18:27Z

> gsm8k

I assume you mean this?

This? https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_lm_eval.html


---
### mgehre-amd @ 2026-02-20T16:32:38Z

> > gsm8k
> 
> I assume you mean this?
> 
> This? [docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_lm_eval.html](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_lm_eval.html)

I'm running this like
```
lm_eval --model vllm \
  --model_args pretrained=<model_name>,max_model_len=4096,gpu_memory_utilization=0.8 \
  --tasks gsm8k \
  --limit 200 \
  --batch_size 1
  ```
---
### koush @ 2026-03-27T18:27:12Z

@mgoin 

```
2026-03-27:18:18:44 WARNING  [config.evaluate_config:281] --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.
2026-03-27:18:18:49 INFO     [_cli.run:376] Selected Tasks: ['gsm8k']
2026-03-27:18:18:49 INFO     [evaluator:211] Setting random seed to 0 | Setting numpy seed to 1234 | Setting torch manual seed to 1234 | Setting fewshot manual seed to 1234
2026-03-27:18:18:49 INFO     [evaluator:236] Initializing local-completions model, with arguments: {'model': 'moonshotai/Kimi-K2.5', 'model_name': 'qwen3_coder', 'base_url': 'http://127.0.0.1:8000/v1/completions', 'tokenized_requests': False, 'trust_remote_code': True, 'api_key': 'token-abc123'}
2026-03-27:18:18:49 INFO     [models.openai_completions:42] Remote tokenizer not supported. Using huggingface tokenizer backend.
2026-03-27:18:18:49 INFO     [models.api_models:172] Using max length 2048 - 1
2026-03-27:18:18:49 INFO     [models.api_models:175] Concurrent requests are disabled. To enable concurrent requests, set `num_concurrent` > 1.
2026-03-27:18:18:49 INFO     [models.api_models:193] Using tokenizer huggingface
2026-03-27:18:18:53 INFO     [tasks:700] Selected tasks:
2026-03-27:18:18:53 INFO     [tasks:691] Task: gsm8k (gsm8k/gsm8k.yaml)
2026-03-27:18:18:53 INFO     [evaluator:314] gsm8k: Using gen_kwargs: {'until': ['Question:', '</s>', '<|im_end|>'], 'do_sample': False, 'temperature': 0.0}
2026-03-27:18:18:53 INFO     [api.task:311] Building contexts for gsm8k on rank 0...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 847.50it/s]
2026-03-27:18:18:54 INFO     [evaluator:584] Running generate_until requests
2026-03-27:18:18:54 INFO     [models.api_models:733] Tokenized requests are disabled. Context + generation length is not checked.
Requesting API: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [07:21<00:00,  2.21s/it]
fatal: not a git repository (or any parent up to mount point /mnt)
Stopping at filesystem boundary (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).
2026-03-27:18:26:18 INFO     [loggers.evaluation_tracker:316] Output path not provided, skipping saving results aggregated
local-completions ({'model': 'moonshotai/Kimi-K2.5', 'base_url': 'http://127.0.0.1:8000/v1/completions', 'tokenized_requests': False}), gen_kwargs: ({}), limit: 200.0, num_fewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.955|±  |0.0147|
|     |       |strict-match    |     5|exact_match|↑  |0.955|±  |0.0147|
```

Is this what you need?
---
### mergify @ 2026-04-01T04:24:42Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @koush.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---