# PR #33612: [Perf] Optimize spec decoding + async scheduling, 1.5% Throughput improvement

- field: scheduling  | area: v1/core
- author: yewentao256
- state: MERGED  created: 2026-02-02T22:52:37Z  merged: 2026-02-04T14:34:32Z
- url: https://github.com/vllm-project/vllm/pull/33612
- labels: ready, v1

## Body

## Purpose

We create `request.spec_token_ids = [-1] * self.num_spec_tokens` each time we call `_update_after_schedule`, which is a waste of resources. We can optimize this using just a place holder num `num_spec_token_placeholders`

## Test

`export MODEL="meta-llama/Llama-3.2-1B-Instruct"`
`export SPEC='{"method":"eagle3","num_speculative_tokens":2,"model":"nm-testing/Llama3_2_1B_speculator.eagle3"}'`
`vllm serve "$MODEL"   --port 9256   -tp 2   --async-scheduling   --speculative-config "$SPEC"   --max-num-seqs 128`


### Perf

```bash
vllm bench serve   --model "$MODEL"   --host 127.0.0.1 --port 9256   --dataset-name random   --random-input-len 2   --random-output-len 512   --request-rate inf   --num-prompts 128   --num-warmups 16
```

```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Benchmark duration (s):                  4.04      
Total input tokens:                      128       
Total generated tokens:                  65536     
Request throughput (req/s):              31.69     
Output token throughput (tok/s):         16224.58  
Peak output token throughput (tok/s):    12004.00  
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          16256.27  
---------------Time to First Token----------------
Mean TTFT (ms):                          100.57    
Median TTFT (ms):                        100.70    
P99 TTFT (ms):                           126.58    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.57      
Median TPOT (ms):                        5.66      
P99 TPOT (ms):                           7.65      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.55      
Median ITL (ms):                         10.33     
P99 ITL (ms):                            11.32     
---------------Speculative Decoding---------------
Acceptance rate (%):                     35.82     
Acceptance length:                       1.72      
Drafts:                                  38158     
Draft tokens:                            76316     
Accepted tokens:                         27335     
Per-position acceptance (%):
  Position 0:                            44.70     
  Position 1:                            26.94     
==================================================

# main
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Benchmark duration (s):                  4.11      
Total input tokens:                      128       
Total generated tokens:                  65536     
Request throughput (req/s):              31.15     
Output token throughput (tok/s):         15950.14  
Peak output token throughput (tok/s):    11906.00  
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          15981.29  
---------------Time to First Token----------------
Mean TTFT (ms):                          118.95    
Median TTFT (ms):                        120.01    
P99 TTFT (ms):                           147.78    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.65      
Median TPOT (ms):                        5.70      
P99 TPOT (ms):                           7.68      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.64      
Median ITL (ms):                         10.46     
P99 ITL (ms):                            11.99     
---------------Speculative Decoding---------------
Acceptance rate (%):                     35.35     
Acceptance length:                       1.71      
Drafts:                                  38362     
Draft tokens:                            76724     
Accepted tokens:                         27123     
Per-position acceptance (%):
  Position 0:                            44.06     
  Position 1:                            26.64     
==================================================
```

### Acc

```bash
lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=$MODEL,num_concurrent=1024" --tasks gsm8k
# now
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3381|±  | 0.013|
|     |       |strict-match    |     5|exact_match|↑  |0.3381|±  | 0.013|
# main
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.3381|±  | 0.013|
|     |       |strict-match    |     5|exact_match|↑  |0.3381|±  | 0.013|
```


## Comments

### mergify @ 2026-02-03T07:07:29Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @yewentao256.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### njhill @ 2026-02-03T17:54:54Z

Thanks @yewentao256, I'm actually surprised that this has such a big impact to performance, since CPU should be overlapped. Perhaps this is a case where the CPU overhangs since the model is so small. What GPU did you run the benchmarks on? It would be cool to see some profiles from these runs.

Re the changes, I feel they are a bit too messy/invasive, it would be good to think of how we can achieve the same thing in a simpler way. A suggestion would be to have a constant `[-1] * num_spec_tokens` list which is reused, and then later instead of doing `del` on the list, just slice it if needed. Since I think the cases where such slicing is needed would actually be rare (and also check to make sure there aren't any other places that we mutate the list).
---
### yewentao256 @ 2026-02-03T20:03:12Z

@njhill Thanks for the review! Updated the code accordingly. This is tested on H200 node.

Profiling:
[async+spec-h200-main.gz](https://github.com/user-attachments/files/25055599/async%2Bspec-h200-main.gz)
[async+spec-h200-now.gz](https://github.com/user-attachments/files/25055600/async%2Bspec-h200-now.gz)

Note: Seems that the bench drops a little bit after the update, but should be still better than main. But we gain in the readability so prefer your idea.

```bash
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Benchmark duration (s):                  4.05      
Total input tokens:                      128       
Total generated tokens:                  65536     
Request throughput (req/s):              31.58     
Output token throughput (tok/s):         16166.99  
Peak output token throughput (tok/s):    12021.00  
Peak concurrent requests:                128.00    
Total token throughput (tok/s):          16198.57  
---------------Time to First Token----------------
Mean TTFT (ms):                          106.01    
Median TTFT (ms):                        104.93    
P99 TTFT (ms):                           132.46    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.65      
Median TPOT (ms):                        5.65      
P99 TPOT (ms):                           7.57      
---------------Inter-token Latency----------------
Mean ITL (ms):                           9.61      
Median ITL (ms):                         10.45     
P99 ITL (ms):                            11.52     
---------------Speculative Decoding---------------
Acceptance rate (%):                     35.17     
Acceptance length:                       1.70      
Drafts:                                  38462     
Draft tokens:                            76924     
Accepted tokens:                         27053     
Per-position acceptance (%):
  Position 0:                            44.56     
  Position 1:                            25.78     
==================================================
```
---
### yewentao256 @ 2026-02-03T20:19:42Z

OK, all addressed @njhill 
---