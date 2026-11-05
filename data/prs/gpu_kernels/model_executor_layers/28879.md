# PR #28879: [Feature] Shared Experts Overlap with FI deepgemm swap kernel, 2.2% throughput improvement and 3.6% TTFT improvement

- field: scheduling  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-11-17T20:55:06Z  merged: 2025-11-21T01:41:49Z
- url: https://github.com/vllm-project/vllm/pull/28879
- labels: ready

## Body

## Purpose

Fix https://github.com/vllm-project/vllm/pull/27134#issuecomment-3514812346, thanks for the context from @robertgshaw2-redhat !

Now it is overlapped, we get 2.2% throughput improvement and 3.6% TTFT improvement for BS 32

For bs 256, we get less improvement but still faster than no overlap.

Origin:

<img width="2376" height="1122" alt="ui perfetto dev_" src="https://github.com/user-attachments/assets/5729a4e2-4208-43e5-8a76-e51582160b41" />

Now

<img width="2382" height="1221" alt="ui perfetto dev_ (1)" src="https://github.com/user-attachments/assets/cd27d8bc-939d-4440-9ced-20b9eab15017" />

## Test

`vllm serve deepseek-ai/DeepSeek-V3.1 -tp 8  --enable-expert-parallel --port 9256`

### Acc

`lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:9256/v1/completions,model=deepseek-ai/DeepSeek-V3.1,num_concurrent=1024" --tasks gsm8k`

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9591|±  |0.0055|
|     |       |strict-match    |     5|exact_match|↑  |0.9591|±  |0.0055|

### Perf

Note that > batch size 256 it will not be used since we have env var `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD`

`vllm bench serve --model deepseek-ai/DeepSeek-V3.1  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 2 --random-output-len 100 --request-rate inf --num-prompts 256 --max-concurrency 32`

#### batch size 32

```bash
now
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  22.26     
Total input tokens:                      256       
Total generated tokens:                  25600     
Request throughput (req/s):              11.50     
Output token throughput (tok/s):         1150.23   
Peak output token throughput (tok/s):    1184.00   
Peak concurrent requests:                64.00     
Total Token throughput (tok/s):          1161.74   
---------------Time to First Token----------------
Mean TTFT (ms):                          51.22     
Median TTFT (ms):                        49.20     
P99 TTFT (ms):                           76.05     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.54     
Median TPOT (ms):                        27.49     
P99 TPOT (ms):                           27.70     
---------------Inter-token Latency----------------
Mean ITL (ms):                           27.54     
Median ITL (ms):                         27.57     
P99 ITL (ms):                            29.97     
==================================================

main
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  22.76     
Total input tokens:                      256       
Total generated tokens:                  25600     
Request throughput (req/s):              11.25     
Output token throughput (tok/s):         1124.89   
Peak output token throughput (tok/s):    1152.00   
Peak concurrent requests:                64.00     
Total Token throughput (tok/s):          1136.14   
---------------Time to First Token----------------
Mean TTFT (ms):                          53.12     
Median TTFT (ms):                        51.15     
P99 TTFT (ms):                           75.23     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.16     
Median TPOT (ms):                        28.13     
P99 TPOT (ms):                           28.31     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.16     
Median ITL (ms):                         28.18     
P99 ITL (ms):                            30.68     
==================================================
```

#### batch size 256

```bash
now
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             256       
Benchmark duration (s):                  8.09      
Total input tokens:                      256       
Total generated tokens:                  25600     
Request throughput (req/s):              31.63     
Output token throughput (tok/s):         3162.51   
Peak output token throughput (tok/s):    3328.00   
Peak concurrent requests:                256.00    
Total Token throughput (tok/s):          3194.13   
---------------Time to First Token----------------
Mean TTFT (ms):                          2166.35   
Median TTFT (ms):                        2016.82   
P99 TTFT (ms):                           4161.67   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.44     
Median TPOT (ms):                        39.43     
P99 TPOT (ms):                           39.51     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.44     
Median ITL (ms):                         39.31     
P99 ITL (ms):                            43.24     
==================================================
main
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             256       
Benchmark duration (s):                  8.16      
Total input tokens:                      256       
Total generated tokens:                  25600     
Request throughput (req/s):              31.35     
Output token throughput (tok/s):         3135.38   
Peak output token throughput (tok/s):    3311.00   
Peak concurrent requests:                256.00    
Total Token throughput (tok/s):          3166.73   
---------------Time to First Token----------------
Mean TTFT (ms):                          2178.18   
Median TTFT (ms):                        2050.07   
P99 TTFT (ms):                           4176.42   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.92     
Median TPOT (ms):                        39.91     
P99 TPOT (ms):                           39.99     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.92     
Median ITL (ms):                         39.69     
P99 ITL (ms):                            48.78     
==================================================
```

## Comments

### yewentao256 @ 2025-11-18T00:45:04Z

Also tried with overlap with with router, but seems we can't get performance improvement.

output len 1024

```bash
# overlap with FI deepgemm
============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  231.65    
Total input tokens:                      256       
Total generated tokens:                  262144    
Request throughput (req/s):              1.11      
Output token throughput (tok/s):         1131.65   
Peak output token throughput (tok/s):    1184.00   
Peak concurrent requests:                64.00     
Total Token throughput (tok/s):          1132.76   
---------------Time to First Token----------------
Mean TTFT (ms):                          51.42     
Median TTFT (ms):                        48.25     
P99 TTFT (ms):                           73.70     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.25     
Median TPOT (ms):                        28.26     
P99 TPOT (ms):                           28.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.25     
Median ITL (ms):                         28.32     
P99 ITL (ms):                            29.19     
==================================================

# overlap with router

============ Serving Benchmark Result ============
Successful requests:                     256       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  232.94    
Total input tokens:                      256       
Total generated tokens:                  262144    
Request throughput (req/s):              1.10      
Output token throughput (tok/s):         1125.39   
Peak output token throughput (tok/s):    1184.00   
Peak concurrent requests:                64.00     
Total Token throughput (tok/s):          1126.49   
---------------Time to First Token----------------
Mean TTFT (ms):                          52.70     
Median TTFT (ms):                        50.58     
P99 TTFT (ms):                           90.59     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.41     
Median TPOT (ms):                        28.40     
P99 TPOT (ms):                           28.52     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.41     
Median ITL (ms):                         28.48     
P99 ITL (ms):                            29.35     
==================================================
```

BS 1

```bash
vllm bench serve --model deepseek-ai/DeepSeek-V3.1  --dataset-name random --host 127.0.0.1 --port 9256 --random-input-len 2 --random-output-len 100 --request-rate 2 --max-concurrency 1 --num_prompts  10
# overlap with FI deepgemm
============ Serving Benchmark Result ============
Successful requests:                     10        
Failed requests:                         0         
Maximum request concurrency:             1         
Request rate configured (RPS):           2.00      
Benchmark duration (s):                  13.38     
Total input tokens:                      10        
Total generated tokens:                  1000      
Request throughput (req/s):              0.75      
Output token throughput (tok/s):         74.75     
Peak output token throughput (tok/s):    77.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          75.50     
---------------Time to First Token----------------
Mean TTFT (ms):                          21.07     
Median TTFT (ms):                        21.06     
P99 TTFT (ms):                           21.84     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.96     
Median TPOT (ms):                        12.96     
P99 TPOT (ms):                           12.99     
---------------Inter-token Latency----------------
Mean ITL (ms):                           12.96     
Median ITL (ms):                         12.96     
P99 ITL (ms):                            13.56     
==================================================

overlap with router
============ Serving Benchmark Result ============
Successful requests:                     10        
Failed requests:                         0         
Maximum request concurrency:             1         
Request rate configured (RPS):           2.00      
Benchmark duration (s):                  13.35     
Total input tokens:                      10        
Total generated tokens:                  1000      
Request throughput (req/s):              0.75      
Output token throughput (tok/s):         74.90     
Peak output token throughput (tok/s):    78.00     
Peak concurrent requests:                2.00      
Total Token throughput (tok/s):          75.65     
---------------Time to First Token----------------
Mean TTFT (ms):                          21.92     
Median TTFT (ms):                        21.87     
P99 TTFT (ms):                           22.78     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          12.92     
Median TPOT (ms):                        12.92     
P99 TPOT (ms):                           13.00     
---------------Inter-token Latency----------------
Mean ITL (ms):                           12.92     
Median ITL (ms):                         12.91     
P99 ITL (ms):                            13.62     
==================================================
```
---
### robertgshaw2-redhat @ 2025-11-18T15:53:34Z

Can you generate a trace?
---
### yewentao256 @ 2025-11-18T16:17:30Z

> Can you generate a trace?

@robertgshaw2-redhat [with router.gz](https://github.com/user-attachments/files/23609596/with.router.gz)
<img width="3084" height="602" alt="image" src="https://github.com/user-attachments/assets/7988362d-3a3e-4ab2-aded-c1643ffb9a4e" />

---