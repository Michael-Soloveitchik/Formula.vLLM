# PR #35330: [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement

- field: scheduling  | area: entrypoints/pooling
- author: yewentao256
- state: MERGED  created: 2026-02-25T20:41:55Z  merged: 2026-02-26T17:14:54Z
- url: https://github.com/vllm-project/vllm/pull/35330
- labels: frontend, ready

## Body

## Purpose

Optimize maxsim scores computation for pooling models

Originally it is calculated in CPU, now we calculate it in GPU and using the batched version, so we get a lot of performance improvement

## Test

### Acc

Covered in unit tests

```bash
tests/entrypoints/pooling/score/test_online_colbert.py::TestColBERTOnline::test_score
tests/entrypoints/pooling/score/test_online_colbert.py::TestColBERTOnline::test_rerank
tests/entrypoints/pooling/score/test_online_colbert.py::TestColBERTOnline::test_rerank_top_n
...
```

### Perf

`vllm serve   --model jinaai/jina-colbert-v2   --runner pooling   --port 9256   --enforce-eager   --max-model-len 4096   --max-num-batched-tokens 4096   --disable-log-stats   --hf-overrides '{"architectures": ["ColBERTJinaRobertaModel"]}'   --trust-remote-code`

`vllm bench serve   --model jinaai/jina-colbert-v2   --backend vllm-rerank   --endpoint /v1/rerank   --host 127.0.0.1 --port 9256   --dataset-name random-rerank   --num-prompts 2000   --request-rate inf   --max-concurrency 64   --seed 0   --random-input-len 2048   --random-range-ratio 0.5   --percentile-metrics e2el   --metric-percentiles 50,95,99`

```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  12.41     
Total input tokens:                      4116113   
Request throughput (req/s):              161.14    
Total token throughput (tok/s):          331637.66 
----------------End-to-end Latency----------------
Mean E2EL (ms):                          391.46    
Median E2EL (ms):                        395.45    
P50 E2EL (ms):                           395.45    
P95 E2EL (ms):                           421.21    
P99 E2EL (ms):                           431.76    
==================================================

# main
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  14.15     
Total input tokens:                      4116113   
Request throughput (req/s):              141.38    
Total token throughput (tok/s):          290961.01 
----------------End-to-end Latency----------------
Mean E2EL (ms):                          446.27    
Median E2EL (ms):                        440.32    
P50 E2EL (ms):                           440.32    
P95 E2EL (ms):                           516.98    
P99 E2EL (ms):                           531.58    
==================================================
```

## Comments

### noooop @ 2026-02-26T03:57:42Z

（I’m not sure if it will work when the API server count is greater than 1. I have reservations about using the GPU in the API server (or during pre-processing and post-processing stages). There might be a risk of OOM or other weird CUDA errors. However, computing maxsim scores on the GPU is indeed better.）
---
### mgoin @ 2026-02-26T17:23:59Z

@yewentao256 can you test this with api-server-count>1 without DP? I have concerns about the API server using GPU resources
---
### yewentao256 @ 2026-02-26T17:24:00Z

> （I’m not sure if it will work when the API server count is greater than 1. I have reservations about using the GPU in the API server (or during pre-processing and post-processing stages). There might be a risk of OOM or other weird CUDA errors. However, computing maxsim scores on the GPU is indeed better.）

No worries, we can fix accordingly if the issue raised. With the current config `max_score_matrix_elements=16_000_000` and `batch_size=16` , the GPU memory would take up ~40MB to ~200MB, so I think we don't need to worry about OOM
---
### mgoin @ 2026-02-26T17:28:39Z

> No worries, we can fix accordingly if the issue raised

Waiting for issues to be reported is not a good testing strategy. We should have raised this PR with other contributors before merging
---