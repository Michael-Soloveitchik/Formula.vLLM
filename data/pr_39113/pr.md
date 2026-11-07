# PR #39113: [Perf] Optimize redundant sync for pooling model, 3.7% Throughput Improvement

- field: scheduling  | area: v1/worker
- author: yewentao256
- state: MERGED  created: 2026-04-06T21:06:14Z  merged: 2026-04-09T06:12:23Z
- url: https://github.com/vllm-project/vllm/pull/39113
- labels: ready, v1

## Body

## Purpose

Part of https://github.com/vllm-project/vllm/issues/35631

Optimize redundant `_sync_device` for pooling model

## Test

### Acc

Covered in Mteb unit tests

```bash
tests/entrypoints/pooling/embed/test_correctness_mteb.py
tests/entrypoints/pooling/scoring/test_cross_encoder_correctness_mteb.py
tests/models/language/pooling_mteb_test/*.py
...
```

### Perf

`export MODEL="BAAI/bge-base-en-v1.5"`
`vllm serve "$MODEL"   --host 127.0.0.1   --port 8000   --served-model-name "$MODEL"`

`vllm bench serve   --backend openai-embeddings   --endpoint /v1/embeddings   --host 127.0.0.1   --port 8000   --model "$MODEL"   --served-model-name "$MODEL"   --dataset-name random   --random-input-len 32   --random-batch-size 1   --num-prompts 4000   --num-warmups 100   --request-rate inf   --max-concurrency 64   --percentile-metrics e2el   --metric-percentiles 50,90,95,99`

```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     4000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  3.67      
Total input tokens:                      128000    
Request throughput (req/s):              1089.45   
Total token throughput (tok/s):          34862.35  
----------------End-to-end Latency----------------
Mean E2EL (ms):                          56.03     
Median E2EL (ms):                        56.38     
P50 E2EL (ms):                           56.38     
P90 E2EL (ms):                           68.52     
P95 E2EL (ms):                           72.08     
P99 E2EL (ms):                           80.59     
==================================================

# main
============ Serving Benchmark Result ============
Successful requests:                     4000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  3.81      
Total input tokens:                      128000    
Request throughput (req/s):              1050.62   
Total token throughput (tok/s):          33619.77  
----------------End-to-end Latency----------------
Mean E2EL (ms):                          58.28     
Median E2EL (ms):                        60.29     
P50 E2EL (ms):                           60.29     
P90 E2EL (ms):                           69.85     
P95 E2EL (ms):                           73.66     
P99 E2EL (ms):                           81.97     
==================================================
```

## Comments

(no discussion comments)