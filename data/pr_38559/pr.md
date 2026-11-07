# PR #38559: [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement

- field: scheduling  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2026-03-30T15:46:57Z  merged: 2026-04-01T03:54:58Z
- url: https://github.com/vllm-project/vllm/pull/38559
- labels: ready

## Body

## Purpose

Optimize mean pooling using chunks and index_add, avoid redundant calculation using `cumsum` and large temp tensor

## Test

### Acc

Covered in unit test

```bash
tests/models/language/pooling_mteb_test/mteb_embed_utils.py
tests/entrypoints/pooling/embed/test_correctness_mteb.py
tests/entrypoints/pooling/embed/test_online.py
```

### Perf

`vllm serve intfloat/multilingual-e5-base   --runner pooling   --max-model-len 512   --max-num-batched-tokens 32768`

`vllm bench serve   --model intfloat/multilingual-e5-base   --backend openai-embeddings   --endpoint /v1/embeddings   --dataset-name random   --seed 0   --random-input-len 512   --random-range-ratio 0   --random-batch-size 1   --num-prompts 2000   --request-rate inf   --max-concurrency 64`


```bash
# now
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  7.91      
Total input tokens:                      1024000   
Request throughput (req/s):              252.72    
Total token throughput (tok/s):          129394.87 
----------------End-to-end Latency----------------
Mean E2EL (ms):                          246.51    
Median E2EL (ms):                        254.69    
P99 E2EL (ms):                           276.31    
==================================================
# main
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  8.39      
Total input tokens:                      1024000   
Request throughput (req/s):              238.52    
Total token throughput (tok/s):          122121.47 
----------------End-to-end Latency----------------
Mean E2EL (ms):                          265.40    
Median E2EL (ms):                        259.34    
P99 E2EL (ms):                           824.84    
==================================================
```

## Comments

### noooop @ 2026-04-01T03:52:37Z

Please unblock the Language Models Test (MTEB) and Language Models Test (Extended Pooling) before merging.
---