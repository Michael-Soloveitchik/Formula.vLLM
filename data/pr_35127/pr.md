# PR #35127: [Perf] Optimize pooling model redundant copy, 1.8% throughput improvement

- field: scheduling  | area: v1/worker
- author: yewentao256
- state: MERGED  created: 2026-02-23T18:13:45Z  merged: 2026-02-24T12:13:12Z
- url: https://github.com/vllm-project/vllm/pull/35127
- labels: ready, v1

## Body

## Purpose

There is a redundant full copy of pooling model output, then pick up the items that are finished.

We can actually only copy the items that are finished to optimize

## Test

### Acc

Covered in unit test

```bash
pytest -q \
  tests/v1/e2e/test_pooling_chunked_prefill.py::test_pooling_chunked_prefill \
  tests/v1/e2e/test_pooling_chunked_prefill.py::test_pooling_prefix_cache

pytest -q \
  tests/entrypoints/pooling/basic/test_encode.py \
  tests/entrypoints/pooling/embed/test_offline.py \
  tests/entrypoints/pooling/score/test_offline.py

pytest -q \
  tests/entrypoints/pooling/embed/test_correctness_mteb.py::test_mteb_embed

pytest -q \
  tests/entrypoints/pooling/score/test_correctness_mteb.py::test_mteb_score \
  tests/entrypoints/pooling/score/test_correctness_mteb.py::test_mteb_rerank
```

### Perf

```bash
c 1 -- vllm serve   --model Qwen/Qwen3-Embedding-0.6B   --runner pooling   --port 9256   --enforce-eager   --enable-chunked-prefill   --max-model-len 4096   --max-num-batched-tokens 512   --disable-log-stats
vllm bench serve   --model Qwen/Qwen3-Embedding-0.6B   --backend openai-embeddings   --endpoint /v1/embeddings   --host 127.0.0.1 --port 9256 --dataset-name random   --num-prompts 2000   --request-rate inf   --seed 0   --random-input-len 2048   --random-range-ratio 0.5   --percentile-metrics e2el   --metric-percentiles 50,95,99

# now
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Benchmark duration (s):                  72.00     
Total input tokens:                      4146962   
Request throughput (req/s):              27.78     
Total token throughput (tok/s):          57599.44  
----------------End-to-end Latency----------------
Mean E2EL (ms):                          36590.14  
Median E2EL (ms):                        36550.30  
P50 E2EL (ms):                           36550.30  
P95 E2EL (ms):                           68079.96  
P99 E2EL (ms):                           70930.74  
=================================================
# main
============ Serving Benchmark Result ============
Successful requests:                     2000      
Failed requests:                         0         
Benchmark duration (s):                  73.31     
Total input tokens:                      4146962   
Request throughput (req/s):              27.28     
Total token throughput (tok/s):          56566.09  
----------------End-to-end Latency----------------
Mean E2EL (ms):                          37604.01  
Median E2EL (ms):                        37672.05  
P50 E2EL (ms):                           37672.05  
P95 E2EL (ms):                           69570.05  
P99 E2EL (ms):                           72376.59  
==================================================
```

## Comments

(no discussion comments)