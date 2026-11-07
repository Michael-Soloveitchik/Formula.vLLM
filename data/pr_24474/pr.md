# PR #24474: [Perf] Convert np array to torch tensor to index into block table for attn chunking

- field: attention  | area: v1/attention
- author: shinoss
- state: MERGED  created: 2025-09-09T02:37:49Z  merged: 2025-09-10T03:01:06Z
- url: https://github.com/vllm-project/vllm/pull/24474
- labels: ready, v1

## Body

## Purpose

upstream pytorch change https://github.com/pytorch/pytorch/pull/160256 causes perf regression when expanding torch.tensor with numpy arrays. This affects perf of `make_local_attention_virtual_batches ` which is used to create virtual batches for chunked local attn (llama4).

On local benchmarks I see the below line take 10+ms, from <1ms before:

```
block_table_local = block_table[batch_indices, block_indices]\
        .view(virtual_batches, -1)
```

the workaround is to convert the numpy arrays to pytorch tensors before indexing. I also tried [converting all the numpy operations to torch](https://github.com/vllm-project/vllm/compare/main...sarckk:vllm:local-attn-torchify) but this led to perf regression as it introduces an additional sync point.

## Test Plan

### Correctness

ruler niah_multikey_2

```
VLLM_USE_V1=1 lm_eval --model vllm --tasks niah_multikey_2 --model_args pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct,tensor_parallel_size=4,max_model_len=256000  --metadata='{"max_seq_lengths":[4096,8192,16384,32768]}' --batch_size auto
```

```
pytest tests/v1/attention/test_chunked_local_attention.py
```

### Perf
Serve
```
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 -tp 8 --max-model-len 4096 --max-num-seqs 32 
```

Benchmark:
```
python benchmarks/benchmark_serving.py --backend vllm --ignore-eos --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --dataset-name random --max-concurrency 4 --request-rate inf --num-prompts 256  --random-input-len 2000 --random-output-len 1000
```



## Test Result

### correctness

Unit tests pass

Baseline:
```
|     Tasks     |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|---------------|------:|------|-----:|-----:|---|----:|---|------|
|niah_multikey_2|      1|none  |     0| 16384|↑  |0.980|±  |   N/A|
|               |       |none  |     0| 32768|↑  |0.946|±  |   N/A|
|               |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|               |       |none  |     0|  8192|↑  |0.996|±  |   N/A|
```

This PR
```
|     Tasks     |Version|Filter|n-shot|Metric|   |Value|   |Stderr|
|---------------|------:|------|-----:|-----:|---|----:|---|------|
|niah_multikey_2|      1|none  |     0| 16384|↑  |0.980|±  |   N/A|
|               |       |none  |     0| 32768|↑  |0.946|±  |   N/A|
|               |       |none  |     0|  4096|↑  |1.000|±  |   N/A|
|               |       |none  |     0|  8192|↑  |0.996|±  |   N/A|
```


### perf

on torch==2.8.0:

Before:
```
============ Serving Benchmark Result ============
Successful requests:                     256       
Maximum request concurrency:             4         
Benchmark duration (s):                  518.89    
Total input tokens:                      511222    
Total generated tokens:                  256000    
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         493.36    
Total Token throughput (tok/s):          1478.58   
---------------Time to First Token----------------
Mean TTFT (ms):                          31.97     
Median TTFT (ms):                        32.65     
P99 TTFT (ms):                           39.05     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.08      
Median TPOT (ms):                        8.08      
P99 TPOT (ms):                           8.21      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.08      
Median ITL (ms):                         8.06      
P99 ITL (ms):                            8.97      
==================================================
```

With PR:
```
============ Serving Benchmark Result ============
Successful requests:                     256       
Maximum request concurrency:             4         
Benchmark duration (s):                  519.45    
Total input tokens:                      511222    
Total generated tokens:                  256000    
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         492.83    
Total Token throughput (tok/s):          1476.99   
---------------Time to First Token----------------
Mean TTFT (ms):                          31.95     
Median TTFT (ms):                        32.82     
P99 TTFT (ms):                           40.57     
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          8.09      
Median TPOT (ms):                        8.10      
P99 TPOT (ms):                           8.22      
---------------Inter-token Latency----------------
Mean ITL (ms):                           8.09      
Median ITL (ms):                         8.06      
P99 ITL (ms):                            9.08      
==================================================
```

on latest torch, TPOT goes from 10.27ms -> 8.66ms

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### facebook-github-bot @ 2025-09-10T01:18:08Z

@sarckk has imported this pull request. If you are a Meta employee, you can view this in [D82077646](https://www.internalfb.com/diff/D82077646?dst_version_fbid=757712787101776).
---