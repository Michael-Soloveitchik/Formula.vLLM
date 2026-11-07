# PR #23696: [Kernel][B200] `mxfp4` fused cutlass moe

- field: gpu_kernels  | area: model_executor/layers
- author: djmmoss
- state: MERGED  created: 2025-08-27T00:23:43Z  merged: 2025-09-11T21:04:57Z
- url: https://github.com/vllm-project/vllm/pull/23696
- labels: documentation, performance, ready

## Body

# Purpose

This PR adds the cutlass fused moe bf16xmxfp4 for gpt-oss on hopper and mxfp8xmxfp4 for gpt-oss on blackwell.


For Hopper there is a performance regression compared to triton (you can see in the results section). This is something we are actively working on fixing and will push to flashinfer once it is resolved. Please let me know how you want to proceed with backend selection, and I can update accordingly. 

For Blackwell the mxfp8 x mxfp4 cutlass provides around `7%` higher throughput compared to trtllm at high concurrency, however for low concurrency trtllm is still the suggest backend.

# Test Plan

## Hopper

### Unit Tests:
```bash
python3 -m pytest tests/kernels/moe/test_mxfp4_moe.py
```

### Accuracy Tests:

`evals` from [here](https://github.com/openai/gpt-oss/tree/main/gpt_oss/evals)

`low` [Ref ](https://openai.com/index/introducing-gpt-oss/)for gpt-oss-20b and gpt-oss-120b are 0.58 and 0.68 respectively.
```bash
vllm serve gpt-oss-20b --max-model-len 32768 -tp 2
python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-20b --reasoning-effort low
```

```bash
vllm serve gpt-oss-120b --max-model-len 32768 -tp 8
python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-120b --reasoning-effort low
```

### Performance:
```bash
vllm serve gpt-oss-120b --max-model-len 32768 -tp 8
vllm bench serve --backend vllm \
    --model gpt-oss-120b \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

```bash
vllm serve gpt-oss-20b --max-model-len 32768 -tp 2
vllm bench serve --backend vllm \
    --model gpt-oss-20b \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
```

## Blackwell

### Unit Tests:
```bash
python3 -m pytest tests/kernels/moe/test_mxfp4_moe.py
```

### Accuracy Tests:

`evals` from [here](https://github.com/openai/gpt-oss/tree/main/gpt_oss/evals)

`low` [Ref ](https://openai.com/index/introducing-gpt-oss/)for gpt-oss-20b and gpt-oss-120b are 0.58 and 0.68 respectively.
```bash
vllm serve gpt-oss-20b --max-model-len 32768
python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-20b --reasoning-effort low
```

```bash
vllm serve gpt-oss-120b --max-model-len 32768
python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-120b --reasoning-effort low
```

### Performance:
```bash
vllm serve gpt-oss-120b --max-model-len 32768
vllm bench serve --backend vllm
        --model gpt-oss-120b \
        --dataset-name random \
        --max-concurrency ${concurrency} \
        --num-prompts $((concurrency * 5)) \
        --ignore-eos \
        --random-input-len 1024 \
        --random-output-len 1024
```

```bash
vllm serve gpt-oss-20b --max-model-len 32768
vllm bench serve --backend vllm
        --model gpt-oss-20b \
        --dataset-name random \
        --max-concurrency ${concurrency} \
        --num-prompts $((concurrency * 5)) \
        --ignore-eos \
        --random-input-len 1024 \
        --random-output-len 1024
```

# Test Result

## Hopper

### Unit Tests:
```bash
python3 -m pytest tests/kernels/moe/test_mxfp4_moe.py
=========================================================================================== test session starts ============================================================================================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0
rootdir: /workspace/vllm
configfile: pyproject.toml
plugins: anyio-4.10.0
collected 67 items                                                                                                                                                                                         

tests/kernels/moe/test_mxfp4_moe.py sssssssssssssssssssssssssssssssssssssssssssssssssss........ssssssss                                                                                              [100%]

====================================================================================== 8 passed, 59 skipped in 6.77s =======================================================================================
```

### Accuracy Tests:
```bash
vllm serve gpt-oss-20b --max-model-len 32768 -tp 2

python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-20b --reasoning-effort low
[{'eval_name': 'gpqa', 'model_name': '__scratch__model__gpt-oss-20b-low_temp1.0_20250825_121929', 'metric': 0.5732323232323232}]
```

```bash
vllm serve gpt-oss-120b --max-model-len 32768 -tp 8

python3 -m evals --eval gpqa --model /scratch/model/gpt-oss-120b --reasoning-effort low
[{'eval_name': 'gpqa', 'model_name': '__scratch__model__gpt-oss-120b-low_temp1.0_20250825_123502', 'metric': 0.0.6609848484848485}]
```


### Peformance:
**FlashInfer:**
```bash
vllm serve gpt-oss-120b --max-model-len 32768 -tp 8
vllm bench serve --backend vllm --model gpt-oss-120b --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  26.62     
Total input tokens:                      215312    
Total generated tokens:                  197144    
Request throughput (req/s):              37.56     
Output token throughput (tok/s):         7404.72   
Total Token throughput (tok/s):          15491.84  
---------------Time to First Token----------------
Mean TTFT (ms):                          3449.17   
Median TTFT (ms):                        3229.56   
P99 TTFT (ms):                           6760.19   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          94.12     
Median TPOT (ms):                        57.07     
P99 TPOT (ms):                           264.21    
---------------Inter-token Latency----------------
Mean ITL (ms):                           45.75     
Median ITL (ms):                         34.61     
P99 ITL (ms):                            220.33    
==================================================
```

```bash
vllm serve gpt-oss-20b --max-model-len 32768 -tp 2
vllm bench serve --backend vllm --model gpt-oss-20b --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  33.94     
Total input tokens:                      215312    
Total generated tokens:                  197922    
Request throughput (req/s):              29.46     
Output token throughput (tok/s):         5831.00   
Total Token throughput (tok/s):          12174.32  
---------------Time to First Token----------------
Mean TTFT (ms):                          6173.79   
Median TTFT (ms):                        5927.93   
P99 TTFT (ms):                           11794.97  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          149.48    
Median TPOT (ms):                        80.18     
P99 TPOT (ms):                           401.31    
---------------Inter-token Latency----------------
Mean ITL (ms):                           63.93     
Median ITL (ms):                         43.34     
P99 ITL (ms):                            401.32    
==================================================
```

**Triton:**
```bash
vllm serve gpt-oss-120b --max-model-len 32768 -tp 8
vllm bench serve --backend vllm --model gpt-oss-120b --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  19.40     
Total input tokens:                      215312    
Total generated tokens:                  198101    
Request throughput (req/s):              51.54     
Output token throughput (tok/s):         10210.09  
Total Token throughput (tok/s):          21307.23  
---------------Time to First Token----------------
Mean TTFT (ms):                          4236.52   
Median TTFT (ms):                        4160.78   
P99 TTFT (ms):                           8159.43   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          81.37     
Median TPOT (ms):                        48.59     
P99 TPOT (ms):                           470.25    
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.87     
Median ITL (ms):                         18.66     
P99 ITL (ms):                            76.82     
==================================================
```


```bash
vllm serve gpt-oss-20b --max-model-len 32768 -tp 2
vllm bench serve --backend vllm --model gpt-oss-20b --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  15.66     
Total input tokens:                      215312    
Total generated tokens:                  197780    
Request throughput (req/s):              63.88     
Output token throughput (tok/s):         12633.59  
Total Token throughput (tok/s):          26387.07  
---------------Time to First Token----------------
Mean TTFT (ms):                          4105.11   
Median TTFT (ms):                        4172.38   
P99 TTFT (ms):                           5828.51   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          46.94     
Median TPOT (ms):                        29.87     
P99 TPOT (ms):                           133.46    
---------------Inter-token Latency----------------
Mean ITL (ms):                           23.34     
Median ITL (ms):                         17.03     
P99 ITL (ms):                            86.10     
==================================================
```

## Blackwell

### Unit Test
```bash
python3 -m pytest tests/kernels/moe/test_mxfp4_moe.py
=========================================================================== test session starts ============================================================================
platform linux -- Python 3.12.11, pytest-8.4.1, pluggy-1.6.0
rootdir: /workspace/vllm
configfile: pyproject.toml
plugins: anyio-4.10.0
collected 67 items                                                                                                                                                         

tests/kernels/moe/test_mxfp4_moe.py sss................................................ssssssss........                                                              [100%]

================================================================ 56 passed, 11 skipped in 300.24s (0:05:00) ================================================================
```

### Accuracy
```bash
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 vllm serve gpt-oss-120b     --max-model-len 32768     --max-num-seqs 2048     --disable-log-requests
python3 -m evals --eval gpqa --model gpt-oss-120b --reasoning-effort low
[{'eval_name': 'gpqa', 'model_name': '__workspace__scratch__repos__orangina__model__gpt-oss-120b-low_temp1.0_20250826_194732', 'metric': 0.6660353535353535}]
```

```bash
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8_CUTLASS=1 vllm serve gpt-oss-20b     --max-model-len 32768     --max-num-seqs 2048     --disable-log-requests
python3 -m evals --eval gpqa --model gpt-oss-20b --reasoning-effort low
[{'eval_name': 'gpqa', 'model_name': '__workspace__scratch__repos__orangina__model__gpt-oss-20b-low_temp1.0_20250826_200219', 'metric': 0.5612373737373737}]
```

### Performance

```bash
vllm serve gpt-oss-20b --max-model-len 32768
vllm bench serve --backend vllm
        --model gpt-oss-20b \
        --dataset-name random \
        --max-concurrency ${concurrency} \
        --num-prompts $((concurrency * 5)) \
        --ignore-eos \
        --random-input-len 1024 \
        --random-output-len 1024
```
| Concurrency |  trtllm (OTPS) |  cutlass (OTPS) | Percent Difference |
|-------------|---------|----------|--------------------|
| 1           |   316   |   197    |      -60.3%        |
| 2           |   588   |   416    |      -41.3%        |
| 4           |  1088   |   827    |      -31.6%        |
| 8           |  1951   |  1540    |      -26.7%        |
| 16          |  3529   |  2841    |      -24.2%        |
| 32          |  5960   |  5105    |      -16.8%        |
| 64          |  9710   |  8753    |      -10.9%        |
| 128         | 15143   | 14249    |       -6.3%        |
| 256         | 20280   | 20256    |       -0.1%        |
| 512         | 22018   | 23330    |        5.6%        |
| 1024        | 23195   | 24629    |        5.8%        |

```bash
vllm serve gpt-oss-120b --max-model-len 32768
vllm bench serve --backend vllm
        --model gpt-oss-120b \
        --dataset-name random \
        --max-concurrency ${concurrency} \
        --num-prompts $((concurrency * 5)) \
        --ignore-eos \
        --random-input-len 1024 \
        --random-output-len 1024
```
| Concurrency |  trtllm (OTPS) |  cutlass (OTPS) | Percent Difference |
|-------------|---------|----------|--------------------|
| 1           |   235   |   134    |      -43.0%        |
| 2           |   430   |   282    |      -34.4%        |
| 4           |  769   |   541    |      -29.6%       |
| 8           |  1332   |  976    |      -26.7%        |
| 16          |  2070   |  1594    |      -23.0%        |
| 32          |  3448   |  2680    |      -22.3%        |
| 64          |  5274   |  4328    |      -17.9%        |
| 128         | 7523   | 6689    |       -11.1%        |
| 256         | 10602   | 10312    |       -2.7%        |
| 512         | 13525   | 13273    |       -1.9%        |
| 1024        | 15592   | 16221    |        4.0%        |
| 1152        | 15369   | 16424    |        6.9%        |


## Comments

### mergify @ 2025-08-27T01:37:32Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @djmmoss.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### djmmoss @ 2025-08-27T22:30:54Z

[This](https://github.com/vllm-project/vllm/actions/runs/17280059474/job/49046346785?pr=23696) seems to be an error with pre-commit looking at an old file
---
### mergify @ 2025-08-29T01:07:35Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @djmmoss.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### nvpohanh @ 2025-08-29T01:21:13Z

@djmmoss There is one `should_use_flashinfer_mxfp4()` you probably missed:

https://github.com/djmmoss/vllm/blob/cd768a8395bff24e4de5d2bcf7bae077fb340ced/vllm/model_executor/layers/quantization/mxfp4.py#L599
---
### djmmoss @ 2025-09-08T16:20:08Z

@robertgshaw2-redhat @zyongye any other blockers for getting this merged in?
---
### robertgshaw2-redhat @ 2025-09-09T12:20:10Z

> @robertgshaw2-redhat @zyongye any other blockers for getting this merged in?

reviewing today
---
### mergify @ 2025-09-10T06:36:57Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @djmmoss.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-10T22:08:46Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @djmmoss.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mgoin @ 2025-09-11T14:24:55Z

Previous build failed only with known basic models failure and commit was just log updates, so force merging 
---
### mergify @ 2025-09-11T16:54:57Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @djmmoss.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---