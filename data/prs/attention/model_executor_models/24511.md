# PR #24511: [Multi Modal][Performance] Fused Q,K's apply_rope into one

- field: attention  | area: model_executor/models
- author: wwl2755
- state: MERGED  created: 2025-09-09T14:10:40Z  merged: 2025-09-14T08:10:21Z
- url: https://github.com/vllm-project/vllm/pull/24511
- labels: ready, qwen

## Body

1. Reduce Q,K's apply_rope into one in Qwen2.5-VL -> reduce time cost from 731 &micro;s to 378 &micro;s, also reduce GPU time from 485 &micro;s to 255 &micro;s
2. Merge multiple rearrange together in the code path of flash_attn -> further reduce 12 &micro;s when flash_attn is used

Overall ~11% improvement in TTFT. I will work on other models if the change makes sense to reviewers. 

Related: https://github.com/vllm-project/vllm/issues/23880, https://github.com/vllm-project/vllm/issues/23884

cc: @ywang96 @DarkLight1337 

<img width="1027" height="756" alt="image" src="https://github.com/user-attachments/assets/0b4fd09e-59c3-4f58-90b2-31b18f55f9a9" />

<img width="1010" height="711" alt="image" src="https://github.com/user-attachments/assets/dfda7c49-6d61-4b51-a68c-d9ba9f863c1b" />

<img width="1106" height="795" alt="image" src="https://github.com/user-attachments/assets/cb054904-973b-4ba2-b73f-8fe74e8eeeae" />

<img width="915" height="789" alt="image" src="https://github.com/user-attachments/assets/383e2ff8-232a-4173-a462-44904910731d" />


### e2e performance
```
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --trust-remote-code

vllm bench serve \
  --backend openai-chat \
  --endpoint-type openai-chat \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --hf-output-len 1 \
  --num-prompts 100 \
  --seed 40

# main
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  12.82     
Total input tokens:                      25524     
Total generated tokens:                  100       
Request throughput (req/s):              7.80      
Output token throughput (tok/s):         7.80      
Total Token throughput (tok/s):          1999.21   
---------------Time to First Token----------------
Mean TTFT (ms):                          6493.82   
Median TTFT (ms):                        6356.22   
P99 TTFT (ms):                           12745.14  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.03      
Median ITL (ms):                         0.03      
P99 ITL (ms):                            0.05      
==================================================

# PR
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  12.48     
Total input tokens:                      25524     
Total generated tokens:                  100       
Request throughput (req/s):              8.01      
Output token throughput (tok/s):         8.01      
Total Token throughput (tok/s):          2053.16   
---------------Time to First Token----------------
Mean TTFT (ms):                          5857.08   
Median TTFT (ms):                        4647.33   
P99 TTFT (ms):                           12405.16  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00      
Median TPOT (ms):                        0.00      
P99 TPOT (ms):                           0.00      
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.03      
Median ITL (ms):                         0.03      
P99 ITL (ms):                            0.04      
==================================================
```

### Correctness
```
lm_eval --model vllm-vlm \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks mmmu_val \
  --num_fewshot 5 \
  --limit 250 \
  --trust_remote_code \
  --batch_size 'auto'  \
  --apply_chat_template

# main
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.4611|±  |0.0161|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5917|±  |0.0448|
| - Business                     |      0|none  |      |acc   |↑  |0.4067|±  |0.0406|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.4867|±  |0.0410|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7000|±  |0.0408|
| - Science                      |      0|none  |      |acc   |↑  |0.3667|±  |0.0399|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3381|±  |0.0327|

# PR
|             Groups             |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|--------------------------------|------:|------|------|------|---|-----:|---|-----:|
|mmmu_val                        |      0|none  |      |acc   |↑  |0.4611|±  |0.0161|
| - Art and Design               |      0|none  |      |acc   |↑  |0.5917|±  |0.0448|
| - Business                     |      0|none  |      |acc   |↑  |0.4067|±  |0.0406|
| - Health and Medicine          |      0|none  |      |acc   |↑  |0.4867|±  |0.0410|
| - Humanities and Social Science|      0|none  |      |acc   |↑  |0.7000|±  |0.0408|
| - Science                      |      0|none  |      |acc   |↑  |0.3667|±  |0.0399|
| - Tech and Engineering         |      0|none  |      |acc   |↑  |0.3381|±  |0.0327|
```


## Comments

### wwl2755 @ 2025-09-14T05:25:43Z

Just a friendly ping in case it is missed. @ywang96 @DarkLight1337 @Isotr0py 
---