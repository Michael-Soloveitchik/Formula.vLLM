# PR #27402: [Kernel] LoRA triton kernels support PDL

- field: gpu_kernels  | area: lora/ops
- author: jeejeelee
- state: MERGED  created: 2025-10-23T08:07:34Z  merged: 2025-11-07T08:05:48Z
- url: https://github.com/vllm-project/vllm/pull/27402
- labels: ready

## Body

## Purpose
FIX part  2 of https://github.com/vllm-project/vllm/issues/27371


## server script
vllm serve Qwen/Qwen3-30B-A3B -tp 1 -dp 1 --max-num-seqs 128 --enable-lora --max-lora-rank=64 --no-enable-prefix-caching --lora-modules test_lora_model=/path/to/loramodel

## benchmark script
vllm bench serve --tokenizer Qwen/Qwen3-30B-A3B --dataset-name random --ignore-eos --metric-percentiles 90 --model test_lora_model --num-prompts 100 --random-input-len 2048 --random-output-len 1024 --request-rate inf

The following are H20 GPU test results, from which we can observe a slight performance gain.
-  main
```
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Benchmark duration (s):                  74.05     
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              1.35      
Output token throughput (tok/s):         1382.92   
Peak output token throughput (tok/s):    1900.00   
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          4148.77   
---------------Time to First Token----------------
Mean TTFT (ms):                          8499.99   
Median TTFT (ms):                        8488.28   
P90 TTFT (ms):                           14912.10  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          63.67     
Median TPOT (ms):                        63.72     
P90 TPOT (ms):                           69.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           63.67     
Median ITL (ms):                         56.89     
P90 ITL (ms):                            59.36     
==================================================
```

- this PR

```
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Benchmark duration (s):                  69.74     
Total input tokens:                      204800    
Total generated tokens:                  102400    
Request throughput (req/s):              1.43      
Output token throughput (tok/s):         1468.21   
Peak output token throughput (tok/s):    2000.00   
Peak concurrent requests:                100.00    
Total Token throughput (tok/s):          4404.62   
---------------Time to First Token----------------
Mean TTFT (ms):                          8480.73   
Median TTFT (ms):                        8468.05   
P90 TTFT (ms):                           14890.11  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          59.51     
Median TPOT (ms):                        59.56     
P90 TPOT (ms):                           65.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           59.51     
Median ITL (ms):                         52.82     
P90 ITL (ms):                            54.47     
==================================================
```


@robertgshaw2-redhat  @varun-sundar-rabindranath  @dcmaddix 


## Comments

### dcmaddix @ 2025-10-23T16:20:48Z

Thanks @jeejeelee. This looks good to me. Can we add similar logic to the shrink and expand of `fused_moe_lora` as a follow-up?
---
### li2haipeng @ 2025-10-23T22:20:13Z

Thanks @jeejeelee. Do you see any performance gain?
---
### li2haipeng @ 2025-10-24T00:16:55Z

Just had a quick benchmark. This is a good feature to have, see following results, we generally see 5% to 20% kernel runtime speedup for smaller Ms. But it requires Triton 3.5.0. I ran into error in Triton 3.4.0. 

Tested on Qwen3-32B model shapes when TP=4:
<img width="855" height="608" alt="Screenshot 2025-10-23 at 17 14 14" src="https://github.com/user-attachments/assets/6de68fab-862b-4a0e-ba52-798024f5e55b" />
 @jeejeelee @dcmaddix @robertgshaw2-redhat 

---
### jeejeelee @ 2025-10-24T14:24:45Z

> Thanks @jeejeelee. This looks good to me. Can we add similar logic to the shrink and expand of `fused_moe_lora` as a follow-up?

Okay, I will add this ASAP
---
### dcmaddix @ 2025-10-24T14:38:34Z

> > Thanks @jeejeelee. This looks good to me. Can we add similar logic to the shrink and expand of `fused_moe_lora` as a follow-up?
> 
> Okay, I will add this ASAP

Thank you so much!
---
### jeejeelee @ 2025-10-28T07:23:57Z

> > > Thanks @jeejeelee. This looks good to me. Can we add similar logic to the shrink and expand of `fused_moe_lora` as a follow-up?
> > 
> > 
> > Okay, I will add this ASAP
> 
> Thank you so much!

Done in https://github.com/vllm-project/vllm/pull/27402/commits/4c9712e0352255960d2c2726985ab0a3f27eea3d
---
### dcmaddix @ 2025-11-05T02:29:08Z

> LGTM ! Thanks @jeejeelee

yes thanks a lot @jeejeelee!
---