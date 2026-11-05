# PR #25196: [Spec Decode] Enable FlashInfer Spec Decoding

- field: attention  | area: v1/attention
- author: benchislett
- state: MERGED  created: 2025-09-18T18:43:06Z  merged: 2025-09-24T02:29:59Z
- url: https://github.com/vllm-project/vllm/pull/25196
- labels: speculative-decoding, ready, v1

## Body

## Purpose

This PR enables FlashInfer for speculative decoding. When possible, the `trtllm-gen` decode-optimized kernel is used for speculative decoding. The fallback case is the prefill kernel, which can handle arbitrary query lengths but is not as performant. 

This PR depends on #25183 for the refactor of the batch reordering threshold variable.

Here's an example launch command for EAGLE3 on 1xB200:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 4}' --max-model-len 2048 --no-enable-prefix-caching
```

Benchmarking with 200 requests from ShareGPT gives the following TPS numbers:
- **Padding enabled + FlashInfer: 530 TPS (1.14x)**
- Padding disabled + FlashInfer: 465 TPS (1.0x)
- Padding enabled + FlashAttention: 492 TPS (1.06x)
- Padding disabled + FlashAttention: 466 TPS (1.0x)

Examination of `nsys` traces shows that the main model's attention kernels are about 2.5x faster when using the decode-optimized trtllm-gen kernels.

## Correctness Testing

Tested on GSM8k (limit 500) with/out spec and with/out FlashInfer. All successful.

### FlashInfer + Padded-Batch (Default)

```
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 2048 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 4}' 
```

```
limit: 500.0, num_fewshot: 5, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.796|±  |0.0180|
|     |       |strict-match    |     5|exact_match|↑  |0.774|±  |0.0187|
```

### FlashInfer + Non-uniform Batch (opt-in / fallback option)

```
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 2048 --no-enable-prefix-caching --port 8049 --speculative-config '{"method": "eagle3", "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 4, "disable_padded_drafter_batch": true}' 
```

```
limit: 500.0, num_fewshot: 5, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.792|±  |0.0182|
|     |       |strict-match    |     5|exact_match|↑  |0.774|±  |0.0187|
```

### Baseline FlashInfer

```
CUDA_VISIBLE_DEVICES=3 vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 2048 --no-enable-prefix-caching --port 8049
```

```
limit: 500.0, num_fewshot: 5, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.796|±  |0.0180|
|     |       |strict-match    |     5|exact_match|↑  |0.778|±  |0.0186|
```

### Baseline FlashAttn

```
CUDA_VISIBLE_DEVICES=3 VLLM_ATTENTION_BACKEND=FLASH_ATTN vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 2048 --no-enable-prefix-caching --port 8049
```

```
limit: 500.0, num_fewshot: 5, batch_size: 128
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.794|±  |0.0181|
|     |       |strict-match    |     5|exact_match|↑  |0.772|±  |0.0188|
```

### ~Known issues~ Resolved Issues
~When CUDA graphs are enabled and the padded-drafter-batch is disabled, there is a crash at high concurrency. This PR fixes a couple issues in FlashInfer where the planning and building of metadata assume that `num_decode_tokens` and `num_decodes` are the same. There is likely another such issue in the planning logic for cuda graph padding. The issue can be patched by using enforce-eager or recording a cuda graph for each input batch size.~

~Update after some investigation. There are illegal memory accesses in the TRTLLM-gen kernels that produce difficult-to-reproduce crashes. It seems unrelated to the logic in this PR which I have marked as ready-for-review. It is possible that some state corruption or race condition is independently causing issues with these kernels on blackwell. I will continue investigating~

Update: Fixed. Issue was `max_q_len` > `max(query_lens)` causing illegal access for non-uniform batches such as `[2, 1]` where the prefill max query size was smaller than the total max query size. Fixed by manually calculating `max_q_len_prefill` after splitting the batch

## Comments

### benchislett @ 2025-09-22T17:33:53Z

@pavanimajety added correctness testing to the description. All successful.
---
### benchislett @ 2025-09-22T17:35:04Z

Also fixed the kernel crashes I was seeing by adding `max_q_len_prefill` to `FlashInferMetadata`. Open to discussing better ways to handle this, maybe we can completely use `max_q_len`? It seems like the only place it is used but it seems ambiguous
---
### mergify @ 2025-09-23T16:06:40Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @benchislett.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---