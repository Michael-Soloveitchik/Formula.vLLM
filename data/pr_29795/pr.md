# PR #29795: [Perf] Improve fp8 quant in mla; replace ReduceSum with ReduceScatterSum

- field: quantization  | area: distributed/device_communicators
- author: IwakuraRein
- state: MERGED  created: 2025-12-01T17:08:41Z  merged: 2025-12-08T23:02:34Z
- url: https://github.com/vllm-project/vllm/pull/29795
- labels: ready, v1, nvidia

## Body

## Purpose

- Choose `ReduceScatterSum` over `ReduceSum` when sizes are the same
  
  <img width="1022" height="210" alt="image" src="https://github.com/user-attachments/assets/4f25ca64-0321-4e38-8dd7-a1b6736652bd" />

  <img width="1066" height="169" alt="image" src="https://github.com/user-attachments/assets/d6983667-bf05-4418-8711-e079d8590603" />

- Avoid calling fp8 quant kernel twice when using fp8 attention by creating a stage buffer for `decode_ql_nope` and `decode_q_pe`. This also eliminates the `torch.cat` in the [flashinfer mla backend](https://github.com/vllm-project/vllm/blob/75957c4e2102a78883d0745962c7986e95316d3b/vllm/v1/attention/backends/mla/flashinfer_mla.py#L135-L137).

  <img width="1464" height="367" alt="image" src="https://github.com/user-attachments/assets/6d993b12-7e2b-4b31-ab77-5ce5bbf15b12" />

## Test Plan

**Launch command**

```
  --dtype auto --kv-cache-dtype fp8 \
  --tensor-parallel-size 8 \
  --swap-space 16 --max-num-seqs 1024 --trust-remote-code --max-model-len 10240 --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 16384 --async-scheduling \
  --max-cudagraph-capture-size 1024 --compilation_config.cudagraph_mode FULL_DECODE_ONLY
```

**Evaluation command**

```
lm-eval --model local-completions --tasks gsm8k --model_args model=nvidia/DeepSeek-R1-0528-FP4-v2,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False --num_fewshot 20
```

**Models**

- Test with [DeepSeek R1 0528 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) and 8 H200.
- Test with [DeepSeek R1 0528 FP4 V2 checkpoint](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2) and 4 GB200.

## Test Result

- Hopper, `VLLM_ATTENTION_BACKEND=FLASHMLA`

  ```
  |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
  |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
  |gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9530|±  |0.0058|
  |     |       |strict-match    |    20|exact_match|↑  |0.9522|±  |0.0059|
  ```

- Blackwell, `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA`

  ```
  |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
  |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
  |gsm8k|      3|flexible-extract|    20|exact_match|_  |0.9538|_  |0.0058|
  |     |       |strict-match    |    20|exact_match|_  |0.9522|_  |0.0059|
  ```

Perf gain: 5% projected improvement in decode when using flashinfer mla backend  when serving DeepSeek R1 FP4 on 4 GB200 with DEP4

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

### chatgpt-codex-connector @ 2025-12-01T22:48:20Z

Codex usage limits have been reached for code reviews. Please check with the admins of this repo to increase the limits by adding credits.
---
### heheda12345 @ 2025-12-02T21:26:38Z

CC @MatthewBonanni  @LucasWilkinson 
---
### pavanimajety @ 2025-12-08T21:44:34Z

@IwakuraRein  Could you please add in the PR description why ReduceScatterSum is better than ReduceSum when sizes are same? or post some comparison results?
---