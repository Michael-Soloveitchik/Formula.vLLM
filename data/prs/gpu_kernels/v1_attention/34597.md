# PR #34597: [Kernel] Add FP8 KV cache support to Triton MLA decode attention

- field: gpu_kernels  | area: v1/attention
- author: grimulkan
- state: MERGED  created: 2026-02-16T03:20:21Z  merged: 2026-03-12T15:32:34Z
- url: https://github.com/vllm-project/vllm/pull/34597
- labels: documentation, ready, v1

## Body

Enable fp8/fp8_e4m3 KV cache for the Triton MLA attention backend, which is the only MLA backend available on sm120 GPUs.

- Add fp8 and fp8_e4m3 to TritonMLABackend.supported_kv_cache_dtypes
- Thread k_scale/v_scale through decode attention kernel launch path
- Add FP8 dequant-on-load in both stage1 Triton kernels (MHA and grouped/MLA)
- Set supports_quant_query_input=False for FP8 (BF16 queries + FP8 KV)
- Add FP8-specific parametrized test cases

<!-- markdownlint-disable -->
## Purpose
Enable FP8 KV cache for MLA models on sm120 (Blackwell consumer GPUs). The Triton MLA backend is the only available MLA backend on sm120, but previously blocked FP8 with `NotImplementedError`.

## Changes
- Add `"fp8"` and `"fp8_e4m3"` to `TritonMLABackend.supported_kv_cache_dtypes`
- Thread `k_scale`/`v_scale` through the decode attention kernel launch path
- Add FP8 dequant-on-load pattern in both Triton kernels (grouped and non-grouped paths)
- Set `supports_quant_query_input=False` for (BF16 queries + FP8 KV cache)
- Add 16 FP8-specific parametrized test cases

## Test Plan
- Run `pytest tests/kernels/attention/test_triton_decode_attention.py`
- Test inference using `vllm serve` on a compatible model using Triton MLA backend
- End-to-end lm_eval (GSM8K)

## Test Results
- All 112 tests pass (96 existing BF16 + 16 new FP8) in `tests/kernels/attention/test_triton_decode_attention.py`
- Tested on 16 x RTX 6000 Pro GPUs with Kimi K2.5 with DCP = 1 and fp8 KV cache
```
local-completions ({'model': 'Kimi-K2.5', 'tokenizer_backend': '/models/moonshotai-Kimi-K2.5', 'base_url': 'http://localhost:8000/v1/completions', 'num_concurrent': 500, 'tokenized_requests': False}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1

--attention-backend TRITON_MLA --kv-cache-dtype auto

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9378|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9378|±  |0.0067|

--attention-backend TRITON_MLA --kv-cache-dtype fp8

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9393|±  |0.0066|
|     |       |strict-match    |     5|exact_match|↑  |0.9393|±  |0.0066|
```
~0.15 pts within expected tolerance. Normalized generation speed (ignoring the potential 2x higher concurrency with fp8) is about the same as bf16, which is to be expected in this approach.

## Known limitations
- DCP (decode context parallelism) + FP8 is not supported (pre-existing limitation in MLA common layer, affects all backends) See https://github.com/vllm-project/vllm/pull/34795 for a fix.
- To keep the edits simple, this approach saves memory but the compute is still in bf16. It is not a true fp8 tensor core op.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### github-actions @ 2026-02-16T03:20:29Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### mergify @ 2026-02-16T03:21:03Z

Documentation preview: https://vllm--34597.org.readthedocs.build/en/34597/
---
### pavanimajety @ 2026-02-16T22:45:28Z

@grimulkan thanks for your contribution! Could you please add end to end lm_eval results for any of the standard models that can run on SM120? 
---
### grimulkan @ 2026-02-17T03:06:50Z

```
local-completions ({'model': 'Kimi-K2.5', 'tokenizer_backend': '/models/moonshotai-Kimi-K2.5', 'base_url': 'http://localhost:8000/v1/completions', 'num_concurrent': 500, 'tokenized_requests': False}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1

--attention-backend TRITON_MLA --kv-cache-dtype auto

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9378|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9378|±  |0.0067|

--attention-backend TRITON_MLA --kv-cache-dtype fp8

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9393|±  |0.0066|
|     |       |strict-match    |     5|exact_match|↑  |0.9393|±  |0.0066|
```
I think the accuracy drop (~0.15 pts) is well within expected tolerance. Normalized generation speed (ignoring the potential 2x higher concurrency with fp8) is about the same as bf16, which is to be expected in this approach.

Let me know if you need more tests like MMLU, etc. They take longer to run.

By the way, I noticed `--kv-cache-dtype bfloat16` is not actually parsed correctly by some higher-level MLA dispatch (not related to this PR) and crashes on inference, probably a separate bug. However `--kv-cache-dtype auto` or not specifying the option, indeed maps internally to b16 (I can see it in the kv cache usage), so that's how I tested the baseline.
---
### voipmonitor @ 2026-02-23T09:49:52Z

I'm confirming that this is working on 8x RTX PRO AMD Turin:

NCCL_P2P_LEVEL=SYS  VLLM_LOG_STATS_INTERVAL=1  NCCL_GRAPH_FILE=/mnt/nccl_graph_opt.xml  VLLM_TEST_FORCE_FP8_MARLIN=1 VLLM_MARLIN_USE_ATOMIC_ADD=1 VLARLIN_INPUT_DTYPE=fp8 vllm serve moonshotai/Kimi-K2.5 --served-model-name Kimi-K2.5 --trust-remote-code --host 0.0.0.0 --port 5000 --tensor-parallel-size 8 --pipeline-parallel-size 1 --enable-chunked-prefill --enable-prefix-caching --load-format fastsafetensors --tool-call-parser kimi_k2 --enable-auto-tool-choice --reasoning-parser kimi_k2 --async-scheduling --gpu-memory-utilization 0.93 --max-num-batched-tokens 4096 --mm-processor-cache-gb 0 --mm-encoder-tp-mode weights --language-model-only --attention-backend TRITON_MLA --kv-cache-dtype fp8 

GPU KV cache size: 449,600 tokens
speed: 79tok/sec 

when --decode-context-parallel-size 8 is used (more KV cache): 
GPU KV cache size: 3,621,504 tokens

speed: 66tok/sec 


---
### grimulkan @ 2026-02-23T18:09:18Z

Cross-posting these results here:

Some speed/VRAM benchmarks on sm120.

Kimi K2.5 on RTX 6000 Pro** (native int4 experts, Marlin gemm, Triton MLA)

| Cards | TP | DCP | PP | KV Cache | Total KV Cache Space | Generation Speed (@ 0 context) |
|-------|----|-----|----|----------|---------------------|-------------------------------|
| 8     | 8  | 8   | 1  | fp8      | 3M tok              | 68 tok/s                      |
| 8     | 8  | 1   | 1  | fp8      | 380K tok            | 79 tok/s                      |
| 8     | 8  | 8   | 1  | bf16     | 1.5M tok            | 67 tok/s                      |
| 8     | 8  | 1   | 1  | bf16     | 190K tok            | 78 tok/s                      |
| 16    | 16 | 16  | 1  | fp8      | 20M tok             | 43 tok/s                      |
| 16    | 16 | 1   | 1  | fp8      | 1.25M tok           | 64 tok/s                      |
| 16    | 16 | 16  | 1  | bf16     | 10M tok             | 42 tok/s                      |
| 16    | 16 | 1   | 1  | bf16     | 638K tok            | 60 tok/s                      |

All fp8 versions use this PR, and the DCP8 versions additionally use https://github.com/vllm-project/vllm/pull/34795. The unlocked KV cache savings are pretty huge.

NOTE: Likely this PR needs to be rebased & features merged if https://github.com/vllm-project/vllm/pull/33529 is merged before this one.
---
### pavanimajety @ 2026-02-23T19:46:25Z

@LucasWilkinson For review
---
### grimulkan @ 2026-03-07T00:02:24Z

Rebased, no change in performance or functionality.

I experimented with `supports_quant_query_input = True` which allows for quantized Q all-gather with fp8 KV cache and DCP, but I found it was a net negative in performance on sm120. The quant-dequant compute load of Q does not beat the savings in DCP, from DCP = 1 to 16. Since sm120 is the main target of this kernel, decided to leave `supports_quant_query_input = False` and accept bf16 Q as input, while still allowing for fp8 KV and getting the benefits from that.

The non-sm120 attention backends currently all set `supports_quant_query_input = True` when fp8 KV cache is enabled, so this is a departure from that convention, but it is a valid one and `supports_quant_query_input = False` appears to be handled just fine by other operations.

No change - just recording here for posterity.
---
### mergify @ 2026-03-09T18:41:24Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @grimulkan.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### grimulkan @ 2026-03-09T23:57:57Z

Rebased (only documentation conflict)
---
### mergify @ 2026-03-11T11:41:52Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @grimulkan.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### andyluo7 @ 2026-03-13T23:17:41Z

## ✅ ROCm MI300X Verification — All Tests Pass

Tested this PR on **AMD Instinct MI300X (gfx942, ROCm 7.0.2)** by patching into `vllm/vllm-openai-rocm:v0.17.1`.

### Unit Tests: 16/16 PASSED ✅
```
test_decode_attention_fp8[1-16384-128-128-32-32-1025-3]   PASSED
test_decode_attention_fp8[1-16384-128-128-8-32-1025-3]    PASSED
test_decode_attention_fp8[1-16384-128-576-32-32-1025-3]   PASSED
test_decode_attention_fp8[1-16384-128-576-8-32-1025-3]    PASSED
test_decode_attention_fp8[1-16384-512-128-32-32-1025-3]   PASSED
test_decode_attention_fp8[1-16384-512-128-8-32-1025-3]    PASSED
test_decode_attention_fp8[1-16384-512-576-32-32-1025-3]   PASSED
test_decode_attention_fp8[1-16384-512-576-8-32-1025-3]    PASSED
test_decode_attention_fp8[16-16384-128-128-32-32-1025-3]  PASSED
test_decode_attention_fp8[16-16384-128-128-8-32-1025-3]   PASSED
test_decode_attention_fp8[16-16384-128-576-32-32-1025-3]  PASSED
test_decode_attention_fp8[16-16384-128-576-8-32-1025-3]   PASSED
test_decode_attention_fp8[16-16384-512-128-32-32-1025-3]  PASSED
test_decode_attention_fp8[16-16384-512-128-8-32-1025-3]   PASSED
test_decode_attention_fp8[16-16384-512-576-32-32-1025-3]  PASSED
test_decode_attention_fp8[16-16384-512-576-8-32-1025-3]   PASSED

16 passed in 12.81s
```

### E2E Serving: DeepSeek-V2-Lite (MLA) with FP8 KV cache ✅
- **Config:** `--attention-backend TRITON_MLA --kv-cache-dtype fp8 --tensor-parallel-size 1`
- **Throughput:** ~72 tok/s @ concurrency=1 (256 output tokens)
- **KV cache capacity:** 9.59M tokens (vs ~4.8M BF16 — ~2x memory savings)
- **Output quality:** Correct, coherent generation confirmed

### ROCm Notes
- Kernel code is fully cross-platform — no changes needed for ROCm
- `k.dtype.is_fp8()` correctly handles both `float8_e4m3fn` (CUDA) and `float8_e4m3fnuz` (ROCm)
- FP8 dequant via simple float32 multiply works identically on both platforms
- No `tl.range()` or other ROCm-incompatible Triton ops

Great work @grimulkan! 🎉
---