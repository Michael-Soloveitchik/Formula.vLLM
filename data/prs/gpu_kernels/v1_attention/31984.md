# PR #31984: [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel

- field: gpu_kernels  | area: v1/attention
- author: jvlunteren
- state: MERGED  created: 2026-01-08T17:18:46Z  merged: 2026-01-10T18:13:45Z
- url: https://github.com/vllm-project/vllm/pull/31984
- labels: ready, v1

## Body

## Purpose

This pull request improves the efficiency of sliding window attention in the 3D kernel by applying the same optimization used in the 2D kernel. It ensures that only tiles within the defined window are processed, resulting in better performance and reduced computational overhead.

## Performance
The following results were obtained for `openai/gpt-oss-20b` on an NVIDIA H100 GPU, by running

<pre>$ VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm bench latency \
    --model openai/gpt-oss-20b \
    --input-len 500 --output-len &ltoutput-length&gt \
    --batch-size &ltbatch-size&gt </pre>

for
1. the `FlashAttention` backend (`VLLM_ATTENTION_BACKEND=TRITON_ATTN` is not used in the above command),
2. the current upstream version of the Triton attention kernel, and
3. the updated Triton attention kernel (this PR).

Results for batch size 1, for which the 3D kernel is executed,  are shown in the graph below. In these experiments, the output length (in tokens) was varied across the following values: 10, 100, 200, 400, 800, 1600, 3200, 6400, and 12800. The number of warmup iterations and measurement iterations remained at the default values of 10 and 30, respectively.

<img width="2304" height="1821" alt="Picture1" src="https://github.com/user-attachments/assets/e36c569d-5331-417e-a3c0-b47417295f34" />


For longer sequence lengths, the performance improvement introduced by this PR becomes more pronounced, reaching approximately 3–4% for output lengths of 6,400 and 12,800 tokens. Additionally, the FlashAttention backend appears to perform worse for this model compared to the Triton Attention backend.

## Test Plan

Existing tests can be used. For example the unit test ./tests/kernels/attention/test_triton_unified_attention.py and lm_eval

## Test Result

unit test results for updated Triton unified attention kernel (this PR):
<pre>python3 -m pytest tests/kernels/attention/test_triton_unified_attention.py

================================================ 512 passed in 61.11s ================================================
</pre>

<br>
<br>

`lm_eval` results for updated Triton attention kernel (this PR):
<pre>VLLM_ATTENTION_BACKEND=TRITON_ATTN lm_eval --model vllm --model_args pretrained=openai/gpt-oss-20b --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.352|±  |0.0214|
|     |       |strict-match    |     5|exact_match|↑  |0.262|±  |0.0197|</pre>

`lm_eval` results for FlashAttention:
<pre>lm_eval --model vllm --model_args pretrained=openai/gpt-oss-20b --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.328|±  |0.0210|
|     |       |strict-match    |     5|exact_match|↑  |0.218|±  |0.0185|</pre>


@tdoublep @bringlein 

---

> [!NOTE]
> The provided diff contains no modifications; no files were added, removed, or updated.
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 07286ec5a6edf3a7a3623ddfc8db6a24b1d3c70a. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit 9e5532517e0107433fd818fc2ee003e53931618c. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Improves efficiency of Triton unified attention by pruning tiles outside the sliding window.
> 
> - Adds sliding-window tile pruning in `kernel_unified_attention_3d`, computing `tile_start/tile_end` and intersecting with per-segment bounds to limit the for-loop range; disabled when `USE_MM_PREFIX` is set
> - Aligns `kernel_unified_attention_2d` to compute `tile_start/tile_end` and iterate only over in-window tiles
> - No API or interface changes; only internal kernel iteration/masking logic is updated
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 9e5532517e0107433fd818fc2ee003e53931618c. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

### mergify @ 2026-01-09T21:24:49Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @jvlunteren.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### jvlunteren @ 2026-01-10T12:04:21Z

restore updates in `triton_unified_attention.py` in new folder structure
---