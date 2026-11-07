# PR #20308: [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel

- field: gpu_kernels  | area: attention/ops
- author: jvlunteren
- state: MERGED  created: 2025-07-01T08:44:06Z  merged: 2025-07-07T19:08:12Z
- url: https://github.com/vllm-project/vllm/pull/20308
- labels: ready

## Body

This PR introduces an optimization to the unified triton attention kernel (#16828 and #19152) that enhances prefill attention performance. The key improvement involves reducing the number of tiles processed during the prefill phase by leveraging the causal mask to skip unnecessary computations. This results in more efficient execution, particularly for long prompts.

## Performance
The following results were obtained for `meta-llama/Llama-3.1-8B-Instruct` on an NVIDIA H100 GPU, by running

<pre>$ VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_V1=1 python benchmarks/benchmark_latency.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len &ltinput-length&gt --output-len 4 \
    --batch-size &ltbatch-size&gt </pre>

for
1. the current triton unified attention kernel, and
2. the updated triton unified attention kernel (this PR).

Results for a batch size 1 are shown in the following graph. The input (prompt) length (in tokens) was varied in these experiments across the following values: 500, 1000, 1500, 2000, 4000, 8000, and 16000. The number of warmup iterations and measurement iterations were left at the default values of 10 and 30 respectively.

![benchmark_latency_results](https://github.com/user-attachments/assets/9c41fdb5-785c-4a46-b840-15caf431a1fe)

As illustrated in the graph above, this PR improves the performance of the Triton Unified Attention Kernel by approximately 1.75 times for a batch size of 1 and an input length of 16000 tokens.

Additional results were collected using `benchmark_serving.py`, which only includes sequence lengths under 2000 tokens:
Current triton unified attention kernel: 
<pre>$ python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
============ Serving Benchmark Result ============
Successful requests:                     984
Benchmark duration (s):                  22.18
Total input tokens:                      210771
Total generated tokens:                  195009
Request throughput (req/s):              44.37
Output token throughput (tok/s):         8793.44
Total Token throughput (tok/s):          18297.62
---------------Time to First Token----------------
Mean TTFT (ms):                          3874.12
Median TTFT (ms):                        3715.54
P99 TTFT (ms):                           7060.57
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          88.60
Median TPOT (ms):                        51.50
P99 TPOT (ms):                           233.82
---------------Inter-token Latency----------------
Mean ITL (ms):                           40.26
Median ITL (ms):                         25.51
P99 ITL (ms):                            239.07
================================================== </pre>

Updated triton unified attention kernel (this PR):
<pre>$ python benchmarks/benchmark_serving.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json
============ Serving Benchmark Result ============
Successful requests:                     984
Benchmark duration (s):                  21.44
Total input tokens:                      210460
Total generated tokens:                  195875
Request throughput (req/s):              45.90
Output token throughput (tok/s):         9137.19
Total Token throughput (tok/s):          18954.74
---------------Time to First Token----------------
Mean TTFT (ms):                          3588.36
Median TTFT (ms):                        3478.75
P99 TTFT (ms):                           6540.15
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          83.17
Median TPOT (ms):                        47.72
P99 TPOT (ms):                           220.70
---------------Inter-token Latency----------------
Mean ITL (ms):                           38.12
Median ITL (ms):                         25.28
P99 ITL (ms):                            223.90 </pre>

Despite the relatively short prompt lengths used in this benchmark, the results still demonstrate a ~3% improvement in throughput, along with over 5% reductions in latency metrics (TTFT, TPOT, and ITL).


## Correctness

V1 `FlashAttention`:
<pre>VLLM_USE_V1=1 lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.794|±  |0.0181|
|     |       |strict-match    |     5|exact_match|↑  |0.772|±  |0.0188| </pre>

Updated triton unified attention kernel (this PR):
<pre>VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k --num_fewshot 5 --batch_size auto --limit 500
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.800|±  |0.0179|
|     |       |strict-match    |     5|exact_match|↑  |0.784|±  |0.0184|</pre>




## How is this performance improvement achieved?

The triton unified attention kernel employs a loop that iteratively processes multiple tiles, computing attention locally for each tile and accumulating the results across tiles to form the final output. During prefill processing, a causal mask is applied to each tile to ensure that attention is computed only over past and current tokens. In the current implementation, up to half of the tiles may be fully masked out during processing, resulting in redundant computation and reduced efficiency. This PR addresses the issue by skipping such tiles, ensuring that only those containing unmasked tokens are processed.

cc @tdoublep 

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### github-actions @ 2025-07-01T08:59:11Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---