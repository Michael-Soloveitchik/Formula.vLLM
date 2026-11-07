# PR #20841: [Perf] Use Triton instead of Torch for DeepGEMM Per Token Group Quant

- field: gpu_kernels  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-07-11T20:20:12Z  merged: 2025-07-13T02:38:45Z
- url: https://github.com/vllm-project/vllm/pull/20841
- labels: performance, ready, deepseek

## Body

## Purpose

Follow up for https://github.com/vllm-project/vllm/pull/20087

Use Triton instead of Torch for Per Token Group Quant

## Test

### Accuracy

```bash
VLLM_USE_DEEP_GEMM=1 lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8476|±  |0.0099|
|     |       |strict-match    |     5|exact_match|↑  |0.8855|±  |0.0088|
# No deepgemm
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8347|±  |0.0102|
|     |       |strict-match    |     5|exact_match|↑  |0.8939|±  |0.0085|
```

### Performance

Qwen3 One card

```bash
VLLM_USE_DEEP_GEMM=1 vllm bench throughput  --model Qwen/Qwen3-30B-A3B-FP8 --load-format dummy --input-len 1000 --output-len 100 --trust_remote_code --enforce-eager --enable-expert-parallel
Throughput: 23.85 requests/s, 26179.47 total tokens/s, 2384.89 output tokens/s
# Before update
Throughput: 20.70 requests/s, 22721.25 total tokens/s, 2069.85 output tokens/s
```

R1: Don't have enough cards to test currently, but guessing it could improve like 15%+

## Comments

### github-actions @ 2025-07-11T20:20:24Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---