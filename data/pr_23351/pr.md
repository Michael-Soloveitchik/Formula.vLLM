# PR #23351: [Feature] Enable DeepGEMM Linear on B200; 1.5% E2E throughput improvement

- field: quantization  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2025-08-21T15:55:57Z  merged: 2025-08-22T04:01:08Z
- url: https://github.com/vllm-project/vllm/pull/23351
- labels: ready

## Body

## Purpose

Enable DeepGEMM Linear on B200

This should also fix some cutlass linear acc error since the weight is quantized to e8m0

## Test

```bash
VLLM_USE_DEEP_GEMM=1 lm_eval   --model vllm   --model_args "pretrained=Qwen/Qwen3-30B-A3B-FP8,max_model_len=32768,enforce_eager=True"   --trust_remote_code   --tasks gsm8k   --num_fewshot 5   --batch_size auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8529|±  |0.0098|
|     |       |strict-match    |     5|exact_match|↑  |0.8855|±  |0.0088|

# main
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8590|±  |0.0096|
|     |       |strict-match    |     5|exact_match|↑  |0.8969|±  |0.0084|
```

```bash
VLLM_USE_DEEP_GEMM=1 vllm bench throughput --model Qwen/Qwen3-30B-A3B-FP8 --load-format dummy --input-len 1000 --output-len 100 --trust_remote_code --enable-expert-parallel
Throughput: 40.59 requests/s, 44557.64 total tokens/s, 4059.10 output tokens/s
# main
Throughput: 39.97 requests/s, 43880.62 total tokens/s, 3997.42 output tokens/s
```

## Comments

### github-actions @ 2025-08-21T16:01:07Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### yewentao256 @ 2025-08-21T16:04:16Z

@mgoin  CC
---
### yewentao256 @ 2025-08-21T19:23:04Z

> LGTM. Should we also take the opportunity to refactor the logic in fp8.py? It seems there is no logic about deepgemm in Fp8LinearMethod and Fp8MoEMethod has this possibly different local logic
> 
> ```python
>         # Check for DeepGemm support.
>         self.allow_deep_gemm = False
>         if envs.VLLM_USE_DEEP_GEMM:
>             if not has_deep_gemm():
>                 logger.warning_once("Failed to import DeepGemm kernels.")
>             elif not self.block_quant:
>                 logger.warning_once("Model is not block quantized. Not using "
>                                     "DeepGemm kernels")
>             elif (is_deep_gemm_supported()):
>                 logger.info_once("Using DeepGemm kernels for Fp8MoEMethod.")
>                 self.allow_deep_gemm = True
>             else:
>                 logger.warning_once(
>                     "DeepGemm not supported on the current platform.")
> ```

Sounds good, I record this done and will have a separate PR for this.
---