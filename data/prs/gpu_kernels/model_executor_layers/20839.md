# PR #20839: [Model] Replace Mamba2 RMSNorm Gated with Fused Triton Kernel

- field: gpu_kernels  | area: model_executor/layers
- author: cyang49
- state: MERGED  created: 2025-07-11T19:38:28Z  merged: 2025-07-25T13:49:37Z
- url: https://github.com/vllm-project/vllm/pull/20839
- labels: ready

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.


## Purpose

This PR improves RMSNormGated performance. The original implementation has unfused operations which results in slow execution time. In the experiments with 16k and 32k prompt length, we found that RMSNormGated layer took as much time as mamba2 SSD computations, much longer than expected.

The fix replaces the current implementation with a fused kernel from the [mamba_ssm](https://github.com/state-spaces/mamba/blob/a6a1dae6efbf804c9944a0c2282b437deb4886d8/mamba_ssm/ops/triton/layernorm_gated.py#L45) repo. It applies to TP=1 and ngroups=1 cases only for now. It is likely that similar fixes are needed for broader use cases.

@tdoublep @tlrmchlsmth 

## Test Plan

Since the RMSNormGated is replaced with a new implementation, the output quality will be tested e2e using lm_eval with gsm8k. The performance gain will be show through benchmark_latency results.


## Test Result
Experiments done on H100-80GB
Before results are tested with f29fd8a

### benchmark_latency.py

#### Before
16k
```
python benchmarks/benchmark_latency.py --model ibm-ai-platform/Bamba-9B-v2 --input-len=16384 --output-len=1 --batch-size=1 --max_num_batched_tokens=16384
Avg latency: 0.6322657451188812 seconds
10% percentile latency: 0.6287146508228034 seconds
25% percentile latency: 0.6306551870657131 seconds
50% percentile latency: 0.6316098154056817 seconds
75% percentile latency: 0.6337889281567186 seconds
90% percentile latency: 0.63668047292158 seconds
99% percentile latency: 0.6390077278809622 seconds
```

32k
```
python benchmarks/benchmark_latency.py --model ibm-ai-platform/Bamba-9B-v2 --input-len=32768 --output-len=1 --batch-size=1 --max_num_batched_tokens=32768
Avg latency: 1.2798980366593848 seconds
10% percentile latency: 1.2776596304494887 seconds
25% percentile latency: 1.2785978385945782 seconds
50% percentile latency: 1.2790819348301739 seconds
75% percentile latency: 1.2802647254429758 seconds
90% percentile latency: 1.2838081065099687 seconds
99% percentile latency: 1.286418360499665 seconds
```

#### After
16k
```
python benchmarks/benchmark_latency.py --model ibm-ai-platform/Bamba-9B-v2 --input-len=16384 --output-len=1 --batch-size=1 --max_num_batched_tokens=16384
Avg latency: 0.5812359099897245 seconds
10% percentile latency: 0.5767858538310975 seconds
25% percentile latency: 0.5795957769732922 seconds
50% percentile latency: 0.5808804880362004 seconds
75% percentile latency: 0.5830415590899065 seconds
90% percentile latency: 0.5850747511722147 seconds
99% percentile latency: 0.5883477316750213 seconds
```

32k
```
python benchmarks/benchmark_latency.py --model ibm-ai-platform/Bamba-9B-v2 --input-len=32768 --output-len=1 --batch-size=1 --max_num_batched_tokens=32768
Avg latency: 1.1791308337822557 seconds
10% percentile latency: 1.1760888285003603 seconds
25% percentile latency: 1.1774692864855751 seconds
50% percentile latency: 1.178756954614073 seconds
75% percentile latency: 1.180810405407101 seconds
90% percentile latency: 1.1822025504428892 seconds
99% percentile latency: 1.1837539302790538 seconds
```

**Approximately 8~9% latency improvements observed.**

### lm_eval Bamba-9B
```
lm_eval --model vllm  --model_args pretrained=ibm-ai-platform/Bamba-9B-v2,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.95 --batch_size auto --trust_remote_code  --cache_requests true --tasks gsm8k
```

#### Before

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4162|±  |0.0136|
|     |       |strict-match    |     5|exact_match|↑  |0.4132|±  |0.0136|
```


#### After

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4246|±  |0.0136|
|     |       |strict-match    |     5|exact_match|↑  |0.4246|±  |0.0136|
```

## (Optional) Documentation Update

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### github-actions @ 2025-07-11T19:38:39Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### cyang49 @ 2025-07-22T14:24:27Z

Waiting for [unit test PR](https://github.com/vllm-project/vllm/pull/21379) which covers RMSNormGated to be merged, instead of adding a new unit test
---