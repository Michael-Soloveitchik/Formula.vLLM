# PR #34732: [Attention] Use FA4 for MLA prefill

- field: quantization  | area: model_executor/models
- author: MatthewBonanni
- state: CLOSED  created: 2026-02-17T18:12:08Z  merged: 2026-03-12T16:10:17Z
- url: https://github.com/vllm-project/vllm/pull/34732
- labels: documentation, performance, ready, ci/build, v1, nvidia

## Body

## Purpose

Depends on https://github.com/vllm-project/flash-attention/pull/123 and https://github.com/vllm-project/flash-attention/pull/126

Enable using FA4 as an MLA prefill (`forward_mha`) backend. This PR makes FA4 the default MLA prefill backend on Blackwell due to its improved performance over TRT-LLM, especially for extend operations relevant to the chunked prefill performed in vLLM

## Test Plan
```
python benchmarks/attention_benchmarks/benchmark.py --config benchmarks/attention_benchmarks/configs/mla_prefill.yaml
```

## Test Result
<img width="841" height="971" alt="Screenshot 2026-03-06 at 3 43 09 PM" src="https://github.com/user-attachments/assets/54427a9f-6228-4068-9e31-ed81e80ed668" />



## Accuracy

DeepSeek-V2-Lite-Chat

FA4 (PR)
```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6732|±  |0.0129|
|     |       |strict-match    |     5|exact_match|↑  |0.6657|±  |0.0130|
```

TRTLLM (Reference)

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6694|±  | 0.013|
|     |       |strict-match    |     5|exact_match|↑  |0.6649|±  | 0.013|
```

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

### mergify[bot] @ 2026-02-17T18:12:47Z

Documentation preview: https://vllm--34732.org.readthedocs.build/en/34732/

### mergify[bot] @ 2026-02-17T18:12:52Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @MatthewBonanni.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
