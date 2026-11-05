# PR #34299: [torch.compile] Enable AR+rms fusion by default available for `-O2`

- field: gpu_kernels  | area: vllm/config
- author: ProExpertProg
- state: MERGED  created: 2026-02-11T01:59:12Z  merged: 2026-02-11T08:30:00Z
- url: https://github.com/vllm-project/vllm/pull/34299
- labels: performance, ready, torch.compile

## Body

## Purpose
Turns out AR+RMS fusion and compiling for a second range only marginally increases warm/cold start compile time, and the benefits are 5-22% (see #24252). This PR enables this fusion by default.

## Test Plan
Startup, bench, eval, CI

## Test Result

### Startup
```
$ vllm bench startup --model=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 -tp=4
COLD STARTUP:
Avg total startup time: 94.18 seconds

WARM STARTUP:
Avg total startup time: 68.90 seconds

$ vllm bench startup --model=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 -tp=4 -cc.pass_nconfig.fuse_allreduce_rms
COLD STARTUP:
Avg total startup time: 108.63 seconds

WARM STARTUP:
Avg total startup time: 76.07 seconds
```

### Bench

#### llama3-70B-fp8 tp=2 B200

5-22% improvement

<img width="1862" height="1470" alt="ttft15" src="https://github.com/user-attachments/assets/c59afa45-c45c-4951-b306-d9d5cb853c47" />

<img width="2485" height="2069" alt="tpot15_1" src="https://github.com/user-attachments/assets/0692263d-38d0-42bf-b994-0fcbfd841ad0" />

#### qwen3-235B TP=4 B200

TODO

### Eval

```
$ vllm serve RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8 -tp=4

local-completions (pretrained=RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=50,max_retries=3), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.89|±  |0.0314|
|     |       |strict-match    |     5|exact_match|↑  | 0.86|±  |0.0349|

$ vllm serve nvidia/Llama-3.3-70B-Instruct-NVFP4 -tp=4
local-completions (pretrained=nvidia/Llama-3.3-70B-Instruct-NVFP4,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=50,max_retries=3), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.91|±  |0.0288|
|     |       |strict-match    |     5|exact_match|↑  | 0.86|±  |0.0349|

$ vllm serve nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4 -tp=4
local-completions (pretrained=nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=50,max_retries=3), gen_kwargs: (None), limit: 100.0, num_fewshot: 5, batch_size: auto
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.90|±  |0.0302|
|     |       |strict-match    |     5|exact_match|↑  | 0.88|±  |0.0327|
```

### CI

Looks good.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### ProExpertProg @ 2026-02-11T16:27:13Z

@robertgshaw2-redhat `nvidia/DeepSeek-V3.1-NVFP4`:

<img width="2485" height="2069" alt="tpot_20" src="https://github.com/user-attachments/assets/f75e63e8-a6cf-4d32-a66a-b81ae1827564" />
<img width="2480" height="2069" alt="ttft_20_1" src="https://github.com/user-attachments/assets/5efe7232-377b-4a68-ba69-a96761e93cf0" />

---