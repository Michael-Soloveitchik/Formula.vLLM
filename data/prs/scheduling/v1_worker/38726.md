# PR #38726: [Bugfix][Core] Fix stuck chunked pipeline parallelism with async scheduling

- field: scheduling  | area: v1/worker
- author: starkwj
- state: MERGED  created: 2026-04-01T14:27:57Z  merged: 2026-04-17T16:02:50Z
- url: https://github.com/vllm-project/vllm/pull/38726
- labels: bug, ready, v1

## Body



## Purpose

TLDR: When enabling PP (pipeline parallelism), and other default settings (`enable-chunked-prefill`, `async-scheduling`), current implementation fails to enable chunked pipeline parallelism (CPP), which is stuck by the broadcast operation in sample_tokens.

High-degree pipeline parallelism is suitable for long-context prompts.
With the default settings, chunks should flow through the pipeline stages like micro-batches. This is effective when disabling `async-scheduling`.

When enabling `async-scheduling` (by default), the PP ranks will receive sampled tokens from the last PP rank via broadcast in `sampled_tokens`.
This is reasonable for decode but not for chunked prefill, as the sampled token is meaningless for non-last-chunk prefill and will be discarded.
**Thus preceding (non-last) PP ranks will wait until the last PP rank finishes, so the preceding PP ranks cannot move on to prefill the next chunk.**

https://github.com/vllm-project/vllm/blob/7cf56a59a267d2f32860c96a601e56bb29f6627a/vllm/v1/worker/gpu_model_runner.py#L4119-L4120

As non-last-chunked prefill doesn't need the sampled_token, this PR fixes this issue by skipping the broadcast operation for chunked prefill (only when all scheduled requests are chunked prefill). It resue the same logic in `prepare_inputs` to determine whether scheduled requests should not be sampled, so this fix should be safe (not 100% sure).

After fixed, the prefill throughput for long-context should be nearly improved to PP-ranks multiples, as expected by CPP.


## Test Plan
`vllm serve` GLM-4.7-W8A8 with -pp 4, -tp 2
**Perf**: Test with `vllm bench serve`, with random-input-len 65536, random-output-len 1
**Acc**: `lm_eval` with gsm8k and longbench_single (a sub-group in longbench, for long context test)

## Test Result
**Perf**

|  | Input token throughput (tos/s) | TTFT (ms) |
|-----|-----|-----|
| before | 1392.32 | 47068.74 |
| after| 4578.14 | 14313.72 |

**Acc**
<details>
<summary> gsm8k </summary>

```
local-completions ({'model': '/mnt/GLM-4.7-W8A8', 'base_url': 'http://localhost:8000/v1/completions', 'tokenized_requests': False, 'tokenizer_backend': None, 'max_length': 202752, 'num_concurrent': 256}), gen_kwargs: (
{}), limit: None, num_fewshot: 5, batch_size: 1
```

before:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8832|±  |0.0088|
|     |       |strict-match    |     5|exact_match|↑  |0.8825|±  |0.0089|

after:
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8848|±  |0.0088|
|     |       |strict-match    |     5|exact_match|↑  |0.8840|±  |0.0088|

</details>

<details>
<summary> longbench_single </summary>

```
local-completions ({'model': '/mnt/GLM-4.7-W8A8', 'base_url': 'http://localhost:8000/v1/completions', 'tokenized_requests': False, 'tokenizer_backend': None, 'max_length': 202752, 'num_concurrent': 32}), gen_kwargs: ({}), limit: None, num_fewshot: None, batch_size: 1
```

before:
|           Tasks            |Version|Filter|n-shot|    Metric    |   |Value |   |Stderr|
|----------------------------|------:|------|-----:|--------------|---|-----:|---|-----:|
|Single-Document QA          |      0|none  |      |score         |↑  |0.2887|±  |0.0129|
| - longbench_multifieldqa_en|      5|none  |     0|qa_f1_score   |↑  |0.3703|±  |0.0304|
| - longbench_multifieldqa_zh|      5|none  |     0|qa_f1_zh_score|↑  |0.3651|±  |0.0255|
| - longbench_narrativeqa    |      5|none  |     0|qa_f1_score   |↑  |0.2065|±  |0.0229|
| - longbench_qasper         |      5|none  |     0|qa_f1_score   |↑  |0.2128|±  |0.0256|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|Single-Document QA|      0|none  |      |score |↑  |0.2887|±  |0.0129|


after:
|           Tasks            |Version|Filter|n-shot|    Metric    |   |Value |   |Stderr|
|----------------------------|------:|------|-----:|--------------|---|-----:|---|-----:|
|Single-Document QA          |      0|none  |      |score         |↑  |0.2966|±  |0.0130|
| - longbench_multifieldqa_en|      5|none  |     0|qa_f1_score   |↑  |0.3706|±  |0.0297|
| - longbench_multifieldqa_zh|      5|none  |     0|qa_f1_zh_score|↑  |0.3846|±  |0.0274|
| - longbench_narrativeqa    |      5|none  |     0|qa_f1_score   |↑  |0.1921|±  |0.0204|
| - longbench_qasper         |      5|none  |     0|qa_f1_score   |↑  |0.2392|±  |0.0269|

|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|Single-Document QA|      0|none  |      |score |↑  |0.2966|±  | 0.013|

BTW, the results on longbench seem to have larger fluctuations. These fluctuations are also observed from repeated tests on the same model and settings.

</details>

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>

Related issuse:
- #34886

Related links:
- #33356 
- https://www.lmsys.org/blog/2026-01-15-chunked-pipeline


## Comments

### starkwj @ 2026-04-01T14:48:11Z

@yewentao256 Could you please review this PR?
---
### starkwj @ 2026-04-02T14:25:30Z

> Thanks for the work! Could you also add e2e acc using `lm_eval ..` to make sure doesn't hurt accuracy?

Sure, I have supplemented the eval results. Please check it above.
---
### starkwj @ 2026-04-03T13:49:06Z

> Looks good, could you re-bench the acc and e2e performance?

Yep, already done and updated.
---
### mergify @ 2026-04-06T15:48:12Z

Hi @starkwj, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
---