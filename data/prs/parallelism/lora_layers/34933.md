# PR #34933: [FIX] fused moe with lora shared expert dual stream (1.07x otps)

- field: parallelism  | area: lora/layers
- author: jhaotingc
- state: MERGED  created: 2026-02-20T01:53:25Z  merged: 2026-02-25T04:40:46Z
- url: https://github.com/vllm-project/vllm/pull/34933
- labels: performance, ready

## Body


## Purpose
MoE model with shared expert has dual stream overlap in TP when there's no LoRA. But the overlapping disappears when adding LoRA module.

Tested with [nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) and [dasereb/Nemotron_3_Nano_BF16_r16_zeros](https://huggingface.co/dasereb/Nemotron_3_Nano_BF16_r16_zeros)

Before:
<img width="1597" height="605" alt="image" src="https://github.com/user-attachments/assets/beac041a-b41a-4f62-b5e7-bdeb92394c05" />

After:
<img width="1196" height="615" alt="image" src="https://github.com/user-attachments/assets/8e65f70b-28ed-42fd-9cfc-326297ec71b0" />

## Benchmark:
<img width="862" height="493" alt="image" src="https://github.com/user-attachments/assets/473e6806-9b34-40e5-bbb8-ae8c3ea7650c" />

```
Before (conc 1 - 64)
| OTPS/gpu | OTPS/user |  OTPS/GPU Speedup |
|---------:|----------:|--------:|
| 128.89   | 128.89    |   –     |
| 298.51   | 149.26    |   –     |
| 470.03   | 117.51    |   –     |
| 690.66   | 86.33     |   –     |
| 1213.20  | 75.83     |   –     |
| 1841.87  | 57.56     |   –     |
| 2807.50  | 43.87     |   –     |

After
| OTPS/gpu | OTPS/user |  OTPS/GPU Speedup |
| 138.67   | 138.67    |  1.08   |
| 318.86   | 159.43    |  1.07   |
| 499.50   | 124.88    |  1.06   |
| 736.38   | 92.05     |  1.07   |
| 1268.83  | 79.30     |  1.05   |
| 1937.97  | 60.56     |  1.05   |
| 2873.77  | 44.90     |  1.02   |

```

```
vllm \
  bench \
  serve \
  --dataset-name random \
  --backend openai-chat \
  --model /scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --served-model-name /scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --host my_server \
  --port 8001 \
  --endpoint /v1/chat/completions \
  --seed 1238 \
  --num-prompts 20 \
  --max-concurrency 4 \
  --ignore-eos \
  --random-range-ratio 0.0 \
  --random-input-len 1600 \
  --random-output-len 600 \
  --random-prefix-len 0 \
  --lora-modules lora1 \
  --save-result \
  --result-dir /scratch/model_sweeps/nemotron3_nano/260219/sweep/VLLM_vllm_bench_H200_TP1_1lora/logs_1238_1600_600_4_2026-02-19_17-58-21

```

## Test Plan
Even though lm-eval doesn't include LoRA, if I enable LoRA when serving the engine should go through LoRA path (verified through nsys).
So I'm acc testing the lora server with no lora request.

```
vllm serve \
/scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
--max-num-seqs 128 \
--max-num-batched-tokens 8192 \
--max-model-len 262144 \
--host 0.0.0.0 \
--port 8001 \
--tensor-parallel-size 4 \
--gpu-memory-utilization 0.95 \
--trust-remote-code \
--async-scheduling \
--no-enable-prefix-caching \
--enable-lora \
--lora-modules lora1=/scratch_other/tmp/hf_models/Nemotron_3_Nano_BF16_r16_zeros \
--max-loras 1 \
--max-lora-rank 16 \
--max-cpu-loras 1 \
--lora-dtype auto \
--compilation-config '{"compile_sizes":[1,2,4,8,16,32,64,128],"cudagraph_capture_sizes":[1,2,4,8,16,32,64,128]}' \
  2>&1 | tee /scratch/model_sweeps/nemotron3_nano/260219/vllm_serve_log/vllm_tp4_ep1_dp1_kv0.95_bs128_tok8192_seq262144_draftnone_asyncTrue_reuseFalse_kvNone_2026-02-19_17-48-48/out.log &
```

```
lm_eval --model local-completions \
  --model_args base_url=http://my_server:8001/v1/completions,model=/scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16,num_concurrent=16,tokenized_requests=True,trust_remote_code=True,max_length=262144 \
  --tasks gsm8k \
  --batch_size 16 \
  --limit 1319 \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn
```

## Test Result

Before this PR
```
local-completions ({'base_url': 'http://my_server:8001/v1/completions', 'model': '/scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', 'num_concurrent': 16, 'tokenized_requests': True, 'max_length': 262144}), gen_kwargs: ({}), limit: 1319.0, num_fewshot: 5, batch_size: 16
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4291|±  |0.0136|
|     |       |strict-match    |     5|exact_match|↑  |0.6528|±  |0.0131|
```

After this PR: (no change)
```
local-completions ({'base_url': 'http://my_server:8001/v1/completions', 'model': '/scratch_other/tmp/hf_models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', 'num_concurrent': 16, 'tokenized_requests': True, 'max_length': 262144}), gen_kwargs: ({}), limit: 1319.0, num_fewshot: 5, batch_size: 16
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.4238|±  |0.0136|
|     |       |strict-match    |     5|exact_match|↑  |0.6505|±  |0.0131|
```


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

### github-actions @ 2026-02-20T01:53:34Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### dosubot @ 2026-02-20T22:47:12Z

<!-- Answer -->
**Related Documentation**

Checked 0 published document(s) in 1 knowledge base(s). No updates required.


<!-- Dosu Comment Footer -->
<sup>How did I do? Any feedback?</sup>&nbsp;&nbsp;[![Join Discord](https://img.shields.io/badge/join-5865F2?logo=discord&logoColor=white&label=)](https://go.dosu.dev/discord-bot)
---
### dcmaddix @ 2026-02-24T00:53:59Z

LGTM thanks @jhaotingc! @jeejeelee PTAL thanks!
---
### jhaotingc @ 2026-02-24T19:54:18Z

Not sure why lora/test_qwenvl.py::test_qwen2vl_multiple_lora_types didn't pass in the first try CI, it seemed to pass after merging to upstream main.
The change shouldn't affect qwen2vl since it doesn't have shared expert.
---
### jhaotingc @ 2026-02-24T21:16:51Z

Hi @jeejeelee seems that CI test failed for irrelevant tests, should we merge this? thank you!
---
### dcmaddix @ 2026-02-24T22:33:22Z

> Hi @jeejeelee seems that CI test failed for irrelevant tests, should we merge this? thank you!

Thanks @jhaotingc ! Yes looks unrelated. @jeejeelee would be great to get this one in thanks!
---