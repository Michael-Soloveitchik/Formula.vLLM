# PR #24322: feat: spec decode with draft models

- field: decoding  | area: model_executor/model_loader
- author: tomasruizt
- state: MERGED  created: 2025-09-05T12:43:07Z  merged: 2026-01-19T21:05:47Z
- url: https://github.com/vllm-project/vllm/pull/24322
- labels: documentation, performance, rocm, structured-output, frontend, speculative-decoding, ready, ci/build, v1, multi-modality, tool-calling, qwen, deepseek, gpt-oss, kv-connector

## Body

## Purpose
Enabling draft models for speculative decoding (SD).
E.g. `Qwen3-1.7B` as draft model and `Qwen3-32B` as target model.
This type of SD requires no special trained heads (like EAGLE, or Medusa).

Example usage:
```shell
vllm serve \
    --model=Qwen/Qwen3-4B \
    --speculative-config '{"model": "Qwen/Qwen3-0.6B", "method": "draft_model", "num_speculative_tokens": 3, "max-model-len": 2000, "disable_padded_drafter_batch": true}' \
    --max-model-len 2000
```
Get a generation:

```shell
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Capital of France", "max_tokens": 16}'
```
## Status
* The acceptance rates in greedy decoding are great for Qwen3 models (see corresponding section).
* Using SD with Qwen3 has higher throughput (TPOT) than not using SD.

## Acceptance Length
As suggested by @ekagra-ranjan, I benchmarked acceptance length (AL) with the command below:
```shell
VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
    --model-dir Qwen/Qwen3-32B \
    --draft-model Qwen/Qwen3-1.7B \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num_prompts 100 \
    --temp 1.0 \
    --gpu-memory-utilization 0.9
```

The AL values within the Qwen3 family seem good, both with temperatures of 0.0 (greedy) and 1.0.
As a sanity check, I benchmarked LLama-3.2-1B as both target and draft, which had almost perfect AL (3.97/4), suggesting its working as intended.
I have not run the default model `meta-llama/Llama-3.1-8B-Instruct`, because I didn't find a good draft model for it, but feel free to suggest one and I can run the benchmarks.


Temperature t=0:
| Target | Draft | K | Temperature | AL |
|---|---|---|---|---|
| Qwen3-32B | Qwen3-1.7B | 3 | 0.0 | 2.79 |
| Qwen3-32B | Qwen3-1.7B | 4 | 0.0 | 3.12 |
| Llama-3.2-1B | Llama-3.2-1B | 3 | 0.0 | 3.97 |

Temperature t=1.0:
| Target | Draft | K | Temperature | AL |
|---|---|---|---|---|
| Qwen3-32B | Qwen3-1.7B | 3 | 1.0 | 2.61 |
| Qwen3-32B | Qwen3-1.7B | 4 | 1.0 | 2.85 |
| Llama-3.2-1B | Llama-3.2-1B | 3 | 1.0 | 2.82 |

Using t=1.0, the AL metric degrades. However, spec-decode with probabilities is not yet implemented, needed for lossless rejection sampling. This is being addressed atm: https://github.com/vllm-project/vllm/pull/20459. After that PR is merged, the AL for non-greedy spec-decode should improve.

All scripts and logs used for the benchmarks can be found in [this Google Drive](https://drive.google.com/drive/folders/1mFgeIVUI16VMlXJe8jm2TImY0lYKSQLC?usp=drive_link).

## Online Throughput Metrics

I measured online throughput metrics using the commands below. Hardware was an RTX PRO 6000 96GB. After making sure the draft model also uses CUDA graph, SD has higher throughput than not using SD. See tables below.

```shell
VLLM_USE_V1=1 vllm serve Qwen/Qwen3-32B \
  --max-model-len 20000 \
  --disable-uvicorn-access-log

# or 

VLLM_USE_V1=1 vllm serve Qwen/Qwen3-32B \
  --speculative_config '{"method": "draft_model", "model": "Qwen/Qwen3-1.7B", "num_speculative_tokens": 3, "max_model_len": 20000, "disable_padded_drafter_batch": true}' \
  --max-model-len 20000 \
  --disable-uvicorn-access-log

nohup vllm bench serve \
  --model Qwen/Qwen3-32B \
  --dataset-name hf \
  --dataset-path philschmid/mt-bench \
  --num-prompts 80 \
  --max-concurrency (100|1) \
  --temperature 0.0 \
  --top-p 1.0 2>&1 > results/qwen3-32b-t0-run1.out &
```

* The tables show shorter runtime and higher throughput for SD (both in batch size 1 and 100). 
* Using SD the TPOT is 50% shorter (better) in batch size 1, and 26% to 33% shorter in batch size 100. The reason is the higher throughput of the draft model.
* Using SD the TTFT and ITL are higher (worse), because tokens are produced in batches by the spec-decoding. Nevertheless, total runtimes are shorter overall when using SD.

The metrics (lower is better) are:
* TTFT: Time-to-first-token
* TPOT: Time-per-output-token
* ITL: Inter-token-latency

<details>
<summary>Batch Size = 1</summary>
For Temperature = 0.0:

| Target | Draft | Runtime |  TTFT | TPOT | ITL |
|---|---|---|---|---|---|
| Qwen3-32B | - | 943s | 69.14ms | 46.06ms | 45.88ms |
| Qwen3-32B | Qwen3-1.7B | 466s | 76.77ms | 22.62ms | 62.44ms |

Using SD runtimes and TPOT are shorter by ~50%.
</details>

<details>
<summary>Batch Size = 100</summary>
For Temperature = 0.0:

| Target | Draft | Runtime |  TTFT | TPOT | ITL |
|---|---|---|---|---|---|
| Qwen3-32B | - | 16.88s | 262.59ms | 65.09ms | 64.84ms |
| Qwen3-32B | Qwen3-1.7B | 13.04s | 284.45ms | 43.70ms | 121.48ms |

For Temperature = 1.0:

| Target | Draft | Runtime |  TTFT | TPOT | ITL |
|---|---|---|---|---|---|
| Qwen3-32B | - | 16.83s | 230.04ms | 64.95ms | 64.70ms |
| Qwen3-32B | Qwen3-1.7B | 14.84s | 272.51ms | 48.00ms | 122.27ms |
</details>

This scenario with batch size 100 is a more realistic inference case.
Using SD runtimes and TPOT are shorter.
### Profiling

This section was removed, since using CUDA graphs on the draft model significantly improved its speed.

<details>
<summary>Profiling script</summary>
I used the command below to profile the generation process and identify that the draft model was running too slow before.

```shell
export VLLM_USE_V1=1
export VLLM_TORCH_PROFILER_DIR=./profiles/
export CUDA_LAUNCH_BLOCKING=1

vllm bench throughput \
    --model=Qwen/Qwen3-32B \
    --speculative-config '{"model": "Qwen/Qwen3-1.7B", "method": "draft_model", "num_speculative_tokens": 3, "max_model_len": 2048}' \
    --dataset-name=hf \
    --dataset-path=likaixin/InstructCoder \
    --max-num-seqs=100 \
    --num-prompts=10 \
    --input-len=1000 \
    --output-len=10 \
    --max-model-len=2048 \
    --gpu-memory-utilization=0.95 \
    --profile
```
Note: The command uses the `--profile` flag, which I introduce in this PR: https://github.com/vllm-project/vllm/pull/24575
</details>

## Test Plan
The added unit test check the correctness metrics. To run it:
```shell
cd tests/v1/e2e/
pytest test_spec_decode.py -k test_draft_model_correctness
```

<details>

<summary>EAGLE testing</summary>

I tested that the EAGLE implementation stays unaffected the command below

```shell
VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py \
    --model-dir meta-llama/Llama-3.1-8B-Instruct \
    --eagle-dir yuhuili/EAGLE3-LLaMA3.1-Instruct-8B \
    --method eagle3 \
    --num_spec_tokens 7 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num_prompts 80 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9
```
The results are in line with previous measurements like https://github.com/vllm-project/vllm/pull/17504#issuecomment-2845707371

```shell
total_num_output_tokens: 16990
num_drafts: 4816
num_draft_tokens: 33712
num_accepted_tokens: 12208
mean acceptance length: 3.53
--------------------------------------------------
acceptance at token 0: 0.74
acceptance at token 1: 0.54
acceptance at token 2: 0.41
acceptance at token 3: 0.31
acceptance at token 4: 0.23
acceptance at token 5: 0.17
acceptance at token 6: 0.13
```

</details>


## Follow-up Optimizations
* Include the tokens in `next_token_ids` together with `target_token_ids` in the first forward pass of the draft model. This reduces the number of forward passes needed in each drafting phase by one, speeding up drafting.

# Qwen3 Metrics 
Note: The code for the visualizations below can be found in this gist: https://gist.github.com/tomasruizt/81aa1dee4fed88bfad2fe746c3f66739

I compare Qwen3-32B against Qwen3-32B with Qwen3-1.7B as drafter model, on a single H100 GPU (TP=1).
The benchmarks show (left to right)
* An increase in Total Token Throughput with 1 to 100 concurrent requests (so larger batch sizes).
* A decrease Benchmark Duration
* Increases in Total Token Throughput of almost 2x

<img width="31%" alt="image" src="https://github.com/user-attachments/assets/6316dc9e-488d-4170-877b-4207baee0db5" />

<img width="31%" alt="image" src="https://github.com/user-attachments/assets/b6a93069-f573-4764-9f96-084e84881a18" />

<img width="31%" alt="image" src="https://github.com/user-attachments/assets/572b97a8-d2b0-463c-9583-c4a555595c6d" />

Broken down we find:
* A steep improvement (reduction) of Time Per Output Token (TPOT)
* Higher Inter-Token-Latency (ITL), because tokens are generated and verified in batches
* Equivalent Time-To-First-Token (TTFT) except in high batch sizes.

<img width="93%" alt="image" src="https://github.com/user-attachments/assets/28a60a9c-90ad-4e29-855b-71b317546070" />

# Llama3 Multi-GPU Metrics
I benchmarked `meta-llama/Meta-Llama-3-70B`, with `meta-llama/Llama-3.2-1B` as a draft model on 4 x H100 GPUs (TP=4). The goal was to measure the effect of tensor parallelism on throughput, since the small draft model is also running on the same level of tensor parallelism as the target model.  I found an acceleration in this setup only up to bsz=32, and a slowdown for bsz=64. This setup shows that using TP > 1 degrades the acceleration of draft_model. Presumably, because of the inter-GPU communication overhead for the draft model. In a follow-up PR this should be optimized.

[serving-script.sh](https://github.com/user-attachments/files/23006446/serving-script.sh)
[benchmark-script.sh](https://github.com/user-attachments/files/23006447/benchmark-script.sh)

<img width="31%" alt="image" src="https://github.com/user-attachments/assets/f125d957-fe17-4995-aba6-b8a3990ce4c6" />
<img width="31%" alt="image" src="https://github.com/user-attachments/assets/6488414f-a119-4e2b-90a1-0dbb5ad661fa" />
<img width="31%" alt="image" src="https://github.com/user-attachments/assets/4f2c4dcd-4751-478e-9b3e-bfcce21aa6f4" />

The TPOT metric improves in general, showing similar results to Qwen3.

<img width="93%" alt="image" src="https://github.com/user-attachments/assets/19789f11-4f73-41ba-a33b-c0ecf6196c99" />

# EAGLE3 Metrics (Reference)
Below are benchmark values using method="eagle3", with target model `meta-llama/Llama-3.3-70B-Instruct` and draft model `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` on 4 x H100 GPUs (TP=4). Eagle achieves over 2x acceleration in batch_size=1. In contrast to draft_model, the eagle drafter is running with `tensor_parallelism = 1`.

[serve-script.sh](https://github.com/user-attachments/files/23017732/serve-script.sh)
[bench-script.sh](https://github.com/user-attachments/files/23017733/bench-script.sh)

<img width="31%" alt="image" src="https://github.com/user-attachments/assets/eabe9097-511e-42ab-9cfd-d14755fba842" />
<img width="31%" alt="image" src="https://github.com/user-attachments/assets/b4d996e6-b193-463f-a9c0-eaf95b2321d0" />
<img width="31%" alt="image" src="https://github.com/user-attachments/assets/e4aec257-5d6f-4abb-91a8-29b7e7e0484d" />

<img width="93%" alt="image" src="https://github.com/user-attachments/assets/9f48750f-3753-4de3-95f5-c384f05708dc" />

---

> [!NOTE]
> Introduces draft-model-based speculative decoding and wires it across configs, runtime, and tests.
> 
> - Adds `DraftModelProposer` with Triton `merge_toks_kernel`, `extend_all_queries_by_1`, and padded-batch path; integrates with `gpu_model_runner`, scheduler lookahead, and K/V cache binding (interleaving target/draft layers)
> - New config capabilities/validation: `SpeculativeConfig.uses_draft_model()`, vocab-size check, explicit `draft_tensor_parallel_size` (rejects `tensor_parallel_size`), per-draft `ParallelConfig` creation, `disable_logprobs`, and `num_lookahead_slots`; `ParallelConfig.replace()` and `VllmConfig.replace()` helpers
> - Model loading updated to accept `prefix` and separate draft `VllmConfig`; enforce TP constraints and isolate quantization between target/draft
> - Metrics helpers to compute acceptance rate/length; ready checker logs when endpoint isn’t ready
> - Example script adds `--method draft_model`, `--draft-model`, GPU mem utilization, and max seqs; passes new speculative fields
> - Extensive tests: correctness/quantization/TP for draft-model SD, engine-args validation, KV cache binding with draft layers, and kernel unit tests
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit d9641ad89c71712b35ab9fdc05777ead16818b56. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Enables speculative decoding using a separate draft model and integrates it across configuration, runtime, and tests.
> 
> - New `DraftModelProposer` with Triton `merge_toks_kernel` and query-extension path; integrates with `gpu_model_runner`, scheduler lookahead, padded-batch path, and interleaved K/V cache binding
> - Config updates: `SpeculativeConfig.uses_draft_model()`, vocab-size validation, explicit `draft_tensor_parallel_size` (rejects `tensor_parallel_size`), draft `ParallelConfig` creation, and TP constraints; add `ParallelConfig.replace()` and `VllmConfig.replace()`
> - Model loading supports draft-specific `VllmConfig` and `prefix` to namespace modules; isolates draft quantization/TP from target
> - Attention utils: `extend_all_queries_by_1`, `CommonAttentionMetadata` helpers; scheduler now sets lookahead for draft-model SD
> - Metrics helpers to compute acceptance rate/length; ready-checker logs when endpoint isn’t ready
> - Example `spec_decode.py` adds `--method draft_model`, `--draft-model`, GPU mem/max seqs flags; forwards new speculative fields
> - Tests: correctness/quantization/TP for draft-model SD, engine-args validation, KV cache binding for draft layers, and kernel unit tests
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit d708778513f8b3cceb3f93ce6e08c77661c14bce. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces draft-model-based speculative decoding and wires it across config, runtime, and tests.
> 
> - New `DraftModelProposer` with Triton `merge_toks_kernel` and query-extension (`extend_all_queries_by_1`); integrates with scheduler lookahead and `gpu_model_runner` (padded-batch path, GPU tokens) and validates KV cache grouping
> - Config: add `uses_draft_model()`, vocab-size check, explicit `draft_tensor_parallel_size` (rejects `tensor_parallel_size`), per-draft `ParallelConfig`/`VllmConfig` via `create_vllm_config_for_draft_model`, and `replace()` helpers for `ParallelConfig`/`VllmConfig`
> - Model loading supports draft-specific configs and `prefix` to namespace modules across loaders (base, GGUF, tensorizer)
> - Worker/utils: bind K/V caches interleaving target and draft layers; scheduler sets `num_lookahead_tokens` for draft-model SD
> - Bench/UX: ready-checker warns when endpoint not ready; example script adds `--method draft_model`, `--draft-model`, memory/max-seqs flags
> - Metrics: helpers to compute acceptance rate/length
> - Tests: correctness/quantization/TP for draft-model SD, engine-arg validation, KV-cache binding, and kernel unit tests
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 5ee3a595a1dade9e062f90b55e14010983553aeb. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

### github-actions @ 2025-09-05T12:43:35Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### ekagra-ranjan @ 2025-09-05T14:34:16Z

@tomasruizt - Thank you for the PR! 
1. Can you also report Acceptance Length (AL) and K used? It is more informative than AR. For e.g., with Eagle with K=3 we get AL ~2.29 on MTBench so we can expect 2.29x speedup assuming 0 draft overhead. AL gives a more holistic overview of the speedup for a given K than AR. 
2. Can you run the metrics on MTBench instead of one off sample for less noisy metric? This would also allow us to compare the AL and TPOT improvement from other SD methods: https://github.com/vllm-project/vllm/issues/17812. You can find the cmd to run the offline inference to find AL and online inference to find TPOT using MTBench in vLLM here: https://github.com/vllm-project/vllm/pull/20459. You will need to update offline inference script to use separate draft model.
---
### ekagra-ranjan @ 2025-09-05T15:10:05Z

> As a main model Qwen3-1.7B runs a decoding forward pass in 9-10ms, while when its used as a draft model, each forward pass takes 19-22ms.

What is the TP you are using for Qwen3-32B? By default, draft model TP is equal to target model TP. Since Qwen3-1.7B is a small model, running it on high TP might be incurring nccl communication cost. Try setting draft TP to 1.
---
### tomasruizt @ 2025-09-05T15:22:36Z

> What is the TP you are using for Qwen3-32B? By default, draft model TP is equal to target model TP. Since Qwen3-1.7B is a small model, running it on high TP might be incurring nccl communication cost. Try setting draft TP to 1.

I ran the benchmarks with TP=1 and num_draft_tokens=3. So we can rule out TP communication issues.

---
### mergify @ 2025-09-08T09:02:51Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-09-08T11:26:09Z

@ekagra-ranjan I updated the metrics with your suggestions. Acceptance lengths look good IMO. After applying CUDA graph on the draft model its speed improved dramatically. Now the throughput of using SD is higher than not using it (in particular TPOT).
---
### ekagra-ranjan @ 2025-09-09T19:04:22Z

@tomasruizt - The online inference cmd shared in PR has this flag `--max-concurrency 100`. This means that you are testing on Batch size of 100. Please use 1 instead of 100 to iterate as of now.  When things are finalized then you can provide numbers for BS=1/4/16. 

Also, please provide the cmd you used to profile the model for reproducibility.
---
### tomasruizt @ 2025-09-10T12:20:17Z

@ekagra-ranjan
* I've added throughput metrics with batch size = 1. 
  * With batch_size=1, using SD reduces runtime and TPOT by 50%. I.e. inference is 2x as fast 🚀 
  * With batch_size=100, using SD also reduces runtime and TPOT, by 26% to 33%. This batch size gives a better impression of SD performance under a more realistic and challenging workload.
* I've added the profiling command in the **Profiling** section
---
### mergify @ 2025-09-11T04:25:03Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### hmellor @ 2025-09-16T11:05:42Z

@benchislett / @luccafong could you take a look at this PR?

I recall there being some discussion about whether we should support draft model speculative decoding so your input woud be valuable here.
---
### LiuXiaoxuanPKU @ 2025-09-17T06:22:30Z

Hi @tomasruizt , I discussed with @WoosukKwon last week about this PR, the major concern is code complexity.
Questions before we move on:
1. Just want to double check, if the draft model has different `num_head` and `head_dim`, then we cannot support it. --> This is fine, we just need to document it somewhere.
2. How do you handle the KV cache for the draft model, say we propose 5 tokens, only 3 of them are accepted. Could you point me the code where you handle this case.

Code complexity related changes:
1. Could you separate this PRs into at least two? One is the draft model based spec decode functionality, the other is cuda graph support for the draft model.
2. Could you separate draft model proposer logic and eagle proposer logic? If they share some common logic, you can create some abstraction, but it's also ok if we don't do it for now.

---
### tomasruizt @ 2025-09-17T07:14:47Z

@LiuXiaoxuanPKU Thanks for the path forward Lily!

I can separate the code paths for EAGLE and draft-models, to make the draft-model path easier to comprehend! 👍 I'm optimistic about this.

About your two questions:
1. I run a benchmark with target=Qwen3-32B and draft=Qwen3-1.7B. Both have different `num_attention_heads` (64 and 16, respectively), according to their config.json files ([Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json)). And it seems to work fine. It's perhaps surprising that it works, but where precisely would we expect the problems to arise?
2. I don't handle the case explicitly. My understanding of what happens under the hood with the KV Cache after `set_forward_context()` is limited, but I know from debugging that the target model and draft model have separate `attn_metadata` objects and separate `kv_cache` tensors.
---
### tomasruizt @ 2025-09-17T19:38:53Z

@shanes-cerebras interesting! 
I haven't encountered that degradation yet. Perhaps you can share your benchmarking commands if you are also using `examples/offline_inference/spec_decode.py`. Otherwise, I'm also happy to look at your code once you submit the PR. And I'm happy collaborate on this feature! :) I'm not explicitly managing the draft KV-cache, so exposing any limitation with my setup would be nice. 
Would you mind sharing the model pairs you are aiming for in the hybrid-attention / full-attention combo?
---
### mergify @ 2025-09-18T05:56:28Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-18T16:15:51Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-09-18T21:41:41Z

I had made a mistake while merging main to this branch, and the bot added a bunch of extra labels. But the commits look Ok again.
---
### tomasruizt @ 2025-09-19T09:52:35Z

@LiuXiaoxuanPKU I separated the proposing logic of EAGLE and Draft Models and they share a superclass for common logic. The CUDA graph code is still in this PR (only for method=draft_model) because the code changes are small and it speeds up the draft model. You suggested splitting the CUDA graph code into a separate PR. Was this to reduce code complexity, or rather to establish before & after performance? I can still break out the CUDA graph code into a separate PR, just let me know again.
---
### tomasruizt @ 2025-09-19T10:05:08Z

@benchislett I added a test veryfing the support for the flag `disable_padded_batch` (True/False), since both configs take different codepaths during input preparation.
---
### tomasruizt @ 2025-09-19T10:17:50Z

After splitting the `DraftModelProposer` from the `EagleProposer` I reran the online throughput metrics for batch size 1, as suggested by @ekagra-ranjan, this time on an H100. Spec decoding still improves throughput by 2x for Qwen3-32B (batch_size=1). The command is the same as in the PR description. Results are [here](https://drive.google.com/drive/folders/1lei5BK63WwPuSY-LLR6gpATjuyQRpTVx?usp=drive_link), including TTFT, TPOT, and ITL.
---
### tomasruizt @ 2025-09-19T15:23:29Z

@benchislett fair.  I'll factor out the duplicated code in `.propose()` to reduce duplication as much as possible. I might submit that as a separate _refactor-only_ PR, to ensure the integrity of the EAGLE implementation.

In terms of the extra decode: I'll try to eliminate it and reach out for help if needed.
---
### mergify @ 2025-09-22T20:06:44Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-09-22T21:23:58Z

@benchislett 

As we discussed in our call, I moved the `propose()` method into the superclass so it could be reused among classes. I also introduced flags to the constructor to toggle the behavior between EAGLE and DraftModel:
1. `pass_hidden_states_to_model: bool` True for EAGLE
2. `pass_cudagraph_args_to_forward_ctx: bool` True for DraftModel
3. `one_extra_forward_pass: bool` True for DraftModel
4. `drop_first_drafted_tokens: bool` True for DraftModel
5. A variable `self.num_forward_passes` which is 1 larger for the DraftModel

Number 3 to 5 will become unnecessary if we implement the optimal prefill we discussed, which would reduce the forward passes by one, and improve runtimes. Nevertheless, it's useful to remember the major factor affecting DraftModel speed is the CUDA graph usage, which is now conveniently a single flag.

At the moment the EAGLE file diff looks horrible, I guess because of the combination of extracting a superclass plus introducing flags to the constructor. Let me know if you would prefer to review the EAGLE refactor as a separate PR to main (or as multiple small PRs) 👍 

**Edit:** Managed to get nice git diff for the EAGLE file by minimizing changes.
---
### mergify @ 2025-09-23T18:21:16Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-09-24T08:31:13Z

The EAGLE code is frequently changed on main, so it is difficult to move EAGLE code around without painful merge conflicts.
To get nice git diffs on the EAGLE file I put the body of EAGLE into the superclass. 

---
### mergify @ 2025-09-25T05:33:56Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-09-26T09:20:03Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-10-06T05:13:42Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-10-06T08:51:14Z

I refactored to remove the extra forward pass. However, the new acceptance length (AL) I'm measuring atm  is lower than before the refactor. This could be related to this recent bug in `main` with AL: https://github.com/vllm-project/vllm/issues/26191
I want to solve that first before making statements about AL with/without the extra forward pass.
---
### tomasruizt @ 2025-10-08T12:04:16Z

After refactoring I've benchmarked acceptance length (AL) and throughput metrics. The AL remains at the same value as before the refactor (2.79). This requires `disable_padded_drafter_batch = True` to avoid the AL regression documented here: https://github.com/vllm-project/vllm/issues/26191.

[acceptance-len-command.slurm.txt](https://github.com/user-attachments/files/22770529/acceptance-len-command.slurm.txt)

In terms of throughput metrics, I'm using the `mtbench` dataset with several levels of concurrency, which shows the performance of spec decode at different batch sizes. The performance metrics are good, with total token throughput of Qwen3-32B increasing between 40% and 90%.

[throughput-benchmark.slurm.txt](https://github.com/user-attachments/files/22770510/throughput-benchmark.slurm.txt)

<img width="49%" alt="image" src="https://github.com/user-attachments/assets/bcfc012d-106a-449b-8fe3-94fc60795206" />
<img width="49%" alt="image" src="https://github.com/user-attachments/assets/d2f632ea-d9d3-4a6d-a2af-6520e9eb2882" />

The total benchmark duration goes down as well:

<img width="60%" alt="image" src="https://github.com/user-attachments/assets/97b9de4e-fd15-46b8-9a31-5fda18ee566a" />

---
### mergify @ 2025-10-08T12:09:17Z

Documentation preview: https://vllm--24322.org.readthedocs.build/en/24322/
---
### tomasruizt @ 2025-10-08T12:10:08Z

The detailed performance metrics are also positive:
* TPOT is considerably reduced, which is why tokens come out faster
* ITL increases because tokens come out in batch
* TTFT seems constant, presumably because its driven by target model prefills

<img width="1470" height="581" alt="image" src="https://github.com/user-attachments/assets/63551582-17b4-4012-9513-e0d85849ba87" />

---
### tomasruizt @ 2025-10-08T12:16:33Z

One final optimization that I left out is to set `uniform_decode=True` in the `BatchDescriptor` during the draft model decodes, which I think could cut the draft model runtimes by ~25%. However, if set that, I get a significant AL reduction.
---
### tomasruizt @ 2025-10-08T12:18:56Z

Let me know what we could do to move this PR forward @benchislett @ekagra-ranjan @LiuXiaoxuanPKU @hmellor 🚀 
---
### LiuXiaoxuanPKU @ 2025-10-09T04:04:17Z

Hi @tomasruizt, I am running the branch, I am hitting the error:
(EngineCore_DP0 pid=1461095)     self._draft_token_ids = self.propose_draft_token_ids(
(EngineCore_DP0 pid=1461095)                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Traceback (most recent call last):
(EngineCore_DP0 pid=1461095)   File "/data/lily/draft-sd-vllm/vllm/v1/worker/gpu_model_runner.py", line 2819, in propose_draft_token_ids
(EngineCore_DP0 pid=1461095)     draft_token_ids = self.drafter.propose(
(EngineCore_DP0 pid=1461095)                       ^^^^^^^^^^^^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/examples/offline_inference/spec_decode.py", line 222, in <module>
(EngineCore_DP0 pid=1461095)   File "/data/lily/draft-sd-vllm/vllm/v1/spec_decode/draft_model.py", line 71, in propose
(EngineCore_DP0 pid=1461095)     draft_token_ids = super().propose(
(EngineCore_DP0 pid=1461095)                       ^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=1461095)   File "/data/lily/draft-sd-vllm/vllm/v1/spec_decode/eagle.py", line 310, in propose
(EngineCore_DP0 pid=1461095)     positions = target_positions[last_token_indices]
(EngineCore_DP0 pid=1461095)                 ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^
(EngineCore_DP0 pid=1461095) torch.AcceleratorError: CUDA error: device-side assert triggered
(EngineCore_DP0 pid=1461095) CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(EngineCore_DP0 pid=1461095) For debugging consider passing CUDA_LAUNCH_BLOCKING=1
(EngineCore_DP0 pid=1461095) Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
(EngineCore_DP0 pid=1461095) 
---
### tomasruizt @ 2025-10-09T06:05:24Z

@LiuXiaoxuanPKU thanks for trying it! what is your vllm command and workload?
---
### LiuXiaoxuanPKU @ 2025-10-09T16:58:25Z

```
VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py \
    --model-dir meta-llama/Llama-3.1-8B-Instruct \
    --draft-model eqhylxx/vicuna-160m \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num_prompts 100 \
    --temp 1.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 2048
```
---
### mergify @ 2025-10-10T08:44:56Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### tomasruizt @ 2025-10-10T12:47:54Z

Dear @LiuXiaoxuanPKU, I was able to reproduce an error with your command. Thank you.
The issue seems to be that both models have different vocabulary sizes. LLama-3.1 8B [has 128k tokens](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json#L37), while vicuna-160m [has 32k tokens](https://huggingface.co/eqhylxx/vicuna-160m/blob/main/config.json#L25). At runtime this leads to an out-of-bounds error when the token IDs of Llama-3.1 are passed to the small vicuna model.

I have added an early validation to the Speculative Config to prevent cases where the tokenizers have different vocab sizes.
---
### tomasruizt @ 2025-10-10T14:41:52Z

After the merging https://github.com/vllm-project/vllm/pull/25109 I simplified away most code here related to CUDA graphs @benchislett  👍 
---
### LiuXiaoxuanPKU @ 2025-10-12T05:41:48Z

Thanks! @tomasruizt 

I am testing more about this PR. When trying with larger batch size, it seems errors:
My command:
```
VLLM_USE_V1=1 python examples/offline_inference/spec_decode.py \
    --model-dir meta-llama/Meta-Llama-3-70B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B \
    --method draft_model \
    --num_spec_tokens 3 \
    --dataset-name sharegpt \
    --dataset-path /data/lily/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num_prompts 128 \
    --temp 0.0 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8094 \
    --output-len 8094 \
    --tp 4 
```

Error message:

```
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780] EngineCore encountered a fatal error.
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780] Traceback (most recent call last):
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 771, in run_engine_core
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     engine_core.run_busy_loop()
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 798, in run_busy_loop
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     self._process_engine_step()
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 827, in _process_engine_step
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     outputs, model_executed = self.step_fn()
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]                               ^^^^^^^^^^^^^^
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 316, in step
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     model_output = self.execute_model_with_error_logging(
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 302, in execute_model_with_error_logging
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     raise err
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/engine/core.py", line 292, in execute_model_with_error_logging
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     return model_fn(scheduler_output)
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]            ^^^^^^^^^^^^^^^^^^^^^^^^^^
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/executor/multiproc_executor.py", line 188, in execute_model
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     (output,) = self.collective_rpc(
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]                 ^^^^^^^^^^^^^^^^^^^^
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/executor/multiproc_executor.py", line 285, in collective_rpc
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     result = get_response(w, dequeue_timeout, self.shutdown_event)
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]   File "/data/lily/draft-sd-vllm/vllm/v1/executor/multiproc_executor.py", line 266, in get_response
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780]     raise RuntimeError(
^[[1;36m(EngineCore_DP0 pid=2586423)^[[0;0m ERROR 10-11 13:32:24 [core.py:780] RuntimeError: Worker failed with error 'The expanded size of the tensor (16384) must match the existing size (16451) at non-singleton dimension 0.  Target sizes: [16384].  Tensor sizes: [16451]', please check the stack trace above for the root cause
Traceback (most recent call last):
  File "/data/lily/draft-sd-vllm/examples/offline_inference/spec_decode.py", line 225, in <module>
    acceptance_length = main(args)
                        ^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/examples/offline_inference/spec_decode.py", line 163, in main
    outputs = llm.generate(
              ^^^^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/vllm/entrypoints/llm.py", line 450, in generate
    outputs = self._run_engine(use_tqdm=use_tqdm)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/vllm/entrypoints/llm.py", line 1719, in _run_engine
    step_outputs = self.llm_engine.step()
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/vllm/v1/engine/llm_engine.py", line 292, in step
    outputs = self.engine_core.get_output()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/data/lily/draft-sd-vllm/vllm/v1/engine/core_client.py", line 710, in get_output
    raise self._format_exception(outputs) from None
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
/data/lily/miniconda3/envs/draft-sd-vllm/lib/python3.12/multiprocessing/resource_tracker.py:279: UserWarning: resource_tracker: There appear to be 1 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

Processed prompts:   0%|          | 0/128 [00:03<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]

```


---
### LiuXiaoxuanPKU @ 2025-10-12T05:50:45Z

> The detailed performance metrics are also positive:
> 
> * TPOT is considerably reduced, which is why tokens come out faster
> * ITL increases because tokens come out in batch
> * TTFT seems constant, presumably because its driven by target model prefills
> 
> <img alt="image" width="1470" height="581" src="https://private-user-images.githubusercontent.com/22105494/498850204-63551582-17b4-4012-9513-e0d85849ba87.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjAyNDg0OTksIm5iZiI6MTc2MDI0ODE5OSwicGF0aCI6Ii8yMjEwNTQ5NC80OTg4NTAyMDQtNjM1NTE1ODItMTdiNC00MDEyLTk1MTMtZTBkODU4NDliYTg3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDEyVDA1NDk1OVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM3NDRlOWIwN2NiN2NmZGU1YTljYmRkMWVlOGQ0MWQwMzBlNGZkYzRmOWMwNzBiMWE2ZThkNGZmMmRkOGNlZmQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.JgbxX9qNGRrZyurlmfzWsNcaJEPtjLgFJxxEDL6R0AI">

Also, for the numbers here, if TTFT is similar, TPOT is greatly reduced, why is ITL getting much higher?
---
### tomasruizt @ 2025-10-12T08:55:46Z

@LiuXiaoxuanPKU thanks for testing with tensor parallelism. I'll make sure to solve that OOB failure. 👍

The ITL is measured on the client side. With spec decoding the tokens are generated faster, but they are streamed to the client in small batches rather than individually. E.g. assuming k=3 and all tokens are accepted by the target model, then the engine produces 3+1=4 tokens (K=3 + bonus token) per step. Each step takes longer (drafting + verification). The client observes slower steps that each yield 4 tokens each.

This is what I meant with "tokens come out in batch", and can be observed also for concurrency=1.
---
### ggg-s @ 2025-10-13T01:56:24Z

Hi, may I ask how to test the combination of a **quantized target model** and a **non-quantized draft model**?
After pulling your PR branch and compiling it locally, it seems that the **Marlin kernel acceleration** doesn’t work.

---
### ggg-s @ 2025-10-13T08:33:12Z

Why does speculative decoding with draft models currently require --disable-padded-drafter-batch?
Is there a technical reason that padded drafter batches can’t be supported yet ?
Would it be possible to relax this restriction in the future?

def _raise_if_padded_drafter_batch(self):
    if not self.vllm_config.speculative_config.disable_padded_drafter_batch:
        raise NotImplementedError(
            "Speculative Decoding with draft models does not support "
            "padded drafter batch yet. Please pass --disable-padded-drafter-batch "
            "in the speculative config."
        )

---
### tomasruizt @ 2025-10-13T08:44:37Z

@ggg-s That restriction will be lifted in the future. Atm it is required to avoid a performance regression (acceptance length) documented here: https://github.com/vllm-project/vllm/issues/26191.
---
### tomasruizt @ 2025-10-13T08:49:30Z

@ggg-s in terms of quantized models, I just haven't tested that combination yet. Perhaps you can share your launch command, what you expected to happen, and what went wrong?
---
### ggg-s @ 2025-10-13T08:52:49Z

Thanks for your response! I haven’t tested it yet, but I’ll share the results once I do.

---
### tomasruizt @ 2025-10-13T09:49:25Z

@LiuXiaoxuanPKU I haven't been able to reproduce the error you showed yet. I sent you a message over Slack to compare environments 👍 
---
### QingNagi @ 2025-10-13T12:44:59Z

I try it :
My command:
vllm serve Qwen3/Qwen3-32B-FP8 \
--served-model-name "qwen3" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 2 \
--trust-remote-code \
--max-model-len 10000 \
--speculative-config '{"method": "draft_model", "model": "Qwen3/Qwen3-0.6B", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1, "max-model-len": 10000, "disable_padded_drafter_batch": true}'  \
--override-generation-config '{"temperature" : 0}' &
Error message:
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] WorkerProc failed to start.
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] Traceback (most recent call last):
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/executor/multiproc_executor.py", line 571, in worker_main
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     worker = WorkerProc(*args, **kwargs)
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/executor/multiproc_executor.py", line 437, in __init__
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     self.worker.load_model()
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py", line 213, in load_model
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     self.model_runner.load_model(eep_scale_up=eep_scale_up)
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py", line 2656, in load_model
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     self.drafter.load_model(self.model)
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/spec_decode/draft_model.py", line 147, in load_model
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     self.model = get_model(
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/__init__.py", line 122, in get_model
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     return loader.load_model(
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/base_loader.py", line 50, in load_model
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     self.load_weights(model, model_config)
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/default_loader.py", line 276, in load_weights
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597]     raise ValueError("Following weights were not initialized from "
(Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] ValueError: Following weights were not initialized from checkpoint: {'model.layers.18.self_attn.attn.prob_scale', 'model.layers.7.self_attn.attn.k_scale', 'model.layers.26.self_attn.attn.v_scale', 'model.layers.0.mlp.gate_up_proj.weight_scale_inv', 'model.layers.27.self_attn.attn.v_scale', 'model.layers.0.self_attn.attn.q_scale', 'model.layers.27.self_attn.qkv_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.q_scale', 'model.layers.27.mlp.down_proj.weight_scale_inv', 'model.layers.2.mlp.gate_up_proj.weight_scale_inv', 'model.layers.12.mlp.down_proj.weight_scale_inv', 'model.layers.23.self_attn.o_proj.weight_scale_inv', 'model.layers.25.self_attn.qkv_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.v_scale', 'model.layers.14.self_attn.attn.k_scale', 'model.layers.18.self_attn.qkv_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.q_scale', 'model.layers.12.self_attn.attn.k_scale', 'model.layers.18.self_attn.attn.k_scale', 'model.layers.9.self_attn.o_proj.weight_scale_inv', 'model.layers.8.self_attn.qkv_proj.weight_scale_inv', 'model.layers.22.self_attn.attn.k_scale', 'model.layers.25.self_attn.attn.k_scale', 'model.layers.20.self_attn.o_proj.weight_scale_inv', 'model.layers.19.mlp.down_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.prob_scale', 'model.layers.24.mlp.gate_up_proj.weight_scale_inv', 'model.layers.26.mlp.gate_up_proj.weight_scale_inv', 'model.layers.17.self_attn.o_proj.weight_scale_inv', 'model.layers.0.mlp.down_proj.weight_scale_inv', 'model.layers.4.self_attn.o_proj.weight_scale_inv', 'model.layers.4.self_attn.attn.v_scale', 'model.layers.6.self_attn.o_proj.weight_scale_inv', 'model.layers.7.self_attn.attn.v_scale', 'model.layers.16.mlp.down_proj.weight_scale_inv', 'model.layers.7.mlp.gate_up_proj.weight_scale_inv', 'model.layers.1.self_attn.o_proj.weight_scale_inv', 'model.layers.0.self_attn.attn.prob_scale', 'model.layers.27.self_attn.attn.prob_scale', 'model.layers.21.self_attn.attn.q_scale', 'model.layers.19.self_attn.attn.k_scale', 'model.layers.27.self_attn.o_proj.weight_scale_inv', 'model.layers.4.self_attn.attn.prob_scale', 'model.layers.2.self_attn.qkv_proj.weight_scale_inv', 'model.layers.10.mlp.down_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.k_scale', 'model.layers.24.self_attn.attn.v_scale', 'model.layers.15.self_attn.attn.q_scale', 'model.layers.21.self_attn.o_proj.weight_scale_inv', 'model.layers.12.self_attn.attn.v_scale', 'model.layers.21.mlp.gate_up_proj.weight_scale_inv', 'model.layers.15.self_attn.attn.prob_scale', 'model.layers.12.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.q_scale', 'model.layers.24.self_attn.o_proj.weight_scale_inv', 'model.layers.12.self_attn.attn.q_scale', 'model.layers.10.self_attn.attn.q_scale', 'model.layers.3.self_attn.attn.q_scale', 'model.layers.7.mlp.down_proj.weight_scale_inv', 'model.layers.5.self_attn.o_proj.weight_scale_inv', 'model.layers.27.self_attn.attn.k_scale', 'model.layers.21.self_attn.attn.v_scale', 'model.layers.20.self_attn.attn.v_scale', 'model.layers.17.self_attn.attn.k_scale', 'model.layers.14.self_attn.qkv_proj.weight_scale_inv', 'model.layers.3.self_attn.o_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.prob_scale', 'model.layers.13.mlp.down_proj.weight_scale_inv', 'model.layers.11.mlp.gate_up_proj.weight_scale_inv', 'model.layers.13.self_attn.attn.v_scale', 'model.layers.1.self_attn.attn.v_scale', 'model.layers.13.self_attn.qkv_proj.weight_scale_inv', 'model.layers.0.self_attn.attn.v_scale', 'model.layers.11.self_attn.attn.prob_scale', 'model.layers.4.mlp.gate_up_proj.weight_scale_inv', 'model.layers.7.self_attn.attn.q_scale', 'model.layers.17.self_attn.attn.q_scale', 'model.layers.20.self_attn.qkv_proj.weight_scale_inv', 'model.layers.4.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.v_scale', 'model.layers.11.self_attn.o_proj.weight_scale_inv', 'model.layers.18.self_attn.attn.v_scale', 'model.layers.20.mlp.gate_up_proj.weight_scale_inv', 'model.layers.10.self_attn.qkv_proj.weight_scale_inv', 'model.layers.19.self_attn.attn.v_scale', 'model.layers.24.self_attn.attn.k_scale', 'model.layers.26.self_attn.attn.prob_scale', 'model.layers.0.self_attn.qkv_proj.weight_scale_inv', 'model.layers.9.self_attn.qkv_proj.weight_scale_inv', 'model.layers.23.self_attn.attn.v_scale', 'model.layers.18.mlp.down_proj.weight_scale_inv', 'model.layers.14.self_attn.attn.prob_scale', 'model.layers.5.mlp.down_proj.weight_scale_inv', 'model.layers.26.self_attn.attn.k_scale', 'model.layers.23.mlp.down_proj.weight_scale_inv', 'model.layers.15.self_attn.attn.k_scale', 'model.layers.3.mlp.gate_up_proj.weight_scale_inv', 'model.layers.9.mlp.down_proj.weight_scale_inv', 'model.layers.22.self_attn.qkv_proj.weight_scale_inv', 'model.layers.12.mlp.gate_up_proj.weight_scale_inv', 'model.layers.22.self_attn.o_proj.weight_scale_inv', 'model.layers.24.mlp.down_proj.weight_scale_inv', 'model.layers.24.self_attn.qkv_proj.weight_scale_inv', 'model.layers.20.mlp.down_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.prob_scale', 'model.layers.1.self_attn.attn.prob_scale', 'model.layers.22.self_attn.attn.prob_scale', 'model.layers.19.self_attn.qkv_proj.weight_scale_inv', 'model.layers.17.self_attn.qkv_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.v_scale', 'model.layers.3.self_attn.attn.v_scale', 'model.layers.5.self_attn.attn.q_scale', 'model.layers.14.mlp.down_proj.weight_scale_inv', 'model.layers.10.self_attn.attn.prob_scale', 'model.layers.27.mlp.gate_up_proj.weight_scale_inv', 'model.layers.1.self_attn.attn.q_scale', 'model.layers.6.mlp.gate_up_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.k_scale', 'model.layers.22.mlp.down_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.v_scale', 'model.layers.8.self_attn.o_proj.weight_scale_inv', 'model.layers.13.self_attn.o_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.prob_scale', 'model.layers.14.self_attn.o_proj.weight_scale_inv', 'model.layers.21.mlp.down_proj.weight_scale_inv', 'model.layers.17.self_attn.attn.v_scale', 'model.layers.8.self_attn.attn.k_scale', 'model.layers.0.self_attn.attn.k_scale', 'model.layers.4.self_attn.qkv_proj.weight_scale_inv', 'model.layers.2.self_attn.o_proj.weight_scale_inv', 'model.layers.22.mlp.gate_up_proj.weight_scale_inv', 'model.layers.21.self_attn.attn.prob_scale', 'model.layers.9.self_attn.attn.v_scale', 'model.layers.4.self_attn.attn.q_scale', 'model.layers.19.self_attn.attn.prob_scale', 'model.layers.8.mlp.gate_up_proj.weight_scale_inv', 'model.layers.5.mlp.gate_up_proj.weight_scale_inv', 'model.layers.12.self_attn.o_proj.weight_scale_inv', 'model.layers.7.self_attn.qkv_proj.weight_scale_inv', 'model.layers.8.self_attn.attn.v_scale', 'model.layers.13.self_attn.attn.prob_scale', 'model.layers.13.self_attn.attn.q_scale', 'model.layers.17.mlp.down_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.q_scale', 'model.layers.19.mlp.gate_up_proj.weight_scale_inv', 'model.layers.24.self_attn.attn.prob_scale', 'model.layers.12.self_attn.attn.prob_scale', 'model.layers.19.self_attn.attn.q_scale', 'model.layers.26.self_attn.o_proj.weight_scale_inv', 'model.layers.2.mlp.down_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.q_scale', 'model.layers.25.self_attn.o_proj.weight_scale_inv', 'model.layers.9.mlp.gate_up_proj.weight_scale_inv', 'model.layers.3.self_attn.attn.k_scale', 'model.layers.4.self_attn.attn.k_scale', 'model.layers.23.mlp.gate_up_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.k_scale', 'model.layers.0.self_attn.o_proj.weight_scale_inv', 'model.layers.13.mlp.gate_up_proj.weight_scale_inv', 'model.layers.18.self_attn.o_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.k_scale', 'model.layers.5.self_attn.qkv_proj.weight_scale_inv', 'model.layers.1.mlp.down_proj.weight_scale_inv', 'model.layers.25.mlp.gate_up_proj.weight_scale_inv', 'model.layers.26.self_attn.attn.q_scale', 'model.layers.26.self_attn.qkv_proj.weight_scale_inv', 'model.layers.15.mlp.down_proj.weight_scale_inv', 'model.layers.22.self_attn.attn.v_scale', 'model.layers.27.self_attn.attn.q_scale', 'model.layers.22.self_attn.attn.q_scale', 'model.layers.23.self_attn.attn.k_scale', 'model.layers.10.mlp.gate_up_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.q_scale', 'model.layers.16.self_attn.attn.k_scale', 'model.layers.10.self_attn.attn.v_scale', 'model.layers.16.mlp.gate_up_proj.weight_scale_inv', 'model.layers.14.self_attn.attn.q_scale', 'model.layers.23.self_attn.attn.prob_scale', 'model.layers.1.mlp.gate_up_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.prob_scale', 'model.layers.7.self_attn.attn.prob_scale', 'model.layers.23.self_attn.attn.q_scale', 'model.layers.15.mlp.gate_up_proj.weight_scale_inv', 'model.layers.10.self_attn.o_proj.weight_scale_inv', 'model.layers.19.self_attn.o_proj.weight_scale_inv', 'model.layers.18.mlp.gate_up_proj.weight_scale_inv', 'model.layers.16.self_attn.qkv_proj.weight_scale_inv', 'model.layers.23.self_attn.qkv_proj.weight_scale_inv', 'model.layers.14.mlp.gate_up_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.v_scale', 'model.layers.11.mlp.down_proj.weight_scale_inv', 'model.layers.10.self_attn.attn.k_scale', 'model.layers.14.self_attn.attn.v_scale', 'model.layers.3.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.v_scale', 'model.layers.13.self_attn.attn.k_scale', 'model.layers.3.self_attn.attn.prob_scale', 'model.layers.17.mlp.gate_up_proj.weight_scale_inv', 'model.layers.7.self_attn.o_proj.weight_scale_inv', 'model.layers.26.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.k_scale', 'model.layers.8.self_attn.attn.q_scale', 'model.layers.8.self_attn.attn.prob_scale', 'model.layers.3.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.qkv_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.q_scale', 'model.layers.16.self_attn.o_proj.weight_scale_inv', 'model.layers.18.self_attn.attn.q_scale', 'model.layers.8.mlp.down_proj.weight_scale_inv', 'model.layers.15.self_attn.qkv_proj.weight_scale_inv', 'model.layers.15.self_attn.o_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.k_scale', 'model.layers.6.self_attn.attn.prob_scale', 'model.layers.21.self_attn.attn.k_scale', 'model.layers.25.mlp.down_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.prob_scale', 'model.layers.6.mlp.down_proj.weight_scale_inv', 'model.layers.21.self_attn.qkv_proj.weight_scale_inv', 'model.layers.24.self_attn.attn.q_scale', 'model.layers.1.self_attn.qkv_proj.weight_scale_inv', 'model.layers.1.self_attn.attn.k_scale', 'model.layers.17.self_attn.attn.prob_scale', 'model.layers.15.self_attn.attn.v_scale'}
---
### tomasruizt @ 2025-10-13T14:17:31Z

@QingNagi Thanks for testing with FP8 quantization 👍! 
I was able to reproduce your error with `target=Qwen/Qwen3-1.7B-FP8`, and `draft=Qwen/Qwen3-0.6B`. I introduced a fix, so either of the models can be quantized (I've only tested with FP8 so far). Let me know if it works for your case now.
---
### QingNagi @ 2025-10-14T01:50:48Z

LGTM! This fix solved my problem, thanks for your help.👍
---
### ggg-s @ 2025-10-14T01:56:00Z

After pulling your latest branch and rebuilding, I got this message:
<img width="1581" height="849" alt="image" src="https://github.com/user-attachments/assets/81b6f7d8-f153-40d0-9a7c-f1ed4cba30a9" />
What’s the reason for it?
---
### tomasruizt @ 2025-10-14T06:16:52Z

@ggg-s The warnings? I also see them. I assume they are unrelated to this PR.
---
### ggg-s @ 2025-10-14T06:19:31Z

Yeah, got it.
---
### ggg-s @ 2025-10-14T08:04:00Z

I used the following test, but the **tpot** didn’t decrease significantly. What could be the reason for this?
vllm serve：
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/Qwen3-32B \
  --port 18000 \
  --max-model-len 10000 \
  --max-num-seqs 32 \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "draft_model",
    "model": "/models/Qwen3-1.7B",
    "num_speculative_tokens": 3,
    "max-model-len": 10000,
    "disable_padded_drafter_batch": true
  }'
temperature：0 bsz：1 nums：5：
<img width="734" height="93" alt="image" src="https://github.com/user-attachments/assets/47d2204c-258c-4a84-b0db-19cf63820d97" />

---
### tomasruizt @ 2025-10-14T08:36:30Z

@ggg-s Only ~27% reduction in TPOT, I see. Can you also share your <del>profiling</del> benchmarking command? I'm looking into how tensor parallelism affects the draft model.
---
### ggg-s @ 2025-10-14T08:50:05Z

I didn’t run profiling yet. Could you please share which profiling command you recommend to analyze the effect of tensor parallelism on the draft model?
---
### tomasruizt @ 2025-10-14T08:56:36Z

@ggg-s Sorry, I meant your benchmarking command. I.e. the code to output the TTFT, TPOT, etc metrics.
E..g. `vllm bench serve --model Qwen/Qwen3-32B ...`
---
### ggg-s @ 2025-10-14T09:02:29Z

Oh got it — I didn’t use vllm bench serve. I’m just using my own script that measures per-chunk response latency from the streamed outputs to estimate TPOT/TTFT.
---
### tomasruizt @ 2025-10-14T11:23:52Z

I attempted to set TP > 1 for the target model, and TP = 1 for the draft model for efficiency, but I found collisions between ranks with torch compile. For now, I enforce that the TP of both models must be the same. In the future a solution like https://github.com/vllm-project/vllm/pull/5414 can be re-implemented to run draft models with TP = 1.
---
### ggg-s @ 2025-10-14T13:01:00Z

Hi, I’m currently testing speculative decoding with a VLM target model and an LLM draft model.
I noticed that when the batch size is 1, the throughput improves by around 14.7%, but when the batch size is greater than 1, there’s no speedup at all.
The dataset I’m using is pure text (no images).
I referred to your PR and expected to see acceleration in this setup, but it doesn’t seem to take effect.
Could you please share any possible reasons or insights on why this might be happening? Thanks a lot!

vllm serve：
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/Qwen2.5-VL-72B-AWQ \
  --port 18000 \
  --max-model-len 10000 \
  --max-num-seqs 32 \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "draft_model",
    "model": "/models/Qwen2.5-1.5B",
    "num_speculative_tokens": 3,
    "max-model-len": 10000,
    "disable_padded_drafter_batch": true
  }'

result：
<img width="1144" height="204" alt="image" src="https://github.com/user-attachments/assets/a6b566cf-7f73-44b1-9070-e0cd374f078d" />


---
### tomasruizt @ 2025-10-14T14:56:15Z

@ggg-s Interesting case with Qwen2.5-VL 😄 I would need to profile the workload to understand it. There is a profiling command in the PR description, in case you want to profile it yourself. The slowness could have 2 causes that I can think of:
* Since the draft model must run with TP > 1, the communication overhead is slowing it down.
* The draft model is running with PIECEWISE CUDA graphs, which is slower than FULL CUDA graph (`uniform_decode=True`).
---
### ggg-s @ 2025-10-15T01:08:25Z

thank u. 
I noticed something strange: even after setting (and removing the TP restriction), the service starts fine, but the draft model still loads on every GPU. Looks like each TP rank is creating its own drafter instead of sharing a single TP=1 instance."draft_tensor_parallel_size": 1
---
### ggg-s @ 2025-10-15T01:52:17Z

> > What is the TP you are using for Qwen3-32B? By default, draft model TP is equal to target model TP. Since Qwen3-1.7B is a small model, running it on high TP might be incurring nccl communication cost. Try setting draft TP to 1.
> 
> I ran the benchmarks with TP=1 and num_draft_tokens=3. So we can rule out TP communication issues.
11

---
### QingNagi @ 2025-10-15T02:06:21Z

What's the achievable Mean Acceptance Length for this combination?


> 嗨，我目前正在使用 VLM 目标模型和 LLM 草稿模型测试推测解码。我注意到，当批量大小为 1 时，吞吐量提高了 14.7% 左右，但当批量大小大于 1 时，根本没有加速。我使用的数据集是纯文本（无图像）。我参考了您的 PR，并希望在此设置中看到加速，但它似乎没有生效。您能否分享一下为什么会发生这种情况的可能原因或见解？多谢！
> 
> vllm serve： CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /models/Qwen2.5-VL-72B-AWQ --port 18000 --max-model-len 10000 --max-num-seqs 32 --tensor-parallel-size 4 --speculative-config '{ “method”： “draft_model”， “model”： “/models/Qwen2.5-1.5B”， “num_speculative_tokens”： 3， “max-model-len”： 10000， “disable_padded_drafter_batch”： true }'
> 
> 结果： <img alt="图像" width="1144" height="204" src="https://private-user-images.githubusercontent.com/71912051/500954515-a6b566cf-7f73-44b1-9070-e0cd374f078d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjA0OTM1OTMsIm5iZiI6MTc2MDQ5MzI5MywicGF0aCI6Ii83MTkxMjA1MS81MDA5NTQ1MTUtYTZiNTY2Y2YtN2Y3My00NGIxLTkwNzAtZTBjZDM3NGYwNzhkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNTEwMTUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMDE1VDAxNTQ1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTM5MTBjYjQwZjZlZWIyYWE2MzBhMGMzNmEzNzFiMjYwYzFmZDRlNGI0OGFmZmNhYjMwNjFlOTZkNmI4M2EwYWYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.NrzDjnYseCscgFARcu45dAYOFJkUkH0eSyj95z3QX-0">

What's the achievable Mean Acceptance Length for this combination?
---
### ggg-s @ 2025-10-15T02:35:16Z

@tomasruizt I didn’t set , but the logs already show , so the drafter seems to be using full CUDA graphs by default.uniform_decode=Truedecode, FULL
<img width="1577" height="161" alt="image" src="https://github.com/user-attachments/assets/faba85e0-ccc5-4bc8-a62e-8fed34c4c340" />

---
### ggg-s @ 2025-10-15T02:37:12Z

it's 2.5
---
### QingNagi @ 2025-10-15T02:43:56Z

> 是 2.5

This Mean Acceptance Length is okay, I think the draft_model method might not be suitable for high concurrency operations on this long input short output dataset. 
---
### ggg-s @ 2025-10-15T02:46:44Z

@QingNagi Could you please elaborate why you say so?
---
### QingNagi @ 2025-10-15T02:52:26Z

> 您能详细说明一下您为什么这么说吗？

The main purpose of the draft_model method is to improve the efficiency of the decode phase and increase resource utilization during the decode phase, which results in increased pressure during the prefill phase.
---
### ggg-s @ 2025-10-15T03:57:24Z

@QingNagi Oh, right — thanks for the reminder ！I forgot about the potential drawbacks speculative decoding can introduce under short-output conditions.

---
### ggg-s @ 2025-10-15T06:14:59Z




> I try it : My command: vllm serve Qwen3/Qwen3-32B-FP8 --served-model-name "qwen3" --gpu-memory-utilization 0.9 --tensor-parallel-size 2 --trust-remote-code --max-model-len 10000 --speculative-config '{"method": "draft_model", "model": "Qwen3/Qwen3-0.6B", "num_speculative_tokens": 5, "draft_tensor_parallel_size": 1, "max-model-len": 10000, "disable_padded_drafter_batch": true}' --override-generation-config '{"temperature" : 0}' & Error message: (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] WorkerProc failed to start. (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] Traceback (most recent call last): (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/executor/multiproc_executor.py", line 571, in worker_main (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] worker = WorkerProc(*args, **kwargs) (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/executor/multiproc_executor.py", line 437, in **init** (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] self.worker.load_model() (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py", line 213, in load_model (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] self.model_runner.load_model(eep_scale_up=eep_scale_up) (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py", line 2656, in load_model (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] self.drafter.load_model(self.model) (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/spec_decode/draft_model.py", line 147, in load_model (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] self.model = get_model( (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/**init**.py", line 122, in get_model (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] return loader.load_model( (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/base_loader.py", line 50, in load_model (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] self.load_weights(model, model_config) (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] File "/opt/vllm_env/lib/python3.10/site-packages/vllm/model_executor/model_loader/default_loader.py", line 276, in load_weights (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] raise ValueError("Following weights were not initialized from " (Worker_TP0 pid=112985) ERROR 10-13 12:01:14 [multiproc_executor.py:597] ValueError: Following weights were not initialized from checkpoint: {'model.layers.18.self_attn.attn.prob_scale', 'model.layers.7.self_attn.attn.k_scale', 'model.layers.26.self_attn.attn.v_scale', 'model.layers.0.mlp.gate_up_proj.weight_scale_inv', 'model.layers.27.self_attn.attn.v_scale', 'model.layers.0.self_attn.attn.q_scale', 'model.layers.27.self_attn.qkv_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.q_scale', 'model.layers.27.mlp.down_proj.weight_scale_inv', 'model.layers.2.mlp.gate_up_proj.weight_scale_inv', 'model.layers.12.mlp.down_proj.weight_scale_inv', 'model.layers.23.self_attn.o_proj.weight_scale_inv', 'model.layers.25.self_attn.qkv_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.v_scale', 'model.layers.14.self_attn.attn.k_scale', 'model.layers.18.self_attn.qkv_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.q_scale', 'model.layers.12.self_attn.attn.k_scale', 'model.layers.18.self_attn.attn.k_scale', 'model.layers.9.self_attn.o_proj.weight_scale_inv', 'model.layers.8.self_attn.qkv_proj.weight_scale_inv', 'model.layers.22.self_attn.attn.k_scale', 'model.layers.25.self_attn.attn.k_scale', 'model.layers.20.self_attn.o_proj.weight_scale_inv', 'model.layers.19.mlp.down_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.prob_scale', 'model.layers.24.mlp.gate_up_proj.weight_scale_inv', 'model.layers.26.mlp.gate_up_proj.weight_scale_inv', 'model.layers.17.self_attn.o_proj.weight_scale_inv', 'model.layers.0.mlp.down_proj.weight_scale_inv', 'model.layers.4.self_attn.o_proj.weight_scale_inv', 'model.layers.4.self_attn.attn.v_scale', 'model.layers.6.self_attn.o_proj.weight_scale_inv', 'model.layers.7.self_attn.attn.v_scale', 'model.layers.16.mlp.down_proj.weight_scale_inv', 'model.layers.7.mlp.gate_up_proj.weight_scale_inv', 'model.layers.1.self_attn.o_proj.weight_scale_inv', 'model.layers.0.self_attn.attn.prob_scale', 'model.layers.27.self_attn.attn.prob_scale', 'model.layers.21.self_attn.attn.q_scale', 'model.layers.19.self_attn.attn.k_scale', 'model.layers.27.self_attn.o_proj.weight_scale_inv', 'model.layers.4.self_attn.attn.prob_scale', 'model.layers.2.self_attn.qkv_proj.weight_scale_inv', 'model.layers.10.mlp.down_proj.weight_scale_inv', 'model.layers.9.self_attn.attn.k_scale', 'model.layers.24.self_attn.attn.v_scale', 'model.layers.15.self_attn.attn.q_scale', 'model.layers.21.self_attn.o_proj.weight_scale_inv', 'model.layers.12.self_attn.attn.v_scale', 'model.layers.21.mlp.gate_up_proj.weight_scale_inv', 'model.layers.15.self_attn.attn.prob_scale', 'model.layers.12.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.q_scale', 'model.layers.24.self_attn.o_proj.weight_scale_inv', 'model.layers.12.self_attn.attn.q_scale', 'model.layers.10.self_attn.attn.q_scale', 'model.layers.3.self_attn.attn.q_scale', 'model.layers.7.mlp.down_proj.weight_scale_inv', 'model.layers.5.self_attn.o_proj.weight_scale_inv', 'model.layers.27.self_attn.attn.k_scale', 'model.layers.21.self_attn.attn.v_scale', 'model.layers.20.self_attn.attn.v_scale', 'model.layers.17.self_attn.attn.k_scale', 'model.layers.14.self_attn.qkv_proj.weight_scale_inv', 'model.layers.3.self_attn.o_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.prob_scale', 'model.layers.13.mlp.down_proj.weight_scale_inv', 'model.layers.11.mlp.gate_up_proj.weight_scale_inv', 'model.layers.13.self_attn.attn.v_scale', 'model.layers.1.self_attn.attn.v_scale', 'model.layers.13.self_attn.qkv_proj.weight_scale_inv', 'model.layers.0.self_attn.attn.v_scale', 'model.layers.11.self_attn.attn.prob_scale', 'model.layers.4.mlp.gate_up_proj.weight_scale_inv', 'model.layers.7.self_attn.attn.q_scale', 'model.layers.17.self_attn.attn.q_scale', 'model.layers.20.self_attn.qkv_proj.weight_scale_inv', 'model.layers.4.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.v_scale', 'model.layers.11.self_attn.o_proj.weight_scale_inv', 'model.layers.18.self_attn.attn.v_scale', 'model.layers.20.mlp.gate_up_proj.weight_scale_inv', 'model.layers.10.self_attn.qkv_proj.weight_scale_inv', 'model.layers.19.self_attn.attn.v_scale', 'model.layers.24.self_attn.attn.k_scale', 'model.layers.26.self_attn.attn.prob_scale', 'model.layers.0.self_attn.qkv_proj.weight_scale_inv', 'model.layers.9.self_attn.qkv_proj.weight_scale_inv', 'model.layers.23.self_attn.attn.v_scale', 'model.layers.18.mlp.down_proj.weight_scale_inv', 'model.layers.14.self_attn.attn.prob_scale', 'model.layers.5.mlp.down_proj.weight_scale_inv', 'model.layers.26.self_attn.attn.k_scale', 'model.layers.23.mlp.down_proj.weight_scale_inv', 'model.layers.15.self_attn.attn.k_scale', 'model.layers.3.mlp.gate_up_proj.weight_scale_inv', 'model.layers.9.mlp.down_proj.weight_scale_inv', 'model.layers.22.self_attn.qkv_proj.weight_scale_inv', 'model.layers.12.mlp.gate_up_proj.weight_scale_inv', 'model.layers.22.self_attn.o_proj.weight_scale_inv', 'model.layers.24.mlp.down_proj.weight_scale_inv', 'model.layers.24.self_attn.qkv_proj.weight_scale_inv', 'model.layers.20.mlp.down_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.prob_scale', 'model.layers.1.self_attn.attn.prob_scale', 'model.layers.22.self_attn.attn.prob_scale', 'model.layers.19.self_attn.qkv_proj.weight_scale_inv', 'model.layers.17.self_attn.qkv_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.v_scale', 'model.layers.3.self_attn.attn.v_scale', 'model.layers.5.self_attn.attn.q_scale', 'model.layers.14.mlp.down_proj.weight_scale_inv', 'model.layers.10.self_attn.attn.prob_scale', 'model.layers.27.mlp.gate_up_proj.weight_scale_inv', 'model.layers.1.self_attn.attn.q_scale', 'model.layers.6.mlp.gate_up_proj.weight_scale_inv', 'model.layers.20.self_attn.attn.k_scale', 'model.layers.22.mlp.down_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.v_scale', 'model.layers.8.self_attn.o_proj.weight_scale_inv', 'model.layers.13.self_attn.o_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.prob_scale', 'model.layers.14.self_attn.o_proj.weight_scale_inv', 'model.layers.21.mlp.down_proj.weight_scale_inv', 'model.layers.17.self_attn.attn.v_scale', 'model.layers.8.self_attn.attn.k_scale', 'model.layers.0.self_attn.attn.k_scale', 'model.layers.4.self_attn.qkv_proj.weight_scale_inv', 'model.layers.2.self_attn.o_proj.weight_scale_inv', 'model.layers.22.mlp.gate_up_proj.weight_scale_inv', 'model.layers.21.self_attn.attn.prob_scale', 'model.layers.9.self_attn.attn.v_scale', 'model.layers.4.self_attn.attn.q_scale', 'model.layers.19.self_attn.attn.prob_scale', 'model.layers.8.mlp.gate_up_proj.weight_scale_inv', 'model.layers.5.mlp.gate_up_proj.weight_scale_inv', 'model.layers.12.self_attn.o_proj.weight_scale_inv', 'model.layers.7.self_attn.qkv_proj.weight_scale_inv', 'model.layers.8.self_attn.attn.v_scale', 'model.layers.13.self_attn.attn.prob_scale', 'model.layers.13.self_attn.attn.q_scale', 'model.layers.17.mlp.down_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.q_scale', 'model.layers.19.mlp.gate_up_proj.weight_scale_inv', 'model.layers.24.self_attn.attn.prob_scale', 'model.layers.12.self_attn.attn.prob_scale', 'model.layers.19.self_attn.attn.q_scale', 'model.layers.26.self_attn.o_proj.weight_scale_inv', 'model.layers.2.mlp.down_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.q_scale', 'model.layers.25.self_attn.o_proj.weight_scale_inv', 'model.layers.9.mlp.gate_up_proj.weight_scale_inv', 'model.layers.3.self_attn.attn.k_scale', 'model.layers.4.self_attn.attn.k_scale', 'model.layers.23.mlp.gate_up_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.k_scale', 'model.layers.0.self_attn.o_proj.weight_scale_inv', 'model.layers.13.mlp.gate_up_proj.weight_scale_inv', 'model.layers.18.self_attn.o_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.k_scale', 'model.layers.5.self_attn.qkv_proj.weight_scale_inv', 'model.layers.1.mlp.down_proj.weight_scale_inv', 'model.layers.25.mlp.gate_up_proj.weight_scale_inv', 'model.layers.26.self_attn.attn.q_scale', 'model.layers.26.self_attn.qkv_proj.weight_scale_inv', 'model.layers.15.mlp.down_proj.weight_scale_inv', 'model.layers.22.self_attn.attn.v_scale', 'model.layers.27.self_attn.attn.q_scale', 'model.layers.22.self_attn.attn.q_scale', 'model.layers.23.self_attn.attn.k_scale', 'model.layers.10.mlp.gate_up_proj.weight_scale_inv', 'model.layers.2.self_attn.attn.q_scale', 'model.layers.16.self_attn.attn.k_scale', 'model.layers.10.self_attn.attn.v_scale', 'model.layers.16.mlp.gate_up_proj.weight_scale_inv', 'model.layers.14.self_attn.attn.q_scale', 'model.layers.23.self_attn.attn.prob_scale', 'model.layers.1.mlp.gate_up_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.prob_scale', 'model.layers.7.self_attn.attn.prob_scale', 'model.layers.23.self_attn.attn.q_scale', 'model.layers.15.mlp.gate_up_proj.weight_scale_inv', 'model.layers.10.self_attn.o_proj.weight_scale_inv', 'model.layers.19.self_attn.o_proj.weight_scale_inv', 'model.layers.18.mlp.gate_up_proj.weight_scale_inv', 'model.layers.16.self_attn.qkv_proj.weight_scale_inv', 'model.layers.23.self_attn.qkv_proj.weight_scale_inv', 'model.layers.14.mlp.gate_up_proj.weight_scale_inv', 'model.layers.25.self_attn.attn.v_scale', 'model.layers.11.mlp.down_proj.weight_scale_inv', 'model.layers.10.self_attn.attn.k_scale', 'model.layers.14.self_attn.attn.v_scale', 'model.layers.3.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.qkv_proj.weight_scale_inv', 'model.layers.11.self_attn.attn.v_scale', 'model.layers.13.self_attn.attn.k_scale', 'model.layers.3.self_attn.attn.prob_scale', 'model.layers.17.mlp.gate_up_proj.weight_scale_inv', 'model.layers.7.self_attn.o_proj.weight_scale_inv', 'model.layers.26.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.k_scale', 'model.layers.8.self_attn.attn.q_scale', 'model.layers.8.self_attn.attn.prob_scale', 'model.layers.3.mlp.down_proj.weight_scale_inv', 'model.layers.6.self_attn.qkv_proj.weight_scale_inv', 'model.layers.6.self_attn.attn.q_scale', 'model.layers.16.self_attn.o_proj.weight_scale_inv', 'model.layers.18.self_attn.attn.q_scale', 'model.layers.8.mlp.down_proj.weight_scale_inv', 'model.layers.15.self_attn.qkv_proj.weight_scale_inv', 'model.layers.15.self_attn.o_proj.weight_scale_inv', 'model.layers.5.self_attn.attn.k_scale', 'model.layers.6.self_attn.attn.prob_scale', 'model.layers.21.self_attn.attn.k_scale', 'model.layers.25.mlp.down_proj.weight_scale_inv', 'model.layers.16.self_attn.attn.prob_scale', 'model.layers.6.mlp.down_proj.weight_scale_inv', 'model.layers.21.self_attn.qkv_proj.weight_scale_inv', 'model.layers.24.self_attn.attn.q_scale', 'model.layers.1.self_attn.qkv_proj.weight_scale_inv', 'model.layers.1.self_attn.attn.k_scale', 'model.layers.17.self_attn.attn.prob_scale', 'model.layers.15.self_attn.attn.v_scale'}

Hi, I noticed that even though I set "draft_tensor_parallel_size": 1, both the logs and GPU memory usage show that the draft model is still being loaded on every GPU.

Could you confirm whether this command actually supports running the target model with TP=2 and the draft model with TP=1?
---
### tomasruizt @ 2025-10-15T06:22:28Z

@ggg-s setting different TP sizes for target and draft model will raise an error in the latest commit. If you don't specify the TP size for the draft model, it will by default be equal to the target model TP.
---
### ggg-s @ 2025-10-15T06:31:41Z

@tomasruizt yep. I pulled the latest commit and locally removed to test heterogenous TP. With set in , the service starts without errors, but the drafter still gets instantiated on every TP rank — logs show each worker printing “Loading drafter model…”, and GPU memory usage is identical across GPUs.self._raise_if_draft_tp_mismatch()"draft_tensor_parallel_size": 1speculative-config

So even though the check is bypassed, the behavior remains “replicated-per-rank” rather than a single TP=1 drafter.  If there’s a flag or code path to pin the drafter to rank0 (and broadcast results), I’m happy to try it.
---
### tomasruizt @ 2025-10-15T06:36:10Z

@ggg-s that's expected. As I said:
> If you don't specify the TP size for the draft model, it will by default be equal to the target model TP.

The only way to have TP = 1 on the drafter is to have TP = 1 on the target model, atm.
---
### ggg-s @ 2025-10-15T06:38:46Z

@tomasruizt Got it, thank you for your explanation！

---
### QingNagi @ 2025-10-16T06:52:54Z

I encountered a new issue involving tensor dimension mismatches that occur under high concurrency conditions. What causes this problem?
My command:
vllm serve Qwen3/Qwen3-32B-FP8 \
--served-model-name "qwen3" \
--gpu-memory-utilization 0.9 \
--tensor-parallel-size 1 \
--trust-remote-code \
--max-model-len 10000 \
--speculative-config '{"method": "draft_model", "model": "Qwen3/Qwen3-0.6B", "num_speculative_tokens": 5, "max-model-len": 10000, "disable_padded_drafter_batch": true}'  \
--override-generation-config '{"temperature" : 0}' &

vllm bench serve --model qwen3fd \
--dataset-name spec_bench \
--tokenizer Qwen3/Qwen3-32B-FP8 \
--percentile-metrics ttft,tpot,itl,e2el \
--dataset-path  ./question.jsonl \
--num-prompts -1 \
--max-concurrency 64 

Error message:
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710] EngineCore encountered a fatal error.
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710] Traceback (most recent call last):
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 701, in run_engine_core
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     engine_core.run_busy_loop()
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 728, in run_busy_loop
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     self._process_engine_step()
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 754, in _process_engine_step
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     outputs, model_executed = self.step_fn()
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 284, in step
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     model_output = self.execute_model_with_error_logging(
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 270, in execute_model_with_error_logging
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     raise err
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/engine/core.py", line 261, in execute_model_with_error_logging
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     return model_fn(scheduler_output)
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/executor/abstract.py", line 103, in execute_model
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     output = self.collective_rpc("execute_model",
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/executor/uniproc_executor.py", line 83, in collective_rpc
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     return [run_method(self.driver_worker, method, args, kwargs)]
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/utils/__init__.py", line 3122, in run_method
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     return func(*args, **kwargs)
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     return func(*args, **kwargs)
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py", line 447, in execute_model
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     output = self.model_runner.execute_model(scheduler_output,
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     return func(*args, **kwargs)
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py", line 2432, in execute_model
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     propose_draft_token_ids(valid_sampled_token_ids)
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py", line 2379, in propose_draft_token_ids
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     self._draft_token_ids = self.propose_draft_token_ids(
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/worker/gpu_model_runner.py", line 2590, in propose_draft_token_ids
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     draft_token_ids = self.drafter.propose(
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/spec_decode/draft_model.py", line 71, in propose
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     draft_token_ids = super().propose(
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py", line 216, in propose
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     self.set_input_ids_first_pass(
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]   File "/opt/vllm_env/lib/python3.10/site-packages/vllm/v1/spec_decode/draft_model.py", line 127, in set_input_ids_first_pass
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710]     self.input_ids[:num_tokens] = target_token_ids
(EngineCore_DP0 pid=139546) ERROR 10-14 05:36:09 [core.py:710] RuntimeError: The expanded size of the tensor (2048) must match the existing size (2062) at non-singleton dimension 0.  Target sizes: [2048].  Tensor sizes: [2062]
---
### tomasruizt @ 2025-10-16T06:56:39Z

I think I know precisely the line causing it. But to reproduce I would need the questions.jsonl file. Would it be possible to share it? @QingNagi 
---
### QingNagi @ 2025-10-16T06:59:54Z

> I think I know precisely the line causing it. But to reproduce I would need the questions.jsonl file. Would it be possible to share it? @QingNagi

Sure! Download the dataset using: wget https://raw.githubusercontent.com/hemingkx/Spec-Bench/refs/heads/main/data/spec_bench/question.jsonl
---
### tomasruizt @ 2025-10-16T08:14:29Z

@QingNagi can you share what commit you are testing on?
---
### QingNagi @ 2025-10-16T08:31:36Z

> @QingNagi can you share what commit you are testing on?

vllm==0.11.0
---
### tomasruizt @ 2025-10-16T08:33:07Z

@QingNagi I meant which commit of this branch?
---
### QingNagi @ 2025-10-16T08:37:26Z

37f013ecf11426cda676551017e4c34016b58d52 this commit.
---
### QingNagi @ 2025-10-16T08:52:12Z

> @QingNagi I meant which commit of this branch?

When I add the --request-rate parameter, the problem doesn't occur, which is strange.
---
### tomasruizt @ 2025-10-16T08:53:25Z

@QingNagi Are you on the vLLM Slack? We can chat over there

Edit: I'm not able to reproduce the issue. Can you try with the latest commit of this branch? It would help with reproduction if the error happens with a smaller model. e.g. target=Qwen3-1.7B-FP8, draft=Qwen3-0.6B, also if the error happens with --enforce-eager, and if you let me what GPU you are running on.
---
### QingNagi @ 2025-10-16T09:13:00Z

Ok, I will try it on the latest commit.
---
### ggg-s @ 2025-10-16T12:23:04Z

@tomasruizt I was wondering if this PR ([#26937](https://github.com/vllm-project/vllm/pull/26937)) could help improve the decoding speed of the draft model?
---
### benchislett @ 2025-10-16T14:19:13Z

@ggg-s currently the EAGLE/draft model is forced to be piecewise. We can optimize this aspect as a follow-up feature.
---
### ggg-s @ 2025-10-17T02:33:23Z

Got it!
---
### ggg-s @ 2025-10-18T04:22:12Z

“ I measured online throughput metrics using the commands below. Hardware was an RTX PRO 6000 96GB. After making sure the draft model also uses CUDA graph, SD has higher throughput than not using SD. See tables below.”


@tomasruizt Does using CUDA graph here refer to using the FULL CUDA graph for SD? If so, could you please let me know which parameters need to be set to enable SD to use it? Thank you!


---
### tomasruizt @ 2025-10-18T07:48:09Z

@ggg-s Full cuda graphs are not yet supported. That comment referred to piecewise CUDA graphs. They are used by default unless you pass `--enforce-eager`.
---
### ggg-s @ 2025-10-20T07:01:33Z

@tomasruizt Can I contact you on Slack?
---
### tomasruizt @ 2025-10-20T07:10:22Z

Here is my link: https://join.slack.com/shareDM/zt-3g4c4k5qe-o_KDj9TXkNCtZBRsumhwsA
---
### tomasruizt @ 2025-10-20T18:10:39Z

@LiuXiaoxuanPKU I ran throughput benchmarks on a multi-GPU (TP=4) setup with `Llama-3-70B`, and added the results to the bottom of the PR description. I found speedups up to batch_size=32. AFAIK you had a somewhat different results. If you can share those details, I can attempt to reproduce. Thanks for your review :)
---
### benchislett @ 2025-10-20T18:14:09Z

@tomasruizt do you have any comparison to EAGLE3? There's a head for Llama 3.3 70B: https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
---
### tomasruizt @ 2025-10-20T18:16:41Z

@benchislett I haven't run these benchmarks for EAGLE3, but I could compute that tomorrow for comparison.
---
### tomasruizt @ 2025-10-21T09:46:59Z

@benchislett I added the EAGLE3 benchmark for reference. It shows very good acceleration, strongest with small batch sizes. It's faster than the draft_model contributed here, but using draft_model doesn't require training a separate model, like EAGLE. Furthermore, there are follow-up optimizations that should substantially improve draft_model performance:
* (1) Making the draft model run on TP=1, regardless of the TP of the target model. The Qwen3 benchmarks on TP=1 in the PR description show that we might reach 2x acceleration this way.
* (2) the draft model should run with full CUDA graphs during pure decode steps (just like the target model). Profiling suggest that this could speed up draft_model decodes by ~20-30%.

I tried both for this PR, but found them non-trivial, so I stepped back. I will be away in November, so I think we should try to merge this PR before November.
---
### mergify @ 2025-10-24T17:09:14Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-11-04T01:31:34Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-11-21T03:45:32Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### gerayking @ 2025-12-10T13:45:00Z

Hi @tomasruizt , I found that the v1 engine still doesn’t support draft models speculative decoding method, and I noticed there’s an open PR for it. Where is this PR currently blocked, and what can I do to help move it forward?
---
### benchislett @ 2025-12-10T18:54:42Z

Here are the blockers in my opinion:
- Doesn't integrate with padded-batch spec decoding, simply due to implementation choices. Since we're hoping to use padded-batch more and more going forward, it's important that support be included.
- "extend" logic is hard to read and isn't very efficient. Causes a non-trivial bottleneck in input preparation according to my benchmarking. I have a custom kernel that fuses this logic but I think it is more generally applicable and I'm not sure how I want to scope it in the context of this PR. I think we can find a better way to implement the extending logic.
- Doesn't integrate perfectly in the current specdec framework. While this branch does reuse code with EAGLE, it still muddles a lot of the responsibilities. I think it needs a thorough refactoring pass to properly separate concerns here and find the right trade-off between complexity and code reuse. This may involve a preliminary PR to refactor vLLM's current spec decoding interfaces before we can contribute this code.
---
### tomasruizt @ 2025-12-10T19:15:05Z

@benchislett thanks for the concrete feedback! 

I would:
* (1) Implement support for padded batch
* (2) try reusing the Triton code you contributed for input preparation, and
* (3) reach out over slack to find a good way to separate concerns in the code. 

I can start with this in around 10 days. 
---
### ofirzaf @ 2025-12-10T20:20:13Z

> This may involve a preliminary PR to refactor vLLM's current spec decoding interfaces before we can contribute this code.

@benchislett What are the issues in the current interface in your opinion?
To me it seemed like Eagle was implemented first and then all the other methods were adjusted to follow the same flow even if it doesn't make much sense. I think there is a problem where different methods have a different flows.
Anyway, I would like to help shaping this
---
### mergify @ 2025-12-25T22:35:44Z

Hi @tomasruizt, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### mergify @ 2026-01-08T19:19:32Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-01-14T02:11:40Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @tomasruizt.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### benchislett @ 2026-01-14T04:12:12Z

Will mark as "ready" to start CI once you update to latest main.
---
### benchislett @ 2026-01-14T04:16:32Z

Another question: do you know if this will work for draft models with a different attention setup as the target model? For instance, Qwen3-4B has 32 attention heads per layer and Qwen3 235B A22B has 64 attention heads per layer. Is this expected to work somehow? What happens if you try to launch vLLM with this setup, does it crash with a meaningful error message?
---
### tomasruizt @ 2026-01-14T13:55:58Z

> Another question: do you know if this will work for draft models with a different attention setup as the target model? For instance, Qwen3-4B has 32 attention heads per layer and Qwen3 235B A22B has 64 attention heads per layer. Is this expected to work somehow? What happens if you try to launch vLLM with this setup, does it crash with a meaningful error message?

I haven't checked this case. Will do, and post the results in this comment.

**Edit**. @benchislett It worked ✅ I ran as target model Qwen3-235B, and as draft model Qwen3-0.6B. Below are the commands and the setup.

<details>

<summary>Setup & Results</summary>

server:
```shell
vllm serve Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --tensor-parallel-size 4 \
    --no-enable-prefix-caching \
    --speculative_config.method draft_model \
    --speculative_config.model Qwen/Qwen3-0.6B \
    --speculative_config.num_speculative_tokens 3 \
    --speculative_config.max_model_len 20000 \
    --max-model-len 20000
```

bench:
```shell
vllm bench serve \
    --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
    --dataset-name hf \
    --dataset-path philschmid/mt-bench \
    --num-prompts 80 \
    --max-concurrency 32 \
    --temperature 0.0 \
    --top-p 1.0
```

Results:
```shell
============ Serving Benchmark Result ============
Successful requests:                     80        
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  11.84     
Total input tokens:                      6078      
Total generated tokens:                  16765     
Request throughput (req/s):              6.76      
Output token throughput (tok/s):         1416.15   
Peak output token throughput (tok/s):    766.00    
Peak concurrent requests:                45.00     
Total token throughput (tok/s):          1929.56   
---------------Time to First Token----------------
Mean TTFT (ms):                          170.26    
Median TTFT (ms):                        90.12     
P99 TTFT (ms):                           302.57    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.56     
Median TPOT (ms):                        17.36     
P99 TPOT (ms):                           26.64     
---------------Inter-token Latency----------------
Mean ITL (ms):                           41.41     
Median ITL (ms):                         41.33     
P99 ITL (ms):                            49.37     
---------------Speculative Decoding---------------
Acceptance rate (%):                     44.89     
Acceptance length:                       2.35      
Drafts:                                  7135      
Draft tokens:                            21405     
Accepted tokens:                         9608      
Per-position acceptance (%):
  Position 0:                            62.34     
  Position 1:                            41.75     
  Position 2:                            30.57     
==================================================
```

</details>
---
### tomasruizt @ 2026-01-17T17:02:46Z

The failing CI test (v1-test-attention-b200) also fails on `main` (00e6402d56fb258e6958381b1f3ceb34217ba830) on a B200, so it's unrelated to this PR. @benchislett 
---
### singh-git10 @ 2026-01-27T02:39:09Z

@tomasruizt  Thanks a lot for enabling it. However, when I use GPT OSS 20B as both the target and the drafter, I encounter the error below:
""File "/home/ubuntu/test_code/vllm/vllm/v1/spec_decode/eagle.py", line 1312, in validate_same_kv_cache_group
(EngineCore_DP0 pid=1006172) ERROR 01-26 23:11:38 [core.py:935] len(
(EngineCore_DP0 pid=1006172) ERROR 01-26 23:11:38 [core.py:935] AssertionError: All drafting layers should belong to the same kv cache group"

Command to run for reproducing:
VLLM_USE_V1=1 python3 examples/offline_inference/spec_decode.py
--model-dir openai/gpt-oss-20b
--draft-model openai/gpt-oss-20b
--method draft_model
--num_spec_tokens 5
--dataset-name hf
--dataset-path philschmid/mt-bench
--num_prompts 100
--temp 0.0
--gpu-memory-utilization 0.9

---
### tomasruizt @ 2026-01-27T15:44:43Z

@singh-git10 thank you for opening the issue https://github.com/vllm-project/vllm/issues/33133. I left a comment there about how to potentially fix this.
---