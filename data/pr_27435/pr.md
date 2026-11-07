# PR #27435: Load tuned fused_moe_lora shrink and expand kernel configs separately

- field: gpu_kernels  | area: lora/ops
- author: yugong333
- state: MERGED  created: 2025-10-23T20:36:50Z  merged: 2025-11-04T10:27:35Z
- url: https://github.com/vllm-project/vllm/pull/27435
- labels: performance, ready

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.


## Purpose
1. Loading the kernel configs of `fused_moe_lora` kernel
2. Separate `shrink` and `expand` kernel configs in the `fused_moe_lora` function
3. Adding `num_stages` and `num_warps` parameters in the configs

Note: Based on PR #21229 and #26319

## Test Plan

To benchmark the shrink and expand kernel in gate/up and down in the benchmark_lora.py: 

`python3 benchmarks/kernels/benchmark_lora.py list_bench --arg-pool-size 32 \
    --batch-sizes 1 \
    --dtype torch.bfloat16 \
    --hidden-sizes 2048 \
    --num-loras 16 \
    --lora-ranks 16 \
    --num-loras 4 \
    --op-types fused_moe_lora_gate_up_shrink \
    --seq-lengths 1 \
    --sort-by-lora-id 1 \
    --cuda-graph-nops 32 \
    --top-k-nums 4 \
    --num-experts 128`

## Test Result

Together with #26319 we can improve the OTPS 80% - 90% in GPT-OSS-120B when concurrency is 1 or 2.
 
<img width="995" height="210" alt="image" src="https://github.com/user-attachments/assets/89c00c96-d31d-40fa-82b9-0368efa48656" />


## (Optional) Documentation Update



## Comments

### mergify @ 2025-10-27T14:39:16Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @yugong333.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-11-01T18:32:16Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @yugong333.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### varun-sundar-rabindranath @ 2025-11-02T03:00:29Z

Thank you for working on the @yugong333  🙌 

I left a few minor comments. But it generally looks good to me. 

About the performance numbers reported, are those for the default config ? Also can you share some numbers for higher concurrency ? (IIUC concurrency refers to number of requests running concurrently - if that is the case, 8 seems too low, can you report numbers for concurrency in multiples of 32 upto 512)  - Mostly worried about regressions for decode. Thanks. 
---
### mergify @ 2025-11-03T21:15:36Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @yugong333.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### dcmaddix @ 2025-11-04T00:04:06Z

> LGTM ! Thank you for working on this @yugong333 . I left some minor nit that'd be nice to have before landing this PR.

Great thanks a lot @varun-sundar-rabindranath!
---