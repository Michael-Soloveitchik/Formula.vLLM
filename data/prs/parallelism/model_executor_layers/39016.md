# PR #39016: [MoE] Triton MoE Perf regression - restore low latency path

- field: gpu_kernels  | area: model_executor/layers
- author: milesial
- state: MERGED  created: 2026-04-05T05:43:29Z  merged: 2026-04-21T06:37:11Z
- url: https://github.com/vllm-project/vllm/pull/39016
- labels: ready

## Body

The refactor in https://github.com/vllm-project/vllm/pull/36286 made it so that the unquantized MoE path taken now bypasses the low-latency optimization introduced in https://github.com/vllm-project/vllm/pull/29354.
FP8 regression was also introduced by https://github.com/vllm-project/vllm/pull/30825.

This PR re-introduces the optimization in the new modular path.


Impact is around 3-4% tok/s on B200 for nemotron nano 3. (ISL50 OSL1000, BS1)

BF16

 | Config | tok/s/user | Req latency | Delta |
  |---|---:|---:|---:|
  | Baseline | 250.42 | 4047.52 ms | |
  | This PR | 258.02 | 3931.10 ms | +3.03% tok/s, -2.88% latency |


FP8

 | Config | tok/s/user | Req latency | Delta |
  |---|---:|---:|---:|
  | Baseline | 392.2 | 2,570.03 ms | |
  | This PR | 409.27 | 2,462.04| +4.35% tok/s, -4.20% latency |


## Comments

### milesial @ 2026-04-05T05:45:42Z

@yzong-rh @robertgshaw2-redhat @RunkaiTao 
---
### robertgshaw2-redhat @ 2026-04-05T14:08:16Z

I dont follow how https://github.com/vllm-project/vllm/pull/36286 could have caused this?
---
### yzong-rh @ 2026-04-05T19:55:23Z

Good find!
It seems the regression on unquantized backends was first introduced in #31052.

- Before, `UnquantizedFusedMoEMethod` called `fused_experts()`, which called `fused_experts_impl()`, which eventually `invoke_fused_moe_kernel()`.
- After, `UnquantizedFusedMoEMethod` calls `TritonExperts.apply()`, which `invoke_fused_moe_kernel()` directly, skipping the optimization that was added to `fused_experts_impl()`.

Similarly we lost this optimization when fp8 backends moved to modular kernels in #30825.

<details>
<summary> FMI: </summary>
The optimization seems to be to skip `moe_align_block_size` (group and pad token ids by experts) when the batch size (`num_tokens * top_k_num`) << number of experts. This results in one tile per item in batch instead of one tile per activated expert but this is acceptable when each expert receives fewer than 1 token on average.
</details>

This change would affect FP8 and other WNA16 formats. Would be good to test FP8 perf as well like the original PR.
---
### milesial @ 2026-04-07T16:41:48Z

Thanks, this does indeed apply to FP8 too, added some benchmarks
---
### milesial @ 2026-04-15T19:33:27Z

Bumping this, should be ready to merge
---
### milesial @ 2026-04-21T06:33:31Z

bump @mgoin 
---