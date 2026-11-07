# PR #28687: [Performance] Reduce DeepGEMM N dim restriction from 128 to 64 multiplier 

- field: quantization  | area: .buildkite
- author: alexm-redhat
- state: MERGED  created: 2025-11-13T22:48:09Z  merged: 2025-11-19T23:47:13Z
- url: https://github.com/vllm-project/vllm/pull/28687
- labels: documentation, performance, ready, ci/build, deepseek

## Body

W8A8BlockFp8LinearOp class uses should_use_deepgemm_for_fp8_linear() function to determine which gemm to run with deepgemm fp8_gemm_nt kernel. Originally it was checking that N dim is divisible by 128, however, after some inspection of the deepgemm tests, I found that 64 works as well. This allows fused_qkv_a_proj to run with deepgemm kernel (and not cutlass) and is ~1.83X faster (after inspecting profiles).

With cutlass:
<img width="647" height="330" alt="image" src="https://github.com/user-attachments/assets/1750d4b3-7c23-438d-8229-2bea74e72f05" />

With deepgemm:
<img width="699" height="190" alt="image" src="https://github.com/user-attachments/assets/02d58bb5-3943-4900-8cb5-dd22a5872d2f" />

* lm_eval correctness verification shows that no accuracy loss.
* e2e TPOT improvement is from 28.32ms to 27.72 so about 2.2%



## Comments

### mergify @ 2025-11-13T22:48:47Z

Documentation preview: https://vllm--28687.org.readthedocs.build/en/28687/
---
### mgoin @ 2025-11-13T23:59:56Z

We should check this on Hopper and Blackwell, and make sure to expand the kernel unit test to check for edge cases
---
### ElizaWszola @ 2025-11-14T06:37:45Z

If we successfully remove the size restriction, it would be good if we could also remove the `should_use_deepgemm_for_fp8_linear` condition from `W8A8BlockFp8LinearOp.apply()` and decide whether we want to run deepgemm with `_dispatch_w8a8_blockscale_op()`
---
### alexm-redhat @ 2025-11-17T21:02:24Z

@yewentao256 I have verified dp==8 DSR1 (on H200) and the performance is actually better with this PR, about 1.5% for TPOT.
---
### alexm-redhat @ 2025-11-17T21:10:35Z

Also, I did a manual inspection of the deepgemm code and did not found any 128 division restriction for the gemm we use (fp8_gemm_nt). The code internally aligns the K dimension to a multiple of 128, but has no special logic for M and N.
---
### yewentao256 @ 2025-11-17T21:28:57Z

@alexm-redhat Sounds good, could you also test on Blackwell?
---
### alexm-redhat @ 2025-11-17T21:38:50Z

@yewentao256 no problem, checking blackwell
---
### alexm-redhat @ 2025-11-17T22:13:56Z

@yewentao256 @robertgshaw2-redhat Verified performance for DSR1 on 8xB200 system that has DeepGEMM installed. For TP==8 and DP==8 (+EP), performance with the PR or without is the same.
---
### yewentao256 @ 2025-11-17T23:06:04Z

> @yewentao256 @robertgshaw2-redhat Verified performance for DSR1 on 8xB200 system that has DeepGEMM installed. For TP==8 and DP==8 (+EP), performance with the PR or without is the same.

This is strange, could you show the command you use and the corresponding result?
---
### alexm-redhat @ 2025-11-18T02:14:43Z

@yewentao256 commands used:

**For client:**
`vllm bench serve --port 8123 --model deepseek-ai/DeepSeek-R1-0528 --dataset-name random --max-concurrency 32 --random-input-len 4 --random-output-len 1024 --num-prompts 80 --seed 1763417423 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 90,95,99 --ignore-eos --trust-remote-code`

**For server:**

_For TP_
`vllm serve deepseek-ai/DeepSeek-R1-0528 --port 8123 --no-enable-prefix-caching --max-num-seqs 128 --max-model-len 16384 --tensor-parallel-size 8
`
_For DP_
`vllm serve deepseek-ai/DeepSeek-R1-0528 --port 8123 --no-enable-prefix-caching --max-num-seqs 128 --max-model-len 16384 --enable-expert-parallel --data-parallel-size 8`

Results, for TP, TPOT is 38.12ms with PR, 38.89 without PR, for DP, TPOT is 37.9 with PR and 37.6 without PR.

---
### alexm-redhat @ 2025-11-18T18:15:40Z

@yewentao256 thanks for checking
---
### mgoin @ 2025-11-18T19:42:03Z

> If we successfully remove the size restriction, it would be good if we could also remove the `should_use_deepgemm_for_fp8_linear` condition from `W8A8BlockFp8LinearOp.apply()` and decide whether we want to run deepgemm with `_dispatch_w8a8_blockscale_op()`

I also agree with the suggestion from @ElizaWszola to simplify W8A8BlockFp8LinearOp
---
### alexm-redhat @ 2025-11-18T20:06:15Z

@mgoin was not aware of this test, will enable the shapes and see how to run it.
---
### alexm-redhat @ 2025-11-19T17:02:57Z

@mgoin after enabling the test I found out that N/K are restricted to the sizes that the test is using. So we do need to check that N%64==0 and K%128==0. In general, it worked because the fused_qkv N dim is 2112 which is divisible by 64. Thanks for pointing out this test.
---
### alexm-redhat @ 2025-11-19T17:03:16Z

Fixed the check to N%64 and K%128
---
### mgoin @ 2025-11-19T18:35:15Z

Thanks Alex, can you update the title and description to the final state?
---
### alexm-redhat @ 2025-11-19T19:13:30Z

@mgoin Updated title and description, thanks for the detailed review!
---