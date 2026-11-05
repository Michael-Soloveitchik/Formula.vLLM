# PR #26440: [Performance] Dual stream execution of "shared_experts" and "selected_experts" inside FusedMoE

- field: parallelism  | area: model_executor/layers
- author: alexm-redhat
- state: MERGED  created: 2025-10-08T19:46:58Z  merged: 2025-10-21T21:38:29Z
- url: https://github.com/vllm-project/vllm/pull/26440
- labels: documentation, ready, deepseek

## Body

This PR executes the shared_experts part of the FusedMoE on a separate GPU stream, so that the execution is parallelized with the "selected_experts" part. This is possible since the outputs of both are independent and are later combined. Thanks @wenscarl for pointing this out.

For DeepSeekR1 FP8 with Flashinfer latency kernels (trtllm-gen) on 8xB200s batch size 32, the TPOT improves from 23.35ms to 22.09ms (with latest FlashInfer codebase), so about ~5.7% e2e improvement.




## Comments

### mergify @ 2025-10-08T19:47:33Z

Documentation preview: https://vllm--26440.org.readthedocs.build/en/26440/
---
### mergify @ 2025-10-11T05:20:23Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @alexm-redhat.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### chatgpt-codex-connector @ 2025-10-13T20:30:01Z


### 💡 Codex Review

https://github.com/vllm-project/vllm/blob/70bfd91ced1966789ba9d4d5c7c06b9875fef706/vllm/model_executor/models/deepseek_v2.py#L276-L283
**<sub><sub>![P0 Badge](https://img.shields.io/badge/P0-red?style=flat)</sub></sub>  Router logits skipped when SharedFusedMoE runs on GPU**

In the SharedFusedMoE path the gate is no longer called before invoking the fused MoE kernel. `DeepseekV2MoE.forward` now calls `self.experts(hidden_states, router_logits=hidden_states)` whenever `self.experts` is a SharedFusedMoE, but `FusedMoE.forward_native` (the GPU path used by `super().forward`) still expects `router_logits` to already contain gate outputs and never invokes `self.gate`. As a result, the custom CUDA op receives `hidden_states` instead of logits, which both violates the expected `(num_tokens, n_experts)` shape and skips routing entirely, leading to runtime errors or incorrect routing for every GPU execution of SharedFusedMoE. The gating step needs to be reinstated or moved inside the GPU path before calling the custom op.
    

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

Your team has set up Codex to review pull requests in this repo. Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.


</details>
---
### LucasWilkinson @ 2025-10-20T13:13:46Z

Do you mind just quickly checking that this doesnt break DBO (i.e. `--enable-dbo`, you can test it with DeepSeek-V2-Lite with dp > 1 and DeepEP LL)
---
### alexm-redhat @ 2025-10-20T13:57:42Z

@LucasWilkinson will check now
---
### ZJY0516 @ 2025-10-27T12:26:16Z

I was wondering whether this can be done throgh `torch.compile` custom pass?
---
### vadiklyutiy @ 2025-11-07T20:22:44Z

Shouldn't we also use `tensor.record_stream()` ?
---