# PR #18745: [Core] Support inplace model weights loading

- field: quantization  | area: model_executor/model_loader
- author: 22quinn
- state: MERGED  created: 2025-05-27T05:53:10Z  merged: 2025-06-02T09:38:50Z
- url: https://github.com/vllm-project/vllm/pull/18745
- labels: tpu, ready, v1, rl

## Body

In RL, we sometimes initialize a model with dummy weights at the beginning of rollout. At a later stage, real weights are loaded with a `load_model` RPC call. Currently this is not well supported because `load_model` API always re-initialize the model. Three main changes in this PR:
- Model loader refactoring: Most model loaders now share a common `load_model` function defined in `BaseModelLoader`
- Add `load_weights` API on `BaseModelLoader`
- In V1 GPU model runner, if a model has already been initialized, call `load_weights` instead of `load_model`

E2E test:
```
# Save https://gist.github.com/foreverlms/7e3ead8118db578bcff454256336e391
torchrun  --standalone  --nnodes=1   --nproc-per-node=4 dummy_load_test.py
```

Before this PR:
> ValueError: Duplicate layer name: model.layers.0.self_attn.attn
Full log: https://gist.github.com/22quinn/6819a91091f3be252e5fba231716e59a

After this PR:
> [gpu_model_runner.py:1537] Model was already initialized. Loading weights inplace...

cc @jeejeelee @DarkLight1337 @houseroad 

## Comments

### github-actions @ 2025-05-27T05:53:20Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### foreverlms @ 2025-05-27T06:09:19Z

Should this work for V0/legacy backend too? if not, will it emit the message that not supported?
---
### 22quinn @ 2025-05-27T18:35:19Z

> Should this work for V0/legacy backend too? if not, will it emit the message that not supported?

The model loader changes are backward-compatible with v0. However, v0 code is now frozen (#18571), so the v0 worker (below) will stick to its existing behavior, no inplace loading supported.

https://github.com/vllm-project/vllm/blob/58738772410c5e0d60b61db39538a9b313d2d7ad/vllm/worker/worker.py#L196-L207
---
### houseroad @ 2025-06-01T01:40:21Z

Also we should create some e2e example for this optimization in the RL, and this can be done in a follow up PR.
---
### 22quinn @ 2025-06-01T09:03:00Z

> Overall, looks good to me. Could you rebase and test some model to verify the accuracy to ensure the model weights are actually loaded?

Added a unit test to make sure inplace loaded weights == non-inplace loaded weights
---
### omrisapir1 @ 2025-06-12T13:02:57Z

Can someone explain me what is the correct use of doing inplace model weights, using -

```

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

```
---
### houseroad @ 2025-06-12T13:59:59Z

@22quinn some e2e example will be helpful here. :-)
---
### 22quinn @ 2025-06-12T18:57:28Z

let me add some examples
---
### 22quinn @ 2025-06-15T07:18:08Z

@omrisapir1 while working on the example, I realize this PR may not expose the proper API to load the real weights. Currently it requires access to the model runner to update the loader config. I'll put up another PR to introduce a new RPC method
---
### vadimkantorov @ 2025-06-16T10:40:17Z

Does this new inplace weight loader support online bf16->fp8 quantization?

This is needed for the GRPO flow, where we need to frequently re-load new weights, and do online conversion of bf16->fp8 for faster rollouts (and rollouts take a 50-70% of wall clock time per iteration). Ideally, vllm should provide load_weights API for `quantization="fp8"` available in LLM engine instantiation (and when possible allow per-parameter operation to not force all-gather of all weights on a single gpu. typically, the weights are sharded in some way, and need to be synced/quantized into the potentially-sharded-differently-or-even-on-other-set-of-nodes vllm instance):
- https://github.com/vllm-project/vllm/issues/19020
- https://github.com/volcengine/verl/issues/1803

I also wonder if fp8 can also be sufficient in reference model log-prob computations
---
### 22quinn @ 2025-06-16T22:12:39Z

@vadimkantorov Currently it does not do online quantization. Quantization happens after loading all weights.
I'm going to add a (re)load_weights RPC API, but no plan for quantization at the moment
But it seems a valid ask - I'm interested to see how online quantization is implemented in other libraries. Do you have an example? I assume you still have to load the original weights before performing quantization?
---
### vadimkantorov @ 2025-06-17T16:22:45Z

> Do you have an example? I assume you still have to load the original weights before performing quantization?

I guess it depends on the quantization method. If it's possible to do per-parameter or per-module bf16->fp8 quantization (I wonder if https://huggingface.co/unsloth/Qwen3-8B-FP8#note-on-fp8 does only per-parameter?)

Maybe even vllm's own https://github.com/vllm-project/llm-compressor/tree/main/examples/quantization_w8a8_fp8 could be plugged for online per-parameter or per-layer usage? Or maybe https://github.com/pytorch/ao/tree/main?tab=readme-ov-file#-inference can be used for this quantization. The important thing for this GRPO rollout usecase is to be able to have run-time speed-ups in quantized (fp8).

If this is possible, then quantizing variant of in-place load_weights can operate per-parameter (and maybe even torch.compile-fuse with NCCL collectives?). And at least, at no moment one requires a bf16 copy of all weights.

---

And even if per-parameter is not possible, I think there is value in making accessible the `LLM(..., quantization="fp8")` method (whatever it does) as inplace mode via `.load_weights(...)`. For small models, prior passing the copy of all bf16 weights is realistic, and fp8 quantization is needed for inference speed-up.

Basically one needs to be able to transfer weights from FSDP2-pytorch bf16 model on a set of vllm instances (potentially co-located, or deployed on another set of nodes)...
---
### 22quinn @ 2025-06-20T03:51:37Z

@vadimkantorov I just read https://github.com/volcengine/verl/issues/1803 in detail. I feel the verl's current interaction with vllm makes quantization more complicated - verl directly accesses vllm model and `update_weights` in every vllm model is implemented individually.
This PR is about loading from a storage, seems not helpful to address the online weights update problems. We need some change on both sides.
P.S. I don't know how verl could use vllm v1 with its current implementation... Maybe it's still on v0?

I'd be interested to contribute to both sides, but I do not use verl myself so might need to find some E2E example runs (maybe you could provide one?) to understand it better.
---
### fxmarty-amd @ 2025-06-27T08:48:09Z

sglang has some online quantization support. If we are talking about online quantization while reloading checkpoints (is that the case @vadimkantorov ?), as the codebase is very similar to vllm, it should be easily portable.

There are two approaches that I see in sglang:
* online quantization happening in `process_weight_after_loading` => good because weight are sharded, bad because it means that all the high-precision weights are loaded on device first
* online quantization happening in an overridden `weight_loader` => tricky if checkpoint is not pre-sharded, you need to implement sharding yourself if you want to do online quantization per-shard. Good because there is no need to load the full high-precision checkpoint on device before online quant.
---
### foreverlms @ 2025-06-30T09:05:15Z

> sglang has some online quantization support. If we are talking about online quantization while reloading checkpoints (is that the case @vadimkantorov ?), as the codebase is very similar to vllm, it should be easily portable.
> 
> There are two approaches that I see in sglang:
> 
> * online quantization happening in `process_weight_after_loading` => good because weight are sharded, bad because it means that all the high-precision weights are loaded on device first
> * online quantization happening in an overridden `weight_loader` => tricky if checkpoint is not pre-sharded, you need to implement sharding yourself if you want to do online quantization per-shard. Good because there is no need to load the full high-precision checkpoint on device before online quant.

Why would the first way load all the weights? You mean full-unsharded weights? I think we could still load sharded weight for each device and do quantization?
---
### vadimkantorov @ 2025-06-30T14:11:20Z

> Why would the first way load all the weights?

I think this refers to that all the parameters need to be loaded first into the vllm.LLM model
---