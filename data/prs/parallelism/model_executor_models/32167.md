# PR #32167: [Model] Re-implement Qwen3Omni Audio Encoder

- field: parallelism  | area: model_executor/models
- author: ywang96
- state: MERGED  created: 2026-01-12T10:44:32Z  merged: 2026-01-14T07:40:30Z
- url: https://github.com/vllm-project/vllm/pull/32167
- labels: ready, qwen

## Body


## Purpose
Re-implement Qwen3-Omni Audio Encoder with vLLM primitives with some vectorization improvements. - roughly 10% speedup at high batch sizes according to profiling run with TP=1.

main
```
(EngineCore_DP0 pid=2620374) INFO 01-12 23:47:30 [gpu_model_runner.py:4696] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 21 audio items of the maximum feature size.
(EngineCore_DP0 pid=2620374) Audio processing time: 0.3811814785003662 seconds
```

This PR
```
(EngineCore_DP0 pid=2620374) INFO 01-12 23:47:30 [gpu_model_runner.py:4696] Encoder cache will be initialized with a budget of 8192 tokens, and profiled with 21 audio items of the maximum feature size.
(EngineCore_DP0 pid=2620374) Audio processing time: 0.3417019844055176 seconds
```

Separate follow-up's: 
- Currently there is a bug on memory profiling with this model with audio modality only because of `get_mm_max_tokens_per_item` from Qwen2.5-VL.
- Enable DP for both audio and vision encoders

## Test Plan

## Test Result

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit ed2b2c7b70d8d8ab49acd02a0c68cda782bf70fd. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> **Re-implement Qwen3-Omni Audio Encoder with vLLM components**
> 
> - Adds vLLM-native audio encoder: `SinusoidsPositionEmbedding`, `Qwen3OmniMoeAudioAttention` (via `MMEncoderAttention` + `QKVParallelLinear`), `Qwen3OmniMoeAudioEncoderLayer`, and `Qwen3OmniMoeAudioEncoder` (3x Conv2d downsampling -> projection -> transformer -> MLP head), with TP-aware heads and vLLM attention backend selection
> - Vectorizes/chunks audio preprocessing and CNN path, computes `cu_seqlens` for chunked attention, and applies fp16-safe clamping; introduces HF weight mapping to stacked `qkv`
> - Integrates encoder into `Qwen3OmniMoeThinkerForConditionalGeneration` (passes `multimodal_config`, `prefix`; `_process_audio_input` now uses tensor output directly)
> - Removes explicit `flash_attn` requirement/forcing; relies on vLLM attention backend
> - Minor fix in `Qwen3_VisionPatchEmbed.forward` (drop unused var)
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit ed2b2c7b70d8d8ab49acd02a0c68cda782bf70fd. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> <sup>[Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) is generating a summary for commit 4574967fd1d61abcd05e972adf176f0794f46708. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>

---

> [!NOTE]
> Introduces a vLLM-native audio encoder and integrates it into the Qwen3-Omni-Moe thinker, removing the explicit `flash_attn` dependency and aligning with vLLM attention backends.
> 
> - Adds `SinusoidsPositionEmbedding`, `Qwen3OmniMoeAudioAttention` (via `MMEncoderAttention` + `QKVParallelLinear`), `Qwen3OmniMoeAudioEncoderLayer`, and `Qwen3OmniMoeAudioEncoder` (3× `Conv2d` downsampling → projection → transformer → MLP head) with TP-aware heads
> - Vectorizes/chunks audio preprocessing and CNN path, computes `cu_seqlens`/`max_seqlen` for chunked attention, and applies fp16-safe clamping
> - Implements HF→vLLM weight mapping for stacked `qkv` in the audio encoder
> - Wires encoder into `Qwen3OmniMoeThinkerForConditionalGeneration` (passes `multimodal_config`/`prefix`); `_process_audio_input` now returns tensor output directly
> - Minor cleanup: drop unused var in `Qwen3_VisionPatchEmbed.forward`; reorder transformer imports
> 
> <sup>Written by [Cursor Bugbot](https://cursor.com/dashboard?tab=bugbot) for commit 4574967fd1d61abcd05e972adf176f0794f46708. This will update automatically on new commits. Configure [here](https://cursor.com/dashboard?tab=bugbot).</sup>


## Comments

(no discussion comments)