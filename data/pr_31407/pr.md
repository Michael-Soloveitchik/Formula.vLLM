# PR #31407: Add Fused MoE Triton kernels for GLM-4.5-Air, GLM-4.5v, GLM-4.6v on 2x RTX Pro 6000

- field: gpu_kernels  | area: model_executor/layers
- author: mratsim
- state: MERGED  created: 2025-12-27T10:00:14Z  merged: 2025-12-28T16:38:34Z
- url: https://github.com/vllm-project/vllm/pull/31407
- labels: performance, ready

## Body

## Purpose

Add optimized config for 2x RTX Pro 6000 for:
- GLM-4.5-Air model (N=129)
- GLM-4.5V, GLM-4.6V and GLM-4.5-Air finetunes with no MTP layers

Note: those were produced through SGLang tuner.

## Test Plan

Tested locally on vLLM and in SGLang

## Test Result

It works. See comment: https://github.com/vllm-project/vllm/issues/26838#issuecomment-3669355236

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mratsim @ 2025-12-28T10:01:10Z

The buildkit/ci/pr failed due to Github connection issue:

https://buildkite.com/vllm/ci/builds/44999/steps/canvas?sid=019b61a6-cf54-424d-8e10-345b1c032bd4

<img width="2648" height="1204" alt="image" src="https://github.com/user-attachments/assets/62452b03-c84b-46d9-8a85-4aed289646d4" />

---