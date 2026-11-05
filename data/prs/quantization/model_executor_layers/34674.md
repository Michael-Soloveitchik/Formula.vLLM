# PR #34674: Adding Nemotron fp8 Triton MoE Config

- field: quantization  | area: model_executor/layers
- author: yugong333
- state: MERGED  created: 2026-02-17T07:03:00Z  merged: 2026-02-24T23:56:38Z
- url: https://github.com/vllm-project/vllm/pull/34674
- labels: 

## Body

<!-- markdownlint-disable -->

## Purpose

Uploading tuned FP8 MoE config, which is used in Nemotron-3-Nano-30B-A3B-FP8 with LoRA adatper.

## Test Plan

We test the performance via llmpef

## Test Result

See around ~5-7% in OTPS

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

(no discussion comments)