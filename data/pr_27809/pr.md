# PR #27809: [Model] Introduce Kimi Linear to vLLM

- field: attention  | area: model_executor/layers
- author: zhiyuan1i
- state: MERGED  created: 2025-10-30T12:25:21Z  merged: 2025-10-30T13:02:27Z
- url: https://github.com/vllm-project/vllm/pull/27809
- labels: documentation, new-model, v1

## Body

<!-- markdownlint-disable -->

## Purpose
Introducing Kimi Linear, an advanced hybrid attention model that combines the efficiency of [Kimi Delta Attention (KDA)](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda), a refined version of Gated DeltaNet, with reduced memory requirements and superior performance across short, long, and reinforcement learning contexts. By introducing an optimized gating mechanism, Kimi Linear significantly cuts down the need for large KV caches by up to 75%, enhancing hardware efficiency and boosting decoding throughput by up to $6\times$ for long-context tasks, such as those with up to 1M tokens.

Previous PR: https://github.com/vllm-project/vllm/pull/27654
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



## Comments

### mergify @ 2025-10-30T12:26:04Z

Documentation preview: https://vllm--27809.org.readthedocs.build/en/27809/
---