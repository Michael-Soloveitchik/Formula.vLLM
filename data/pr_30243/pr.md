# PR #30243: [LoRA]  Reduce the loading time of MoE LoRA

- field: parallelism  | area: lora/models.py
- author: jeejeelee
- state: MERGED  created: 2025-12-08T07:41:22Z  merged: 2025-12-08T13:29:47Z
- url: https://github.com/vllm-project/vllm/pull/30243
- labels: ready

## Body

<!-- markdownlint-disable -->

## Purpose
Follows up on https://github.com/vllm-project/vllm/pull/29322

Merging all LoRA weights before executing `pin_memory` related logic can greatly reduce the overhead caused by pin memory

Reduces disk loading time for LoRA weights by ~2x

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

### mergify @ 2025-12-08T08:46:23Z

Hi @jeejeelee, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.
---
### DarkLight1337 @ 2025-12-08T08:48:47Z

Please fix pre-commit
---