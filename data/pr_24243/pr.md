# PR #24243: Eagle3 that supports the Minicpm3 model

- field: decoding  | area: vllm/config
- author: LDLINGLINGLING
- state: MERGED  created: 2025-09-04T11:19:47Z  merged: 2025-09-26T17:04:58Z
- url: https://github.com/vllm-project/vllm/pull/24243
- labels: ready

## Body

## Purpose
Eagle3 that supports the Minicpm3 model
## Test Plan
————————————————
200 samples, 10 concurrency, single RTX 4090:​​
Total output tokens: 106,678
Number of draft proposals: 43,221
Number of draft tokens generated: 259,326
Number of accepted tokens: 63,669
Mean acceptance length: 2.47

Acceptance rate at token 0: 0.58
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.22
Acceptance rate at token 3: 0.15
Acceptance rate at token 4: 0.11

Baseline (non-speculative): 215.84 s
Speculative decoding model: 133.24 s
Speedup: 1.62×

​20 samples, single concurrency, single RTX 4090:​​
Total output tokens: 21,690
Number of draft proposals: 8,914
Number of draft tokens generated: 53,484
Number of accepted tokens: 12,814
Mean acceptance length: 2.44

Acceptance rate at token 0: 0.57
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.21
Acceptance rate at token 3: 0.14
Acceptance rate at token 4: 0.10

Baseline (non-speculative): 382.29 s
Speculative decoding model: 224.20 s
Speedup: 1.71×
Average acceptance length: 2.44

​100 samples, 5 concurrency, single RTX 4090:​​
Total output tokens: 103,920
Number of draft proposals: 42,227
Number of draft tokens generated: 253,362
Number of accepted tokens: 61,896
Mean acceptance length: 2.47

Acceptance rate at token 0: 0.58
Acceptance rate at token 1: 0.34
Acceptance rate at token 2: 0.22
Acceptance rate at token 3: 0.15
Acceptance rate at token 4: 0.10

Baseline (non-speculative): 412.49 s
Speculative decoding model: 238.08 s
Speedup: 1.73×
Average acceptance length: 2.47
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

### youkaichao @ 2025-09-08T08:05:54Z

cc @luccafong @benchislett can you help take a look?
---
### benchislett @ 2025-09-09T13:59:38Z

Overall I have no issues with this PR besides the nitpicks [Edit: and the CI failures. Please fix pre-commit].

However, we are seeing a steady increase in the support of EAGLE3 for new models. We should discuss (separately to this PR) how we want to handle this build-out, as currently there is a lot of duplicated code in every model which needs to support EAGLE3.
---
### luccafong @ 2025-09-11T04:44:57Z

> Overall I have no issues with this PR besides the nitpicks [Edit: and the CI failures. Please fix pre-commit].
> 
> However, we are seeing a steady increase in the support of EAGLE3 for new models. We should discuss (separately to this PR) how we want to handle this build-out, as currently there is a lot of duplicated code in every model which needs to support EAGLE3.

yeah, also feels can be simplified for the interface methods to be merged  https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py#L2337-L2338 
---
### luccafong @ 2025-09-12T02:44:05Z

@LDLINGLINGLING could you fix pre-commit errors?
---
### mergify @ 2025-09-16T13:06:31Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @LDLINGLINGLING.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### hmellor @ 2025-09-16T18:39:56Z

@luccafong / @benchislett feel free to merge this if the tests pass now
---
### LDLINGLINGLING @ 2025-09-17T06:25:20Z

> This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @LDLINGLINGLING.
> 
> https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

ok
---
### LDLINGLINGLING @ 2025-09-17T06:32:31Z

> This pull request has merge conflicts that must be resolved before it can be merged. Please rebase the PR, @LDLINGLINGLING.
> 
> https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork

Thank you very much, this issue has been resolved.
---
### mergify @ 2025-09-22T19:42:52Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @LDLINGLINGLING.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---