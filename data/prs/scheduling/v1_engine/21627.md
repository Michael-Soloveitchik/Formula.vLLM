# PR #21627: [Core] Move EngineCoreRequest to Request conversion out of EngineCore

- field: scheduling  | area: v1/engine
- author: linzebing
- state: MERGED  created: 2025-07-25T17:43:32Z  merged: 2025-07-30T22:00:54Z
- url: https://github.com/vllm-project/vllm/pull/21627
- labels: ready, v1

## Body

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.


## Purpose

In engine core thread, we used 18us to convert EngineCoreRequest to Request which is on model forward critical path.

Ideally, we should be able to move the conversion from engine core thread to input request thread to relax the logic from critical path.

There's an extra benefit for making the change, as Request became available in input processing threads, which would significantly simplify the pending changes for block hashing optimization mentioned in https://github.com/vllm-project/vllm/issues/21247

<img width="1926" height="572" alt="image" src="https://github.com/user-attachments/assets/e45b35c0-ab59-4e57-bf3a-bae4b1f258e3" />

<img width="962" height="351" alt="image" src="https://github.com/user-attachments/assets/1e11e413-83dd-4fbf-9482-1beea720e8a4" />


## Test Plan
```
export VLLM_USE_MODELSCOPE=False;
export VLLM_TORCH_PROFILER_DIR=~/vllm_profile; # for profiling
export CUDA_VISIBLE_DEVICES=4;
VLLM_USE_V1=1 vllm serve facebook/opt-125m \
    --swap-space 16 \
    --disable-log-requests \
    --host :: \
    --dtype float16
```
```
VLLM_USE_V1=1 vllm bench serve \
    --dataset-name random \
    --model facebook/opt-125m \
    --served-model-name facebook/opt-125m \
    --random-input-len 700 \
    --random-output-len 1 \
    --endpoint /v1/completions \
    --ignore-eos \
    --host localhost \
    --port 8000 \
    --num-prompts 100 \
    --profile
```
## Test Result
With the change, handle_client_request in engine core thread reduced from 35us to 7us.
<img width="1443" height="486" alt="image" src="https://github.com/user-attachments/assets/ed56e120-1176-415d-8b7f-e7ffd093928f" />

As expected, right now, Request conversion is executing in parallel with model forward:
<img width="1707" height="565" alt="Screenshot 2025-07-25 at 12 11 49 PM" src="https://github.com/user-attachments/assets/d872eb2b-dfac-49ec-8632-149b3eeb49d6" />

## (Optional) Documentation Update

<!--- pyml disable-next-line no-emphasis-as-heading -->


## Comments

### github-actions @ 2025-07-25T17:43:41Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org.

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

🚀
---
### linzebing @ 2025-07-25T17:46:20Z

@Jialin has generously delegated https://github.com/vllm-project/vllm/pull/21329 to me as my first PR to vLLM.

@njhill : I have addressed your comments and used a tuple to represent `request` and `current_wave`. Let me know if you have further comments
---
### linzebing @ 2025-07-29T18:05:15Z

Thanks @njhill for the review! I have added comments around thread safety.
---
### njhill @ 2025-07-30T11:16:03Z

@linzebing I think some of the CI test failures are related: https://buildkite.com/vllm/ci/builds/25313#01985782-1b21-4082-9849-1642052665bc/212-1966
---
### linzebing @ 2025-07-30T14:56:07Z

@njhill : thanks for pointing out! I forgot to examine the callsites of `EngineCore::add_request` in the tests. Now it should be fixed:
```
pytest tests/v1/engine/test_engine_core.py
```

I manually inspected the codebase, I should have changed all callsites of `EngineCore::add_request` now.
---
### mergify @ 2025-07-30T16:05:38Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @linzebing.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### linzebing @ 2025-07-30T21:08:12Z

Looks like the v1-test failure is related to CI. I tested locally, no issue:
```
pytest tests/v1/kv_connector/unit/test_shared_storage_connector.py
```
---
### Jialin @ 2025-07-30T21:30:44Z

> Looks like the v1-test failure is related to CI. I tested locally, no issue:
> 
> ```
> pytest tests/v1/kv_connector/unit/test_shared_storage_connector.py
> ```

@linzebing If you confirmed the failing test is not related, you could either rebase to kick off the CI or request force merge in sig-ci slack channel.
---