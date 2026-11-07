# PR #29826: [Perf] Avoid pageable HtoD transfer in MinTokensLogitsProcessor

- field: decoding  | area: v1/sample
- author: jthomson04
- state: MERGED  created: 2025-12-02T00:38:03Z  merged: 2025-12-02T21:25:53Z
- url: https://github.com/vllm-project/vllm/pull/29826
- labels: ready, v1

## Body

<!-- markdownlint-disable -->

## Purpose
The `MinTokensLogitsProcessor` introduces a pageable HtoD transfer here: https://github.com/vllm-project/vllm/blob/342c4f147266e7d01a8a6a64754df7e0f7f8991a/vllm/v1/sample/logits_processor/builtin.py#L232. This is problematic, since it is synchronizing wrt the device. When running with `--async-scheduling`, this synchronizing behavior causes the scheduler to no longer be overlapped with model execution. At low concurrencies, this leads to a significant (up to 30%) increase in TPOT when including min_tokens in the request.

I collected some nsys traces illustrating this.

Without min_tokens in request:
<img width="1950" height="522" alt="image" src="https://github.com/user-attachments/assets/5f7809aa-3603-45dd-af2e-b993b9017ca1" />

With min_tokens in request:
<img width="1950" height="522" alt="image" src="https://github.com/user-attachments/assets/505e9124-8c71-46c4-a5e5-65ae7ac3a7bc" />


## Test Plan
Existing tests for `MinTokensLogitsProcessor` are sufficient; no further tests are required.

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

### jthomson04 @ 2025-12-02T02:49:30Z

Converted this to use the `.fill_(-float("inf"))` method.
---
### DarkLight1337 @ 2025-12-02T10:46:53Z

Please merge from main to fix the build failure
---
### njhill @ 2025-12-02T16:04:08Z

The test failure looks likely related given it's to a logitsproc test, I can look into it soon.
---
### njhill @ 2025-12-02T18:17:45Z

@jthomson04 @nvpohanh I just pushed an update to fix it.

Using `fill_` on an indexed tensor doesn't work since a new tensor is created. 
---