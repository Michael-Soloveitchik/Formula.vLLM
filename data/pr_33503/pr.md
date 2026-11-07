# PR #33503: feat(spec_decode): fuse EAGLE step slot mapping and metadata updates

- field: decoding  | area: v1/spec_decode
- author: sladyn98
- state: MERGED  created: 2026-02-01T07:30:24Z  merged: 2026-03-11T04:35:33Z
- url: https://github.com/vllm-project/vllm/pull/33503
- labels: speculative-decoding, ready, v1

## Body

Reduce kernel launch overhead during EAGLE autoregressive drafting by fusing the slot mapping and metadata updates between steps into a single Triton kernel.

## Purpose

Implement kernel fusion for slot mapping and metadata updates between EAGLE autoregressive drafting steps to reduce kernel launch overhead.

**Addresses:** [#33455](https://github.com/vllm-project/vllm/issues/33455)

Between EAGLE steps, slot mapping and metadata updates were performed via many small PyTorch operations (position increment, clamping, block table lookup, slot mapping computation, seq_len updates), each causing separate GPU kernel launches. This PR fuses those operations into a single Triton kernel to reduce launch overhead during drafting.

## Test Plan

1. **Unit tests (new kernel)** — Run on CUDA machine:
   ```bash
   pytest tests/v1/spec_decode/test_eagle_step_kernel.py -v
   ```

2. **EAGLE propose tests** — Run existing EAGLE propose tests:
   ```bash
   pytest tests/v1/spec_decode/test_eagle.py::test_propose -v -k "eagle and not tree"
   ```

3. **EAGLE e2e correctness** — Run EAGLE correctness tests:
   ```bash
   pytest tests/v1/e2e/test_spec_decode.py::test_eagle_correctness -v -k "llama3_eagle3"
   ```

4. **Performance (optional)** — Compare tokens/sec before/after:
   ```bash
   python examples/offline_inference/spec_decode.py --method eagle --num-spec-tokens 8 --output-len 256
   ```

## Test Result

- **Unit tests:** `test_eagle_step_slot_mapping_kernel` and `test_eagle_step_slot_mapping_kernel_exceeds_max` pass. Kernel output matches the Python reference implementation for random inputs and edge cases (positions exceeding `max_model_len`).
- **Environment:** Tests were run on Mac without CUDA, where kernel tests are skipped. Full validation (including `test_propose` and `test_eagle_correctness`) should be run on a CUDA-enabled machine before merging.

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

### github-actions @ 2026-02-01T07:30:32Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀
---
### dosubot @ 2026-02-01T07:30:34Z

<!-- Answer -->
**Related Documentation**

No published documentation to review for changes on this repository.

[Write your first living document](https://app.dosu.dev/documents?utm_campaign=pr-doc-check-nudge)


<!-- Dosu Comment Footer -->
<sup>How did I do? Any feedback?</sup>&nbsp;&nbsp;[![Join Discord](https://img.shields.io/badge/join-5865F2?logo=discord&logoColor=white&label=)](https://go.dosu.dev/discord-bot)
---
### mergify @ 2026-02-01T07:34:55Z

Hi @sladyn98, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> or <code>markdownlint</code> failing?</summary>
> <br/>
> <code>mypy</code> and <code>markdownlint</code> are run differently in CI. If the failure is related to either of these checks, please use the following commands to run them locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> # For markdownlint
> pre-commit run --hook-stage manual markdownlint
> ```
> </details>
---
### tomasruizt @ 2026-02-01T08:32:34Z

How much does this change improve metrics? I guess TPOT should improve. You can find out with `vllm bench serve`.
---
### benchislett @ 2026-02-04T22:23:49Z

Nice work! I measure a 1.9% speedup for Llama 3.1 8B on 1xB200 with K=3 spec tokens, 1K ISL, BS=1.

Confirmed by torch profiler that the fusion is effective at reducing overhead:
Before:
<img width="1897" height="158" alt="image" src="https://github.com/user-attachments/assets/75e714ca-0445-4977-9312-88cbdaa98210" />
After:
<img width="1745" height="165" alt="image" src="https://github.com/user-attachments/assets/08de5eeb-9d41-4bdb-9474-1ac02766d460" />

Regarding the remaining ops here, we have:
- `input_ids = draft_token_ids_list[-1].int()` (the .int() is an elementwise cast here, unavoidable)
- `self._set_positions(batch_size, clamped_positions)` (DtoD copy, left a comment about this)
- `self.input_ids[:batch_size] = input_ids` (DtoD copy, unavoidable)
- a second set_positions call (duplicate, left a comment about this)
- `self.hidden_states[:batch_size] = hidden_states` (DtoD copy, unavoidable)
- `self._slot_mapping_buffer[num_actual:num_tokens].fill_(PADDING_SLOT_ID)` (padding fill for cudagraph padding range)

Overall this is very minimal overhead for drafting. Great work!


---
### mergify @ 2026-02-05T19:20:41Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @sladyn98.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### sladyn98 @ 2026-02-28T19:39:15Z

Sorry i was out last week, let me address the comments and get this merged

---
### mergify @ 2026-03-03T15:59:11Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @sladyn98.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-03-10T21:02:52Z

Hi @sladyn98, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
---