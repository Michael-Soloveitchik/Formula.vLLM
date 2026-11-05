# PR #31034: [P/D] rework mooncake connector and introduce its bootstrap server

- field: scheduling  | area: distributed/kv_transfer
- author: dtcccc
- state: MERGED  created: 2025-12-19T15:33:31Z  merged: 2026-02-03T16:08:26Z
- url: https://github.com/vllm-project/vllm/pull/31034
- labels: documentation, ready, v1, kv-connector

## Body

<!-- markdownlint-disable -->

## Purpose

Rework mooncake connector to achieve better performance and prepare for more features in future.
Introduce a central bootstrap server on P.
<img width="648" height="406" alt="1" src="https://github.com/user-attachments/assets/0a899877-14ff-4bd2-a201-1faa18025c6c" />
init phase:
All P workers register their info (dp/tp/pp rank, zmq worker addr) with the bootstrap server.
After all P workers finished registering, proxy and D workers can query when they meet a new engine_id.

Note:
(deprecated)
> Since #30739 `data_parallel_size` and `data_parallel_size_local` are stick to 1 for non-Moe models. So `origin_data_parallel_size` and `origin_data_parallel_size_local` are introduced to get the raw value.
> For non-Moe model with dp_size > 1, all its engines by dp have the same engine_id and unable to distinguish dp_ranks and workers. So `[engine_id][dp_rank]` is used. See comments in `MooncakeBootstrapServer` for detail.
> 
> After startup, the proxy will send reqs to both P and D concurrently. Due to #27987 P and D cannot know the exact request_id of each other. #32630 seems to be a solution, and I just try to drop the last 9 chars as workaround here. See comments in `TruncatingDict`.

Thanks to #33037 and #32937 we can drop all workarounds now.



This design is partially inspired by sglang, aiming to improve Time To First Token (TTFT) performance and to lay the groundwork for a future layerwise transfer feature.

With random dataset and max-concurrency = 1, TTFT on two A10 machines running Qwen2.5-7B-Instruct is improved:
random-input-len 128: 83.41ms -> 77.53ms
random-input-len 1024: 252.35ms ->246.25ms
This result shows TTFT win about 6ms from running in P and D simultaneously.


Other highlights of this PR:

- Introduced the proxy and example scripts for the new bootstrap server architecture, along with updated documentation to facilitate learning and testing.
- Prepare the future introduction of heterogeneous TP. For now, heterogeneous TP and PP is explicitly rejected.
- Pick a bugfix from https://github.com/vllm-project/vllm/pull/26012

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

### mergify @ 2025-12-19T15:34:15Z

Documentation preview: https://vllm--31034.org.readthedocs.build/en/31034/
---
### mergify @ 2025-12-19T16:00:52Z

Hi @dtcccc, the pre-commit checks have failed. Please run:

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
### NickLucche @ 2026-01-29T09:43:25Z

/gemini review
---
### mergify @ 2026-01-29T10:37:13Z

Hi @dtcccc, the pre-commit checks have failed. Please run:

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
### dtcccc @ 2026-01-29T10:58:21Z

> We should be looking to add a few tests to make the connector more maintainable though.

Agreed, this is already in my plan.
---
### mergify @ 2026-01-30T10:04:29Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @dtcccc.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---