# PR #36012: [Performance] Add prefetch for checkpoints to OS page cache

- field: quantization  | area: model_executor/models
- author: arpera
- state: CLOSED  created: 2026-03-04T14:32:19Z  merged: 2026-03-16T11:32:02Z
- url: https://github.com/vllm-project/vllm/pull/36012
- labels: ready

## Body

## Purpose

Add a new `safetensors_load_strategy` option **"prefetch"** to speed up the model loading phase. Prefetch is designed for speeding up the model loading step in case a model is fully on disk and not yet in RAM. When enabled, checkpoint files are preloaded into the OS page cache (via sequential reads in 16MB blocks) in a background thread in parallel with workers parsing tensors from those files. This overlaps prefetch I/O with tensor parsing and is useful on network storage or high-latency storage.

## Test Result

Testing was done on model `deepseek-ai/DeepSeek-V3-0324` on 8xB200 with:

```bash
vllm serve deepseek-ai/DeepSeek-V3-0324 -tp 8 --safetensors-load-strategy prefetch
```

Results on two internal servers:

| Server   | Weights loading time without prefetch (s) | Weights loading time with prefetch (s) |
|----------|-------------------------------------------|----------------------------------------|
| Server 1 | 1500                                      | 435                                    |
| Server 2 | 3660                                      | 215                                    |

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

### github-actions[bot] @ 2026-03-04T14:36:28Z

👋 Hi! Thank you for contributing to the vLLM project.

💬 Join our developer Slack at https://slack.vllm.ai to discuss your PR in #pr-reviews, coordinate on features in #feat- channels, or join special interest groups in #sig- channels.

Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. 

You ask your reviewers to trigger select CI tests on top of `fastcheck` CI. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

To run CI, PR reviewers can either: Add `ready` label to the PR or enable auto-merge.

If you have any questions, please reach out to us on Slack at https://slack.vllm.ai.

🚀

### simon-mo @ 2026-03-11T19:04:37Z

How is this different from eager mode? I guess it is pipelined? I'm in favor of speeding up model loading and wonder what are the ways to turn this on by default. How do we figure out the size of page cache to touch ahead of time? How about using a library instead, I know there's binary like `vmtouch`. 

@claude review

### arpera @ 2026-03-11T20:23:26Z

> How is this different from eager mode?

With eager mode, every checkpoint is read by every worker. In my approach, each checkpoint is touched exactly once, which allows us to prefetch checkpoints from disk into RAM. After that, workers read checkpoints from RAM instead of disk, which significantly reduces disk I/O pressure.

> I guess it is pipelined?

Yes. Checkpoints are sharded across workers, and each worker asynchronously prefetches its own files.

> What are the ways to turn this on by default?

My current plan is to send a follow-up patch after this merge that will enable prefetching by default when the system has enough RAM to hold the prefetched checkpoints.

> How do we figure out the size of page cache to touch ahead of time?

We can base this on the available RAM on the machine. For example, if the model fits into say 0.9 * RAM_SIZE, we can safely prefetch and speed up model loading by a large factor. In TensorRT-LLM this is implemented as [follows](https://github.com/NVIDIA/TensorRT-LLM/blob/0350b7f229bc25c76573b93bc32d1ebf289241db/tensorrt_llm/_torch/models/checkpoints/hf/weight_loader.py#L40-L59). This functionality will be done in the same follow-up patch as turning prefetching mode on by default.

> How about using a library instead, I know there's binary like vmtouch.

Thanks for the idea. In a future I plan to experiment with different ways of prefetching checkpoints. The optimal approach depends a lot on the user’s system characteristics, primarily disk I/O throughput and filesystem type. The current implementation has shown good performance when loading the DeepSeek V3 model on real production servers, and it is simple and easy to reason about. But I'm open to iterating on this and trying alternative prefetching approaches in follow-up changes.

### vadiklyutiy @ 2026-03-11T20:40:12Z

Regarding `vmtouch`. It's a good approach, but my concern is `vmtouch` isn't a part of default Ubuntu installation, and user without root access won't be able to use it

### mergify[bot] @ 2026-03-13T07:24:07Z

Hi @arpera, the pre-commit checks have failed. Please run:

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

### arpera @ 2026-03-13T17:31:53Z

@mgoin, @simon-mo this PR is ready to merge. Please merge it.
