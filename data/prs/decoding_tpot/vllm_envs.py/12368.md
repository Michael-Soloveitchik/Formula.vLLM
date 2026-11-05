# PR #12368: [Core] add and implement `VLLM_LOGITS_PROCESSOR_THREADS`

- field: scheduling  | area: vllm/envs.py
- author: akeshet
- state: MERGED  created: 2025-01-23T20:42:45Z  merged: 2025-02-05T02:46:26Z
- url: https://github.com/vllm-project/vllm/pull/12368
- labels: ready

## Body

This PR adds an environment variable `VLLM_LOGITS_PROCESSOR_THREADS`.

If set, this will cause VLLM to use a threadpool of the given size to multithread (across sequences in a batch) its calls to the provided logits processors.

This can increase GPU utilization and decrease ITL in cases where batches are large and where logits processors either (a) launch additional CUDA kernels or (b) do significant CPU-bound work while not holding the python GIL, or both.

## Comments

### github-actions @ 2025-01-23T20:42:56Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### akeshet @ 2025-01-23T20:44:09Z

Note: this PR is slightly different from the one I have already tested internally. I'd like to test it a bit before its truly ready for review, but I didn't see an option to create this PR as a draft.
---
### akeshet @ 2025-01-23T21:06:13Z

> Note: this PR is slightly different from the one I have already tested internally. I'd like to test it a bit before its truly ready for review, but I didn't see an option to create this PR as a draft.

My local tests look good, ready for review!
---
### russellb @ 2025-01-28T18:41:28Z

This makes sense to me. Do you happen to have any performance data you can share? Perhaps benchmarking with structured output and xgrammar? It would be nice to show the benefit concretely. There are some scripts that would help automate this -- benchmarks/benchmark_guided.py and benchmarks/benchmark_serving_guided.py.

Also, I wonder about just making this the default behavior instead of requiring it to be enabled via a tunable. Allowing the threadpool to be adjusted seems fine, but if the benefit is noticeable with one of our in-tree logits processors (xgrammar, in particular), then putting it on by default may make sense.
---
### akeshet @ 2025-01-28T18:55:09Z

> This makes sense to me. Do you happen to have any performance data you can share?

I can't share my raw numbers or traces, but I can say that we see a roughly 10% improvement in tok/s and req/s when using this feature in combination with our internal logits processor (and with request concurrency of ~50, and threadpool size 50).

> Also, I wonder about just making this the default behavior instead of requiring it to be enabled via a tunable. Allowing the threadpool to be adjusted seems fine, but if the benefit is noticeable with one of our in-tree logits processors (xgrammar, in particular), then putting it on by default may make sense.

I wanted to be conservative and not alter existing behavior. As is often the case with python multithreading, there are cases where I suspect this would hurt rather than help performance. For instance, to get the benefit from this PR with our internal logits processor logic I had to make some tweaks (a well placed cuda sync) to ensure that the logits processors in separate threads don't block eachother via cuda's memcpy lock.
---
### russellb @ 2025-01-28T19:39:54Z

> > This makes sense to me. Do you happen to have any performance data you can share?
> 
> I can't share my raw numbers or traces, but I can say that we see a roughly 10% improvement in tok/s and req/s when using this feature in combination with our internal logits processor (and with request concurrency of ~50, and threadpool size 50).
> 
> > Also, I wonder about just making this the default behavior instead of requiring it to be enabled via a tunable. Allowing the threadpool to be adjusted seems fine, but if the benefit is noticeable with one of our in-tree logits processors (xgrammar, in particular), then putting it on by default may make sense.
> 
> I wanted to be conservative and not alter existing behavior. As is often the case with python multithreading, there are cases where I suspect this would hurt rather than help performance. For instance, to get the benefit from this PR with our internal logits processor logic I had to make some tweaks (a well placed cuda sync) to ensure that the logits processors in separate threads don't block eachother via cuda's memcpy lock.

OK, thanks. If we're not able to show how it helps something in-tree, then I agree with having it off by default.

Can you expand on the doc text to add some guidance on when it would be useful?  Something like "useful when using custom logits processors that either (a) launch additional CUDA kernels or (b) do significant CPU-bound work while not holding the python GIL, or both."
---
### akeshet @ 2025-01-29T19:18:26Z

> Can you expand on the doc text to add some guidance on when it would be useful? Something like "useful when using custom logits processors that either (a) launch additional CUDA kernels or (b) do significant CPU-bound work while not holding the python GIL, or both."

Done.
---
### alejopaullier96 @ 2025-02-27T13:19:57Z

@akeshet is there a minimal example on how to use logits processor with this environment variable?
---
### akeshet @ 2025-02-28T00:15:44Z

> @akeshet is there a minimal example on how to use logits processor with this environment variable?

@alejopaullier96 The minimum example would be simply to `export VLLM_LOGS_PROCESSOR_THREADS=$VALUE` prior to launching vllm. But it will really depend on the details of yours logits processor you are using, as to whether this gives any performance benefit. I came to this optimization by examining `nsys` traces of our workload with a custom logits processor.
---