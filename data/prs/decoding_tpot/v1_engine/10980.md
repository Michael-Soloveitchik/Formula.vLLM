# PR #10980: [V1] V1 engine implements parallel sampling (AsyncLLM and LLMEngine)

- field: scheduling  | area: v1/engine
- author: afeldman-nm
- state: MERGED  created: 2024-12-07T17:49:54Z  merged: 2025-02-24T16:29:41Z
- url: https://github.com/vllm-project/vllm/pull/10980
- labels: documentation, frontend, ready, ci/build, v1

## Body

This PR adds support for parallel sampling to v1 AsyncLLM and LLMEngine.

Parallel sampling is implemented outside the engine. This does not impact the vLLM v0 parallel sampling implementation. 

Design doc: https://docs.google.com/document/d/1_fvbHVCfexkPAj2Vx0Q0gNvE-b53WwtrLrD6NldgBFU/edit?usp=sharing (message me if you need access permissions)

A request with `n>1` will spawn `n` requests with `n=1` and aggregate their outputs in accordance with the `output_kind`. ~~If prefix caching is enabled, an initial warmup request with `max_tokens=1` will be sent to the engine to fill the prefix cache.~~ **Update:** The vLLM v1 engine can exploit APC when a prompt repeats within a batch, even if that prompt was not seen in a previous batch. Therefore, no warmup request is required.

The abstractions are cleanest for async v1 engine because AsyncLLM presents a `generate()` method that handles adding requests and running the engine; parallel sampling is implemented by writing a wrapper which invokes this method `n` times against parallel sampling child requests.

v1 LLMEngine presents `add_request()` and `step()` methods, so parallel sampling is implemented by writing an `add_request()` wrapper which branches parallel sampling requests into `n` `add_request()` calls for child requests, and then having `step()` aggregate child request outputs into a parent request output.

FIX #13419 
FIX #13420

## Comments

### github-actions @ 2024-12-07T17:50:05Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which starts running only a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of those by going to your `fastcheck` build on Buildkite UI (linked in the PR checks section) and unblock them. If you do not have permission to unblock, ping `simon-mo` or `khluu` to add you in our Buildkite org. 

Once the PR is approved and ready to go, your PR reviewer(s) can run CI to test the changes comprehensively before merging.

 To run CI, PR reviewers can do one of these:
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### mergify @ 2024-12-13T15:47:04Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @afeldman-nm.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### FrederickVu @ 2025-01-30T23:57:27Z

It seems like this PR is implementing ideas similar to those implemented in PR #9302 for the V0 engine. That PR created some issues that were addressed in PR #11898 and which may exist in the proposed V1 code. 

In particular, the proposed code currently does not properly handle the case when a seed value is provided for the parent request; the seed value is duplicated in child requests, leading to identical outputs in the child requests. The fix in #11898 was simply to move the copying of the `SamplingParams` object inside a for-loop and to increment the seed value of the parent request. 

Additionally, the proposed code for the V1 engine defines `get_num_unfinished_requests()` in a way that is currently incompatible with the same function for the V0 engine (though it's really the V0 engine's code which is in the wrong, I am choosing to mention this here as it seems relevant and the PR is still open). In the V0 engine, that function actually counts the number of `SequenceGroup` objects in the engine as opposed to the number of requests. This led to bugs mentioned in Issues #10949 and #11519, and an attempt to fix them is in PR #12428. However, the proposed fix is not compatible with the proposed V1 code. I will comment on that PR as well so that maintainers can adapt the V0 code to align with the proposed V1 code. 

---
### mergify @ 2025-02-05T06:44:49Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @afeldman-nm.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### afeldman-nm @ 2025-02-11T15:28:21Z

> It seems like this PR is implementing ideas similar to those implemented in PR #9302 for the V0 engine. That PR created some issues that were addressed in PR #11898 and which may exist in the proposed V1 code.
> 
> In particular, the proposed code currently does not properly handle the case when a seed value is provided for the parent request; the seed value is duplicated in child requests, leading to identical outputs in the child requests. The fix in #11898 was simply to move the copying of the `SamplingParams` object inside a for-loop and to increment the seed value of the parent request.
> 
> Additionally, the proposed code for the V1 engine defines `get_num_unfinished_requests()` in a way that is currently incompatible with the same function for the V0 engine (though it's really the V0 engine's code which is in the wrong, I am choosing to mention this here as it seems relevant and the PR is still open). In the V0 engine, that function actually counts the number of `SequenceGroup` objects in the engine as opposed to the number of requests. This led to bugs mentioned in Issues #10949 and #11519, and an attempt to fix them is in PR #12428. However, the proposed fix is not compatible with the proposed V1 code. I will comment on that PR as well so that maintainers can adapt the V0 code to align with the proposed V1 code.

Hi @FrederickVu , thank you for this feedback. I re-implemented this PR from scratch & ensured that if a non-`None` `seed` value is provided to `sampling_params`, then each child request has a different integer seed value (basically what you described - increment `seed` in a loop.)

As for `get_num_unfinished_requests`, the new implementation of this PR does not rely on this function.
---
### WoosukKwon @ 2025-02-13T08:53:27Z

@afeldman-nm

> If prefix caching is enabled, an initial warmup request with max_tokens=1 will be sent to the engine to fill the prefix cache.

Why do we need this? V1's prefix caching implementation can deduplicate the prompts even if they are scheduled in the same batch without the warmup.
---
### m-harmonic @ 2025-02-13T21:12:43Z

> > It seems like this PR is implementing ideas similar to those implemented in PR #9302 for the V0 engine. That PR created some issues that were addressed in PR #11898 and which may exist in the proposed V1 code.
> > In particular, the proposed code currently does not properly handle the case when a seed value is provided for the parent request; the seed value is duplicated in child requests, leading to identical outputs in the child requests. The fix in #11898 was simply to move the copying of the `SamplingParams` object inside a for-loop and to increment the seed value of the parent request.
> > Additionally, the proposed code for the V1 engine defines `get_num_unfinished_requests()` in a way that is currently incompatible with the same function for the V0 engine (though it's really the V0 engine's code which is in the wrong, I am choosing to mention this here as it seems relevant and the PR is still open). In the V0 engine, that function actually counts the number of `SequenceGroup` objects in the engine as opposed to the number of requests. This led to bugs mentioned in Issues #10949 and #11519, and an attempt to fix them is in PR #12428. However, the proposed fix is not compatible with the proposed V1 code. I will comment on that PR as well so that maintainers can adapt the V0 code to align with the proposed V1 code.
> 
> Hi @FrederickVu , thank you for this feedback. I re-implemented this PR from scratch & ensured that if a non-`None` `seed` value is provided to `sampling_params`, then each child request has a different integer seed value (basically what you described - increment `seed` in a loop.)
> 
> As for `get_num_unfinished_requests`, the new implementation of this PR does not rely on this function.

Thanks for working on this. I haven't had a chance to look into the specifics of the new implementation but also wanted to ask about n>1 behavior as this PR (https://github.com/vllm-project/vllm/pull/9302) reduced our throughput by about 3x. It seems that sequence forking was extremely important in the v0 engine. Is an equivalent strategy going to be used in v1? We're now waiting for n>1 support in the v1 engine before we can upgrade past 0.6.3. Thanks
---
### WoosukKwon @ 2025-02-13T22:19:06Z

@m-harmonic

> Thanks for working on this. I haven't had a chance to look into the specifics of the new implementation but also wanted to ask about n>1 behavior as this PR (https://github.com/vllm-project/vllm/pull/9302) reduced our throughput by about 3x. It seems that sequence forking was extremely important in the v0 engine. Is an equivalent strategy going to be used in v1? We're now waiting for n>1 support in the v1 engine before we can upgrade past 0.6.3. Thanks

Thanks for the feedback. I think this is because you didn't add `--enable-prefix-caching`. Starting with the PR you mentioned, vLLM relies on prefix caching to deduplicate the prompts in parallel sampling. As prefix caching is not enabled by default in V0, you may see slowdown if you didn't add `--enable-prefix-caching`. With V1, prefix caching is enabled by default, so you should be able to enjoy the speedup without any extra config.
---
### m-harmonic @ 2025-02-13T22:41:19Z

> @m-harmonic
> 
> > Thanks for working on this. I haven't had a chance to look into the specifics of the new implementation but also wanted to ask about n>1 behavior as this PR (#9302) reduced our throughput by about 3x. It seems that sequence forking was extremely important in the v0 engine. Is an equivalent strategy going to be used in v1? We're now waiting for n>1 support in the v1 engine before we can upgrade past 0.6.3. Thanks
> 
> Thanks for the feedback. I think this is because you didn't add `--enable-prefix-caching`. Starting with the PR you mentioned, vLLM relies on prefix caching to deduplicate the prompts in parallel sampling. As prefix caching is not enabled by default in V0, you may see slowdown if you didn't add `--enable-prefix-caching`. With V1, prefix caching is enabled by default, so you should be able to enjoy the speedup without any extra config.

Hi thanks for the reply. We did pass `enable_prefix_caching=True` in the engine arguments, which did not affect the throughput. Is there something else we should be doing?

Edit: I got a notification for a comment by @ afeldman-nm but oddly enough it's not showing up now. Per your question, we see the decreased throughput in VLLM's log stats. The number of generated tokens / sec falls from roughly 7k/sec to 2k/sec. In contrast, the prompt throughput rises by about 4-5x. This is for a value of n between 50 and 100. We did have prefix caching enabled in the engine args.
---
### afeldman-nm @ 2025-02-14T12:11:13Z

> > @m-harmonic
> > > Thanks for working on this. I haven't had a chance to look into the specifics of the new implementation but also wanted to ask about n>1 behavior as this PR (#9302) reduced our throughput by about 3x. It seems that sequence forking was extremely important in the v0 engine. Is an equivalent strategy going to be used in v1? We're now waiting for n>1 support in the v1 engine before we can upgrade past 0.6.3. Thanks
> > 
> > 
> > Thanks for the feedback. I think this is because you didn't add `--enable-prefix-caching`. Starting with the PR you mentioned, vLLM relies on prefix caching to deduplicate the prompts in parallel sampling. As prefix caching is not enabled by default in V0, you may see slowdown if you didn't add `--enable-prefix-caching`. With V1, prefix caching is enabled by default, so you should be able to enjoy the speedup without any extra config.
> 
> Hi thanks for the reply. We did pass `enable_prefix_caching=True` in the engine arguments, which did not affect the throughput. Is there something else we should be doing?
> 
> Edit: I got a notification for a comment by @ afeldman-nm but oddly enough it's not showing up now. Per your question, we see the decreased throughput in VLLM's log stats. The number of generated tokens / sec falls from roughly 7k/sec to 2k/sec. In contrast, the prompt throughput rises by about 4-5x. This is for a value of n between 50 and 100. We did have prefix caching enabled in the engine args.

Hi @m-harmonic , sorry I deleted my response because I thought @WoosukKwon 's suggestion to use prefix caching was more relevant, although it appears that that did not resolve the issue. Although, based on the metrics you shared, it sounds like prefix caching is working but the decode throughput is the issue.
---
### m-harmonic @ 2025-02-14T17:56:42Z

> > > @m-harmonic
> > > > Thanks for working on this. I haven't had a chance to look into the specifics of the new implementation but also wanted to ask about n>1 behavior as this PR (#9302) reduced our throughput by about 3x. It seems that sequence forking was extremely important in the v0 engine. Is an equivalent strategy going to be used in v1? We're now waiting for n>1 support in the v1 engine before we can upgrade past 0.6.3. Thanks
> > > 
> > > 
> > > Thanks for the feedback. I think this is because you didn't add `--enable-prefix-caching`. Starting with the PR you mentioned, vLLM relies on prefix caching to deduplicate the prompts in parallel sampling. As prefix caching is not enabled by default in V0, you may see slowdown if you didn't add `--enable-prefix-caching`. With V1, prefix caching is enabled by default, so you should be able to enjoy the speedup without any extra config.
> > 
> > 
> > Hi thanks for the reply. We did pass `enable_prefix_caching=True` in the engine arguments, which did not affect the throughput. Is there something else we should be doing?
> > Edit: I got a notification for a comment by @ afeldman-nm but oddly enough it's not showing up now. Per your question, we see the decreased throughput in VLLM's log stats. The number of generated tokens / sec falls from roughly 7k/sec to 2k/sec. In contrast, the prompt throughput rises by about 4-5x. This is for a value of n between 50 and 100. We did have prefix caching enabled in the engine args.
> 
> Hi @m-harmonic , sorry I deleted my response because I thought @WoosukKwon 's suggestion to use prefix caching was more relevant, although it appears that that did not resolve the issue. Although, based on the metrics you shared, it sounds like prefix caching is working but the decode throughput is the issue.

Thanks @afeldman-nm, appreciate the reply. One thing I'm wondering is whether matching n>1 throughput performance pre-0.6.4 is a priority for this change or follow-up work. It seems like fundamentally this new approach confers lower throughput. Relatedly, is there an architectural reason why this new approach is necessary in v1 or is it possible forking could be added later on if it makes a big difference?
---
### afeldman-nm @ 2025-02-18T21:38:15Z

Merged #13421 (LLMEngine support for parallel sampling) into this PR.
---
### WoosukKwon @ 2025-02-21T19:02:34Z

@afeldman-nm could you please check the failed tests and fix or re-run them?
---