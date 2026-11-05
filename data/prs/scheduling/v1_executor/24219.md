# PR #24219: [Core] Support async scheduling with uniproc executor 

- field: scheduling  | area: v1/executor
- author: njhill
- state: MERGED  created: 2025-09-04T05:37:50Z  merged: 2025-09-12T23:34:29Z
- url: https://github.com/vllm-project/vllm/pull/24219
- labels: ready, v1

## Body

Follow-on from https://github.com/vllm-project/vllm/pull/23569.

This provides most of the speedup of that PR, like +22% rather than +25%. We could experiment with a slightly more complicated version where the worker runs in a separate thread, but this seems like a good first implementation.

## Comments

### weijinqian0 @ 2025-09-05T01:32:10Z

In RL training scenarios, inference groups are typically managed externally, so 'external launcher' is needed. 
---
### Ronald1995 @ 2025-09-05T09:11:39Z

> This PR is based on top of #23569.
> 
> This provides most of the speedup of that PR, like +22% rather than +25%. We could experiment with a slightly more complicated version where the worker runs in a separate thread, but it seems unlikely it would exceed the MP performance anyhow given that the uniproc executor now appears to be no faster than the mulitproc one in general.

Implement async_scheduling in uniproc executor  not because it's faster than multiproc, but because we need to use uniproc(external launcher method) in RL training.
---
### njhill @ 2025-09-05T13:46:03Z

@Ronald1995 @weijinqian0 I pushed another commit to also support external launcher executor, maybe you could try it out?
---
### Ronald1995 @ 2025-09-06T02:32:14Z

> @Ronald1995 @weijinqian0 I pushed another commit to also support external launcher executor, maybe you could try it out?

@njhill  thanks for implementation of exeternal launcher method. i have tested it in my local environment base on your branch, and make some little bugfix, both the performance and precision are validated, would you please just cherry-pick my  [bugfix commit](https://github.com/Ronald1995/vllm/commit/cea2ff462c272237b1a2bb6a656ab724001447b7).
---
### njhill @ 2025-09-06T03:11:01Z

> > @Ronald1995 @weijinqian0 I pushed another commit to also support external launcher executor, maybe you could try it out?
> 
> @njhill thanks for implementation of exeternal launcher method. i have tested it in my local environment base on your branch, and make some little bugfix, both the performance and precision are validated, would you please just cherry-pick my [bugfix commit](https://github.com/Ronald1995/vllm/commit/cea2ff462c272237b1a2bb6a656ab724001447b7).

Thanks @Ronald1995. From your commit:
```python
        if isinstance(outputs, Exception):
            logger.error("EngineCore step failed with error: %s", outputs)
            raise outputs
```
I don't think step_fn will ever return an exception, do you agree or could you show me why you think this is needed?
---
### Ronald1995 @ 2025-09-06T03:39:46Z

> > > @Ronald1995 @weijinqian0 I pushed another commit to also support external launcher executor, maybe you could try it out?
> > 
> > 
> > @njhill thanks for implementation of exeternal launcher method. i have tested it in my local environment base on your branch, and make some little bugfix, both the performance and precision are validated, would you please just cherry-pick my [bugfix commit](https://github.com/Ronald1995/vllm/commit/cea2ff462c272237b1a2bb6a656ab724001447b7).
> 
> Thanks @Ronald1995. From your commit:
> 
> ```python
>         if isinstance(outputs, Exception):
>             logger.error("EngineCore step failed with error: %s", outputs)
>             raise outputs
> ```
> 
> I don't think step_fn will ever return an exception, do you agree or could you show me why you think this is needed?

1.

> > > @Ronald1995 @weijinqian0 I pushed another commit to also support external launcher executor, maybe you could try it out?
> > 
> > 
> > @njhill thanks for implementation of exeternal launcher method. i have tested it in my local environment base on your branch, and make some little bugfix, both the performance and precision are validated, would you please just cherry-pick my [bugfix commit](https://github.com/Ronald1995/vllm/commit/cea2ff462c272237b1a2bb6a656ab724001447b7).
> 
> Thanks @Ronald1995. From your commit:
> 
> ```python
>         if isinstance(outputs, Exception):
>             logger.error("EngineCore step failed with error: %s", outputs)
>             raise outputs
> ```
> 
> I don't think step_fn will ever return an exception, do you agree or could you show me why you think this is needed?

i just see get_output method in SyncMPClient handle the exception. but i think you are right, the execute_model_with_error_logging will catch the exception and just raise it, output will not be exception
---
### njhill @ 2025-09-08T17:53:10Z

@Ronald1995 any chance you could try out the latest version of this PR with your use case?

I think this is ready apart from extra CI test coverage.
---
### Ronald1995 @ 2025-09-09T13:01:55Z

> @Ronald1995 any chance you could try out the latest version of this PR with your use case?
> 
> I think this is ready apart from extra CI test coverage.


OK，I will test the latest version in my local environment. because the GPU resources is occupied by my colleague, i can't test it right now, i will complete this test until tomorrow morning  and report the results for you.
---
### Ronald1995 @ 2025-09-10T01:15:22Z

> > @Ronald1995 any chance you could try out the latest version of this PR with your use case?
> > I think this is ready apart from extra CI test coverage.
> 
> OK，I will test the latest version in my local environment. because the GPU resources is occupied by my colleague, i can't test it right now, i will complete this test until tomorrow morning and report the results for you.

@njhill i have tested the latest version with external_launcher scenario, both performance and precision meet expectation.
---
### njhill @ 2025-09-12T00:59:04Z

Asked here about the bc-lint failure: https://github.com/vllm-project/vllm/pull/21234#issuecomment-3283127147
---
### mergify @ 2025-09-12T17:39:18Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @njhill.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---