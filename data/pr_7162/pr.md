# PR #7162: [Performance] Optimize e2e overheads: Reduce python allocations

- field: scheduling  | area: attention/backends
- author: alexm-redhat
- state: MERGED  created: 2024-08-05T14:54:55Z  merged: 2024-08-09T04:34:28Z
- url: https://github.com/vllm-project/vllm/pull/7162
- labels: ready

## Body

This PR introduces a bunch of end-to-end overhead optimizations to reduce python object allocations/deallocations over scheduler iterations. In particular:
1. Avoid python object allocations for "InterDataForSeqGroup" objects: These objects have lot of fields, and most of the them were allocated dynamically. In this PR, these objects are pre-allocated and reused between runs. The pre-allocation is done per "number of sequences per group" (to support beam search), where most of the time we simply have one sequence per group.
2. Avoid python object allocations for "SequenceGroupMetadata" inside scheduler, in the same way as for (1).
3. Avoid python object allocations for "SequenceGroupToSample" inside SamplingMetadata.prepare(..)
4. Add an optimization to scheduler to be aware of "one sequence per group" case (since most of them we do not run beam search).
5. A bunch of misc small stuff to reduce object allocations

End-to-end throughput of Llama3 8B on 1xH100 is 24% faster with this PR. Command used:
`python3 benchmark_throughput.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --backend vllm --input-len 512 --output-len 256 --num-prompts 1000 --tensor-parallel 1 `

Main branch from 08/05/2024:
`Throughput: 19.48 requests/s, 14962.18 tokens/s`
This PR:
`Throughput: 24.32 requests/s, 18123.51 tokens/s`



## Comments

### github-actions @ 2024-08-05T14:55:06Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only run `fastcheck` CI which consists a small and essential subset of CI tests to quickly catch errors. You can run other CI tests on top of default ones by unblocking the steps in your `fast-check` build on Buildkite UI. 

Once the PR is approved and ready to go, please make sure to run full CI as it is required to merge (or just use auto-merge).

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### comaniac @ 2024-08-05T15:19:30Z

This is one way to introduce incremental prepare input. Will review soon.

cc @youkaichao @zhuohan123
---
### comaniac @ 2024-08-05T16:23:35Z

I took a brief look and have the following impressions:
1. This optimization applies to continuous decoding steps. In other words, if requests in the batch are added or removed, the cache is invalid.
2. This optimization doesn't work with chunked prefill as it will schedule prefill and decode requests in the same batch.

Please correct me if I misunderstood anything. IIUC, It seems to me that this optimization should be done with multi-step worker, as it focuses exactly on the scenarios listed above. For general cases, this optimization introduces decoding specific branches (e.g., `if no prefill tokens`) and we should avoid that as possible. 




---
### alexm-redhat @ 2024-08-05T16:52:52Z

@comaniac and I discussed the relation to multi-step. In general, a micro-benchmark shows that the benefit from flash-attn decode optimization in this PR is around 2-3%, and most of the benefit comes from python object allocation reductions. There are two possible ways to go about this:
1. Remove flash-attn optimization from this PR in favor of multi-step (since multi-step should cover flash-attn decode)
2. Proceed with flash-attn optimization until multi-step is fully enabled.
   
---
### alexm-redhat @ 2024-08-05T16:53:29Z

/ready
---
### njhill @ 2024-08-05T18:26:58Z

Thanks @alexm-neuralmagic this is great!

> 
> 1. This optimization applies to continuous decoding steps. In other words, if requests in the batch are added or removed, the cache is invalid.

I think (1) isn't a concern since the proportion of steps in which reqs are added or removed is typically small.

> 2. This optimization doesn't work with chunked prefill as it will schedule prefill and decode requests in the same batch.
> 

I didn't look closely enough yet but it presumably it will still work once in this mode once the prefill tokens have been exhausted i.e. once the batch returns to decode-only state?

> 1. Remove flash-attn optimization from this PR in favor of multi-step (since multi-step should cover flash-attn decode)
> 2. Proceed with flash-attn optimization until multi-step is fully enabled.

Given my comments above I think both parts of this would still make sense to include, but perhaps separate into two separate PRs anyhow?
---
### comaniac @ 2024-08-05T18:47:42Z

> Thanks @alexm-neuralmagic this is great!
> 
> > 1. This optimization applies to continuous decoding steps. In other words, if requests in the batch are added or removed, the cache is invalid.
> 
> I think (1) isn't a concern since the proportion of steps in which reqs are added or removed is typically small.

Isn't we continue adding/removing decoding requests at every step as long as there are new requests coming?

> > 2. This optimization doesn't work with chunked prefill as it will schedule prefill and decode requests in the same batch.
> 
> I didn't look closely enough yet but it presumably it will still work once in this mode once the prefill tokens have been exhausted i.e. once the batch returns to decode-only state?

I'm not sure if prefill requests exhausting is a reasonable assumption especially for high QPS.

In summary I feel your assumptions make sense in offline batching, but I'm not sure about online serving. Meanwhile, considering the code complexity this PR will introduce, I personally would prefer to isolate it to the scenario it fits into. Of course we could still take the general Python code optimization.

---
### njhill @ 2024-08-05T19:00:16Z

> > Thanks @alexm-neuralmagic this is great!
> > > 1. This optimization applies to continuous decoding steps. In other words, if requests in the batch are added or removed, the cache is invalid.
> > 
> > 
> > I think (1) isn't a concern since the proportion of steps in which reqs are added or removed is typically small.
> 
> Isn't we continue adding/removing decoding requests at every step as long as there are new requests coming?
>

Offline is getting more interest lately with SDG etc. It obviously depends on the particular workload in terms of input/output split sizes and request patterns but I think even in the online serving case the batch constituency doesn't change for a significant proportion of steps. We should probably add metrics for this if we don't have already :)

> > > 2. This optimization doesn't work with chunked prefill as it will schedule prefill and decode requests in the same batch.
> > 
> > 
> > I didn't look closely enough yet but it presumably it will still work once in this mode once the prefill tokens have been exhausted i.e. once the batch returns to decode-only state?
> 
> I'm not sure if prefill requests exhausting is a reasonable assumption especially for high QPS.

Again I think this is very workload dependent, what about for "Write me an essay about x" type use cases.

> In summary I feel your assumptions make sense in offline batching, but I'm not sure about online serving. Meanwhile, considering the code complexity this PR will introduce, I personally would prefer to isolate it to the scenario it fits into. Of course we could still take the general Python code optimization.

I agree about taking care to minimze additional complexity.
---
### alexm-redhat @ 2024-08-05T19:01:30Z

@njhill @comaniac thanks for the review and comments. @njhill I think it is a good idea to separate the optimizations. I will remove the flash-attn from this PR and will focus on python optimizations first, to avoid a conflict with mutli-step worker PR from Will.
---
### comaniac @ 2024-08-07T16:10:11Z

@youkaichao PTAL and let's try to merge this PR by today or tomorrow.
---
### alexm-redhat @ 2024-08-08T16:22:26Z

working on fixing tests
---
### alexm-redhat @ 2024-08-09T01:49:07Z

fixed all of the failing tests, should be green now I hope.
---
### njhill @ 2024-08-09T05:17:21Z

Thanks @alexm-neuralmagic for this, it's a huge speedup!

It looks like there's a still couple of unaddressed comments though? https://github.com/vllm-project/vllm/pull/7162#discussion_r1710649611 and https://github.com/vllm-project/vllm/pull/7162#discussion_r1707510994
---
### alexm-redhat @ 2024-08-09T13:51:31Z

@njhill no problem, to address the leftover comments I have posted a small followup: https://github.com/vllm-project/vllm/pull/7364
---