# PR #6455: [Distributed][Model] Rank-based Component Creation for Pipeline Parallelism Memory Optimization

- field: parallelism  | area: model_executor/models
- author: wushidonguc
- state: MERGED  created: 2024-07-15T22:58:42Z  merged: 2024-07-17T02:20:26Z
- url: https://github.com/vllm-project/vllm/pull/6455
- labels: ready

## Body

Prior to this change, the Llama model initialization in Pipeline Parallel scenarios would create all components and layers for every rank, leading to unnecessary memory overhead. This pull request optimizes memory usage by creating certain components only on the relevant ranks.

The primary changes include:
1. Creating the embedding layer on the first rank only.
2. Creating the norm layer on the last rank only.
3. Creating the lm head, logits processor, and sampler components on the last rank only.

By selectively creating components on the relevant ranks, this optimization reduces the overall memory footprint. Testing with `--pipeline-parallel-size 8` showed up to 25% per-rank memory savings for the Llama-3-70B model. This memory optimization can potentially enable higher throughput by allowing more memory to be utilized for serving.

This optimization enables serving larger models and running inference on resource-constrained environments by leveraging Pipeline Parallelism more efficiently.

Testing:
- Benchmarks have been conducted to measure the memory savings and performance impact of this optimization.

Please review the changes and provide feedback or suggestions for further improvements.

## Comments

### github-actions @ 2024-07-15T22:58:53Z

👋 Hi! Thank you for contributing to the vLLM project.
 Just a reminder: PRs would not trigger full CI run by default. Instead, it would only trigger `fastcheck` CI to run, which consists only a small and essential subset of tests to quickly catch errors with the flexibility to run extra individual tests on top (you can do this by unblocking test steps in the Buildkite run). 

Full CI run is still required to merge this PR so once the PR is ready to go, please make sure to run it. If you need all test signals in between PR commits, you can trigger full CI as well.

 To run full CI, you can do one of these:
- Comment `/ready` on the PR
- Add `ready` label to the PR
- Enable auto-merge.

🚀
---
### sfc-gh-hazhang @ 2024-07-16T00:56:56Z

looks nice. I'll do a test on H100 setups and confirm improvement.
---
### youkaichao @ 2024-07-16T01:21:47Z

thanks for the contribution! the overall idea looks good to me.

we actually use the minimum of the number of blocks across all processes to determine the final block size, which means it is only effective if we can reduce the lower bound of model memory usage.

In this case, you:
- removed lm head for the first pp rank
- removed both word embedding and lm head for the rest ranks
- removed word embedding for the last rank

Overall, the minimum memory usage of the model is reduced by about the size of a word embedding or an lm head layer (typically the same size).

It would be better if you can move this logic into common layers, so that the rest models can directly benefit from it. I added `PPMissingLayer` in https://github.com/vllm-project/vllm/pull/6406 recently, which might help. My goal is to reduce the code change to support pp, to make sure the code is not too intrusive and people can find it easy to integrate.

One niche case, is you need to take care of models that use tie embeddings.
---
### andoorve @ 2024-07-16T01:29:51Z

Thanks for your change! This should bring some improvement. I'll let @sfc-gh-hazhang report back.
---
### wushidonguc @ 2024-07-16T20:46:54Z

> thanks for the contribution! the overall idea looks good to me.
> 
> we actually use the minimum of the number of blocks across all processes to determine the final block size, which means it is only effective if we can reduce the lower bound of model memory usage.
> 
> In this case, you:
> 
>     * removed lm head for the first pp rank
> 
>     * removed both word embedding and lm head for the rest ranks
> 
>     * removed word embedding for the last rank
> 
> 
> Overall, the minimum memory usage of the model is reduced by about the size of a word embedding or an lm head layer (typically the same size).
> 
> It would be better if you can move this logic into common layers, so that the rest models can directly benefit from it. I added `PPMissingLayer` in #6406 recently, which might help. My goal is to reduce the code change to support pp, to make sure the code is not too intrusive and people can find it easy to integrate.
> 
> One niche case, is you need to take care of models that use tie embeddings.

@youkaichao Thanks for the feedback and suggestions!

Instead of modifying `VocabParallelEmbedding` and `ParallelLMHead`, I propose keeping the pipeline parallelism logic within the model-specific classes. We can introduce helper methods like `get_rank_dependent_embedding` and `get_rank_dependent_lm_head` to encapsulate the rank-aware logic. These helpers would return the appropriate instances based on the current rank.

This way, `VocabParallelEmbedding` and `ParallelLMHead` remain generic and reusable across models. The pipeline parallelism logic stays in the model-specific classes, making it easier to maintain and extend.

Please let me know if this approach works for you.
---
### youkaichao @ 2024-07-16T20:56:00Z

> The pipeline parallelism logic stays in the model-specific classes, making it easier to maintain and extend.

this idea looks good to me. I'd like to make the code change as small as possible, and it would be better if we can make the code intuitive.
---
### wushidonguc @ 2024-07-16T22:46:43Z

@youkaichao I've rebased the branch on the latest `main` and integrated the `PPMissingLayer` component to handle weight skipping. No helper functions are added at this stage, but they can be introduced in a separate pull request if deemed necessary.

Please review the changes and provide any feedback or concerns you may have. I tested the integration of `PPMissingLayer` locally, and it seems to be working as expected.
---
### youkaichao @ 2024-07-17T00:58:36Z

/ready
---