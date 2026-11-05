# PR #27306: [Feature]: Add corrupted request metric to V1 metrics system.

- field: scheduling  | area: v1/metrics
- author: atalhens
- state: MERGED  created: 2025-10-21T23:38:43Z  merged: 2025-11-05T21:45:29Z
- url: https://github.com/vllm-project/vllm/pull/27306
- labels: ready, v1

## Body

## 
Signed off: sneh.lata@nutanix.com
## Purpose

The purpose of this PR is to fix: https://github.com/vllm-project/vllm/issues/27301 and https://github.com/vllm-project/vllm/issues/27298. This PR implements a new Prometheus metric `vllm:num_requests_corrupted` to track and monitor corrupted requests in vLLM. The feature provides visibility into request corruption issues that can occur during inference, enabling better monitoring and debugging capabilities.

- New Prometheus gauge metric `vllm:num_requests_corrupted`
- Console logging of corrupted request counts per logging interval.
- Enabled via the env variable  `VLLM_COMPUTE_NANS_IN_LOGITS` flag for performance control, thus it won't impact any functionality unless enabled; the default value is set to False.
- Updated documentation and updated dashboard examples config as well.

**Performance Considerations:**
- Corrupted request counting is optional and disabled by default
- When enabled, it adds no computational overhead


## Test Plan
- Not needed as it is a by-default disabled metric.

## Test Result
- N/A

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

### mergify @ 2025-10-21T23:39:17Z

Documentation preview: https://vllm--27306.org.readthedocs.build/en/27306/
---
### mergify @ 2025-10-22T13:50:03Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @atalhens.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### atalhens @ 2025-10-24T12:22:55Z



> Thanks for all your work on this - it's pretty obvious that #18777 has very limited value without this
> 
> And ... I know the way `SchedulerStats.num_corrupted_reqs` is implemented as a "count of currently running requests that are corrupted" leads to using a `Gauge` ...
> 
> But I really think a `Counter` is a more natural use of a time-series metric for this information. Probably that would mean adding `is_output_corrupted` to `FinishedRequestStats` and removing `num_corrupted_reqs` from `SchedulerStats`?

Thanks for the suggestion. Due to the way SchedulerStats was implemented, it previously did make sense to have it as a gauge metric. But definitely for time series metrics like this, it makes more sense as a counter metric. In this case, I suggest the following changes that align with your thought of changing SchedulerStats and introducing a new field over the FinishedStats.

1. Added `is_corrupted` flag in FinishedStats with default as False.
2. Added the num_corrupted_request field in IterationStats, within which update_from_output () helps update the response of a request from EngineCoreOutput. However, currently, it doesn't have any context about the nans in output logits.
3. Thus, added a field `num_nans_in_logits` that gets populated from the already implemented requests.num_nans_in_logits.
4. Added a field in LoggingStatsLogger `num_corrupted_reqs` which gets populated from the iteration stats.
5. Added Prometheus metric `corrupted_requests`: Number of requests corrupted out of running requests.


cc: @markmc 


---
### mergify @ 2025-10-24T13:36:18Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @atalhens.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2025-10-24T18:42:22Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @atalhens.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### markmc @ 2025-10-28T14:51:56Z

One more minor thing ... this can be removed from the PR title:

> and Update v1 Metrics docs.

(the title will be used in the git log when the PR is merged)
---
### markmc @ 2025-10-28T18:00:21Z

> Can you further justify the need for `VLLM_COMPUTE_NANS_IN_LOGITS` env var and why can't this be enabled by default?

Looks like it was added in #18777 in response to [this comment](https://github.com/vllm-project/vllm/pull/18777#pullrequestreview-2941138025)

> I cannot find a place where we were doing CPU <> GPU syncs in counting NaNs here.

Yeah, looks like the original performance overhead concern wasn't very specific


---
### atalhens @ 2025-10-28T18:32:11Z

> > Can you further justify the need for `VLLM_COMPUTE_NANS_IN_LOGITS` env var and why can't this be enabled by default?
> 
> Looks like it was added in #18777 in response to [this comment](https://github.com/vllm-project/vllm/pull/18777#pullrequestreview-2941138025)
> 
> > I cannot find a place where we were doing CPU <> GPU syncs in counting NaNs here.
> 
> Yeah, looks like the original performance overhead concern wasn't very specific

Thanks for pointing out simon,
The env variable was kept to address the overhead concern raised in the previous PR mentioned by Marc above. This PR brought the implementation to compute `_get_nans_in_output_logits`.

About the GPU<>CPU sync to calculate the NaNs, it is called this way. 
`request.num_nans_in_logits = num_nans_in_logits[req_id]` is gets populated from 
 
 `num_nans_in_logits = model_runner_output.num_nans_in_logits` and the model runner output utilizes the https://github.com/vllm-project/vllm/blob/141e6a0505968f022cf3b33b543357518b856d86/vllm/v1/worker/gpu_model_runner.py#L3112
 In order to get nans, as it was gatekept here: https://github.com/vllm-project/vllm/blob/141e6a0505968f022cf3b33b543357518b856d86/vllm/v1/worker/gpu_model_runner.py#L2247 by the environment and disabled by default, I utilized the same variable just to stay consistent. 
 
 cc: @simon-mo , @markmc 
---
### simon-mo @ 2025-10-28T18:47:44Z

Can we quantify this for the users? 

When running model X on hardware Y, there's x% throughput regression when sending metrics requests every 1s... 

So users can understand the implication of turning this on. 
---
### atalhens @ 2025-11-04T15:33:45Z

Hi, I ran the VLLM performance benchmarks with and without the flag being enabled to see the performance impact on a 2 H100 hardware node. 
From my sight, I observed that enabling the flag adds an overhead of:
- About 1-2% in latency 
- Though surprisingly, there was not a very big impact on throughput. I observed less than 1% in most of the scenarios; for high-throughput scenarios, it was around 3-4%
T
This doc includes the benchmark result. - [Doc Link](https://docs.google.com/document/d/1Vh_PWsngago8P8wycUL7qxri-vn3VGRm0Z9sC1lZvqI/edit?usp=sharing)


## Throughput Tests

| Test name                  | Metric              | Native | With VLLM_COMPUTE_NANS_IN_LOGITS flag enabled | Change (%) |
|:---------------------------|:--------------------|-------:|----------------------------------------------:|-----------:|
| throughput_llama8B_tp1     | Tput (req/s)        | 22.7456 | 22.7126 | -0.14 |
| throughput_llama8B_tp1     | Tput (tok/s)        | 9803.83 | 9789.58 | -0.15 |
| throughput_llama8B_tp1     | Elapsed time (s)    | 8.79289 | 8.80569 | +0.15 |
| throughput_mixtral8x7B_tp2 | Tput (req/s)        | 6.5888 | 6.5919 | +0.05 |
| throughput_mixtral8x7B_tp2 | Tput (tok/s)        | 3250.45 | 3251.98 | +0.05 |
| throughput_mixtral8x7B_tp2 | Elapsed time (s)    | 30.3546 | 30.3403 | -0.05 |

---

In conclusion, I think this difference is significant enough to warrant flagging this feature.

Thanks!
Snehlata

cc: @markmc @simon-mo 
---