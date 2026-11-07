# PR #36162: [Mamba] Flashinfer selective_state_update

- field: gpu_kernels  | area: vllm/config
- author: roikoren755
- state: MERGED  created: 2026-03-05T17:45:28Z  merged: 2026-04-14T19:10:59Z
- url: https://github.com/vllm-project/vllm/pull/36162
- labels: ready, ci/build, v1

## Body


## Purpose

Add wrapper for FI's selective_state_update kernel, with a runtime dispatcher, connected to a config field, to select between it and the existing triton implementation.

As suggested by @tdoublep in #35753, I've introduced `MambaConfig` in this PR, and a followup (or this, if you'd prefer) could move config fields relevant to Mamba to it.

## Test Plan
New test file for the dispatcher's functionality.
Add tests e2e for Nemotron 3 Nano that use FI's kernel.
Benchmark the two options.

## Test Result
Tests pass, e2e showing the same GSM8K score for both backends for `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
Running `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` with the following command (replacing the `--mamba-backend` argument for the `flashinfer` measurement):
```
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --tensor-parallel-size 4 --max-model-len 8192 --mamba-backend triton \
    --trust-remote-code
```

And benchmarking with:
```
vllm bench serve \
    --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --num-prompts 500 --request-rate inf \
    --input-len 256 --output-len 256
```

Got the following results:
Triton:
```
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Benchmark duration (s):                  5.88
Total input tokens:                      128002
Total generated tokens:                  128000
Request throughput (req/s):              85.02
Output token throughput (tok/s):         21765.03
Peak output token throughput (tok/s):    31129.00
Peak concurrent requests:                500.00
Total token throughput (tok/s):          43530.41
---------------Time to First Token----------------
Mean TTFT (ms):                          1256.51
Median TTFT (ms):                        1233.83
P99 TTFT (ms):                           1770.09
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.77
Median TPOT (ms):                        17.90
P99 TPOT (ms):                           19.41
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.82
Median ITL (ms):                         15.93
P99 ITL (ms):                            69.00
==================================================
```

Flashinfer:
```
============ Serving Benchmark Result ============
Successful requests:                     500
Failed requests:                         0
Benchmark duration (s):                  5.64
Total input tokens:                      128002
Total generated tokens:                  128000
Request throughput (req/s):              88.60
Output token throughput (tok/s):         22682.25
Peak output token throughput (tok/s):    32938.00
Peak concurrent requests:                500.00
Total token throughput (tok/s):          45364.86
---------------Time to First Token----------------
Mean TTFT (ms):                          1183.95
Median TTFT (ms):                        1161.50
P99 TTFT (ms):                           1656.52
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.13
Median TPOT (ms):                        17.26
P99 TPOT (ms):                           18.60
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.18
Median ITL (ms):                         15.53
P99 ITL (ms):                            63.94
==================================================
```

Which comes up as ~4-5% speedup in most all metrics:
```
  ┌────────────────────────┬──────────────┬──────────────┬───────┐
  │         Metric         │    Triton    │  FlashInfer  │ Delta │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ Duration               │ 5.88s        │ 5.64s        │ -4.1% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ Request throughput     │ 85.02 req/s  │ 88.60 req/s  │ +4.2% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ Output throughput      │ 21,765 tok/s │ 22,682 tok/s │ +4.2% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ Peak output throughput │ 31,129 tok/s │ 32,938 tok/s │ +5.8% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ TPOT median            │ 17.90 ms     │ 17.26 ms     │ -3.6% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ TPOT P99               │ 19.41 ms     │ 18.60 ms     │ -4.2% │
  ├────────────────────────┼──────────────┼──────────────┼───────┤
  │ ITL P99                │ 69.00 ms     │ 63.94 ms     │ -7.3% │
  └────────────────────────┴──────────────┴──────────────┴───────┘
```
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

### mergify @ 2026-03-30T10:25:19Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @roikoren755.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### mergify @ 2026-04-10T15:58:22Z

This pull request has merge conflicts that must be resolved before it can be
merged. Please rebase the PR, @roikoren755.

https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork
---
### roikoren755 @ 2026-04-13T10:03:53Z

> Overall looks good!
> 
> I do have some comments though:
> 
> 1. I agree with @amirkl94 we should have the triton backend as default so we don't need to deal with `None`
> 2. I'm not sure the current dispatch mechanism to choose between triton/FlashInfer is the most fitting one. I understand the pros and cons, but worth thinking if we should use a pattern that is already used elsewhere for backend selection instead of introducing something new. Maybe the recently added "Custom Ops IR" is more fitting?
> 3. [most important] You mention in the description that the current FlashInfer impl lacks support for some features (specifically SpecDec). Yet, these are not reflected in the code. So if the user selects the flashinfer backend and runs with SpecDec, they will get ab opaque error from flashinfer (or worse - silent failures resulting in bad outputs). Maybe we should add some defensive runtime checks? Something like this:
> 
> ```python
>   if dst_state_batch_indices is not None and \
>      dst_state_batch_indices is not state_batch_indices:
>       raise NotImplementedError(
>           "FlashInfer SSU backend does not yet support separate "
>           "dst_state_batch_indices. Use --mamba-backend triton.")
> 
>   if num_accepted_tokens is not None:
>       raise NotImplementedError(
>           "FlashInfer SSU backend does not yet support "
>           "num_accepted_tokens (cache rewind). Use --mamba-backend triton.")
> ```

1. Dealing with None was actually simpler, but I'll fix that now.
2. I may be mistaken, but it looks like this doesn't quite fit, as there isn't (currently) a PyTorch native implementation, and the triton and flashinfer kernels support the same features, so it boils down to an implementation priority list, which I don't think is what we want to aim for...
3. That's a mistake, and I've deleted that part of the description. SpecDec (and all other features) is fully supported, and is covered in tests after this PR, so there's no need for the `NotImplementedError`s.
---
### roikoren755 @ 2026-04-15T08:30:19Z

> LGTM
> 
> We might want to revisit whether the dispatch could be better handled by vLLM IR at a later stage.

That was @tomeras91 's suggestion as well, but from what I saw there are a couple of issues there:
* We need a native PyTorch base implementation (#39157 will add it)
* The main differentiator in IR seems to be which args the different implementations support. Since the flashinfer and triton kernels have the same support, I'm not sure this is quite the right fit...
---
### tomeras91 @ 2026-04-15T09:32:17Z

> * The main differentiator in IR seems to be which args the different implementations support. Since the flashinfer and triton kernels have the same support, I'm not sure this is quite the right fit...

I think in #39262 IR is used also as a priority mechanism. Not exactly what we want here (control from cli arg), but also not only for different arg support for different implementations. 

---