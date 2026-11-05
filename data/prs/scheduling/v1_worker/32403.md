# PR #32403: [Performance] Improve Triton prefill attention kernel's performance 

- field: scheduling  | area: v1/worker
- author: Isotr0py
- state: CLOSED  created: 2026-01-15T09:05:20Z  merged: 2026-01-18T04:19:59Z
- url: https://github.com/vllm-project/vllm/pull/32403
- labels: ready, v1

## Body

<!-- markdownlint-disable -->

## Purpose
Refine prefill attention kernel with FA2 implementation based on https://github.com/ModelTC/LightLLM/blob/2fbd2b8a7b180570129bd539bb34cbe0a5dbb22a/lightllm/common/basemodel/triton_kernel/att/prefill_att/context_flashattention_nopad.py#L287-L288
- In previous implementation, we use `float("-inf")` in attention mask, which needs additional invalid values mask for SWA. This PR replaces it with extremely large values $-10^8$ to overflow.
- Pre-scale `sm_scale` with $\frac{1}{ln(2)}$, then we can fuse `triton.math.exp` with faster `triton.math.exp2`

## Test Plan
Benchmark with https://github.com/noooop/snippet/blob/main/benchmarks/triton_attention/v1_offline_triton.py

## Test Result
**Main branch**
```
Batchsize 8, Input_len 2048 Throughput: 33.9996 requests/s, 69631.2040 tokens/s, Latency 113.12 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 34.4612 requests/s, 70576.6036 tokens/s, Latency 207.27 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 34.8346 requests/s, 71341.2651 tokens/s, Latency 358.84 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 34.8565 requests/s, 71386.2010 tokens/s, Latency 478.15 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 34.7642 requests/s, 71197.0056 tokens/s, Latency 719.13 ms, n_step 4
```

**Main branch + exp2 fusion**
```
Batchsize 8, Input_len 2048 Throughput: 34.4454 requests/s, 70544.1212 tokens/s, Latency 111.66 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 34.7209 requests/s, 71108.4363 tokens/s, Latency 205.72 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 35.1655 requests/s, 72018.9614 tokens/s, Latency 355.46 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 35.2775 requests/s, 72248.3425 tokens/s, Latency 472.44 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 34.9167 requests/s, 71509.3940 tokens/s, Latency 715.99 ms, n_step 4
```

**PR**
```
Batchsize 8, Input_len 2048 Throughput: 34.9690 requests/s, 71616.4340 tokens/s, Latency 109.99 ms, n_step 26
Batchsize 16, Input_len 2048 Throughput: 35.0962 requests/s, 71876.9206 tokens/s, Latency 203.52 ms, n_step 14
Batchsize 32, Input_len 2048 Throughput: 35.5592 requests/s, 72825.2077 tokens/s, Latency 351.53 ms, n_step 8
Batchsize 64, Input_len 2048 Throughput: 35.5579 requests/s, 72822.6684 tokens/s, Latency 468.72 ms, n_step 6
Batchsize 128, Input_len 2048 Throughput: 35.2918 requests/s, 72277.6372 tokens/s, Latency 708.38 ms, n_step 4
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
