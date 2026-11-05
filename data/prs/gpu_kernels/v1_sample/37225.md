# PR #37225: [Perf] Optimize top-k search in apply_top_k_top_p_triton sampler

- field: decoding  | area: v1/sample
- author: mgoin
- state: MERGED  created: 2026-03-16T20:36:45Z  merged: 2026-03-17T18:35:18Z
- url: https://github.com/vllm-project/vllm/pull/37225
- labels: performance, ready, v1

## Body

<!-- markdownlint-disable -->

## Summary
- Fix `min_larger` in top-k ternary search to compute min of values above the pivot instead of the global min. This makes the termination condition converge correctly (~3-5 iterations instead of hitting the 18-iteration cap).
- Fuse the `num_min_larger` second pass into the first pass for the top-k paths, eliminating a redundant full data scan per search iteration.

## Benchmark results (benchmark_topk_topp.py)
- **top-k only: 17–28% faster** across all batch sizes and vocab sizes
- **top-k + top-p combined: 16–25% faster**
- **top-p only: unchanged** (not touched)

| Scenario | Batch | Vocab | Before (ms) | After (ms) | Δ |
|---|---|---|---|---|---|
| topk_whole | 128 | 32k | 0.071 | 0.056 | -21% |
| topk_whole | 2048 | 32k | 0.761 | 0.550 | -28% |
| topk_whole | 2048 | 128k | 2.312 | 1.891 | -18% |
| topk_topp_whole | 2048 | 32k | 1.211 | 0.907 | -25% |
| topk_topp_whole | 2048 | 128k | 3.206 | 2.652 | -17% |

## Test plan
- [x] `pytest tests/v1/sample/test_topk_topp_sampler.py -v` passes
- [x] `benchmarks/benchmark_topk_topp.py` shows no regressions

Co-authored-by: Claude <noreply@anthropic.com>

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

### mgoin @ 2026-03-16T20:56:01Z

PTAL @cakeng @njhill 
---
### cakeng @ 2026-03-16T21:17:51Z

@mgoin Hi, thank you for fixing the bug! I am sorry that there was such an obvious bug in the Top-K code. I think I made some mistakes while refactoring the code for the PR. The optimization looks good, I didn't realize that you could update the min value and the count in a single pass.
---
### mgoin @ 2026-03-16T21:53:08Z

@cakeng no worries at all, thanks for the great implementation! Glad this change makes sense to you too. I think there is more we could do, especially for small batch sizes due to no batch-level parallelism and reducing a bit more code duplication, but I don't think it matters so much.
---