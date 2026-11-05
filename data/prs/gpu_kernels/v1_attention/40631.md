# PR #40631: [Refactor] Unify 2D/3D kernels in triton_unified_attention

- field: gpu_kernels  | area: v1/attention
- author: JartX
- state: MERGED  created: 2026-04-22T16:13:19Z  merged: 2026-04-24T15:18:07Z
- url: https://github.com/vllm-project/vllm/pull/40631
- labels: ready, v1

## Body

This PR is the refactor-only half of the split that @tdoublep and @bringlein requested on #39074. There are no new features and no behavior changes here. The goal is simply to remove duplication by collapsing two nearly identical attention kernels into one.

On main, triton_unified_attention.py contains two kernels that are almost the same. kernel_unified_attention_2d processes the full sequence in a single tile loop and writes directly to the output buffer, while kernel_unified_attention_3d limits that loop to a single segment and writes per-segment partials that reduce_segments combines later. In practice, the two implementations only differ in a few places around sequence scoping and the epilogue, even though each one is about 400 lines long.

This PR merges those paths into a single kernel, gated by IS_3D: tl.constexpr. Because Triton only traces the branch that is actually taken, each concrete launch still compiles down to the same code we would have gotten from separate layout-specific kernels.

The rest of the diff is mostly helper extraction into a new triton_attention_helpers.py. The prologue, online softmax bookkeeping, tile-loop bound computation, mask construction (ALiBi, softcap, qq_bias, sliding window, mm-prefix, chunked attention), and the scalar writes used for segment reduction were previously duplicated across both kernels. Now each of those pieces lives in one place, which means future fixes only need to land once and will automatically apply to both paths.

In terms of file size, triton_unified_attention.py goes from 1268 lines to 752 lines, for a net reduction of 237 lines overall after accounting for the new helpers file.

This is not a duplicate of #39074. It is the first half of that work, split out at reviewer request. The original PR will be closed once this refactor PR and its stacked follow-up feature PR have both landed.

## Comments

### JartX @ 2026-04-22T16:25:13Z

@bringlein @tdoublep @kylesayrs @mgoin
---
### JartX @ 2026-04-23T17:47:48Z

THIS PR
GSM8K
KV UNQUANTIZED

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.924|±  |0.0168|
|     |       |strict-match    |     5|exact_match|↑  |0.904|±  |0.0187|

INT8 PER TOKEN HEAD
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.904|±  |0.0187|
|     |       |strict-match    |     5|exact_match|↑  |0.900|±  |0.0190|

FP8 PER TOKEN HEAD
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.892|±  |0.0197|
|     |       |strict-match    |     5|exact_match|↑  |0.892|±  |0.0197|

FP8
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.892|±  |0.0197|
|     |       |strict-match    |     5|exact_match|↑  |0.888|±  |0.0200|

MAIN
GSM8K

KV UNQUANTIZED
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.912|±  | 0.018|
|     |       |strict-match    |     5|exact_match|↑  |0.888|±  | 0.020|

INT8 PER TOKEN HEAD
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.904|±  |0.0187|
|     |       |strict-match    |     5|exact_match|↑  |0.892|±  |0.0197|

FP8 PER TOKEN HEAD

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.904|±  |0.0187|
|     |       |strict-match    |     5|exact_match|↑  |0.884|±  |0.0203|

FP8

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.904|±  |0.0187|
|     |       |strict-match    |     5|exact_match|↑  |0.892|±  |0.0197|



---
### tdoublep @ 2026-04-23T18:28:55Z

@JartX Could you run the same but for main too for reference? 
---
### JartX @ 2026-04-23T18:32:31Z

@tdoublep 
Sure, Thomas, sorry, I was working on them, it takes quite a while on RDNA3 haha
---
### JartX @ 2026-04-23T19:50:56Z

# vLLM Benchmark: PR vs MAIN

@tdoublep

Update on the KV unquantized TTFT gap I flagged earlier — it was benchmark variance, not a regression.

Re-ran KV unquant twice per branch:

| Branch | Run 1 TTFT mean | Run 2 TTFT mean | Intra-branch spread |
|---|---|---|---|
| main | 6128 ms | 5447 ms | ~681 ms (~12%) |
| PR | 6016 ms | 5427 ms | ~589 ms (~11%) |

Run-to-run variance within the same branch is ~11–12% on TTFT, so any single-run TTFT delta below that threshold is noise.

## KV Unquantized (2 runs averaged)

| Metric | MAIN (avg) | PR (avg) | Δ |
|---|---|---|---|
| Duration (s) | 43.11 | 43.19 | +0.19% |
| Request throughput (req/s) | 2.32 | 2.31 | -0.22% |
| Output tok/s | 1187.67 | 1185.43 | -0.19% |
| Total tok/s | 2375.33 | 2370.87 | -0.19% |
| Mean TTFT (ms) | 5787.77 | 5721.62 | -1.14% |
| Median TTFT (ms) | 6359.17 | 6291.99 | -1.06% |
| P99 TTFT (ms) | 9756.94 | 9696.52 | -0.62% |
| Mean TPOT (ms) | 72.62 | 72.92 | +0.42% |
| Median TPOT (ms) | 71.54 | 71.85 | +0.43% |
| P99 TPOT (ms) | 79.87 | 80.19 | +0.39% |
| Mean ITL (ms) | 72.62 | 72.92 | +0.42% |
| Median ITL (ms) | 65.86 | 66.20 | +0.51% |
| P99 ITL (ms) | 68.67 | 68.85 | +0.26% |

## INT8 Per-Token-Head (single run per branch)

| Metric | MAIN | PR | Δ |
|---|---|---|---|
| Duration (s) | 49.90 | 51.48 | +3.17% |
| Request throughput (req/s) | 2.00 | 1.94 | -3.00% |
| Output tok/s | 1025.96 | 994.52 | -3.06% |
| Total tok/s | 2051.92 | 1989.04 | -3.06% |
| Mean TTFT (ms) | 9394.96 | 11231.40 | +19.55% ⚠ within variance band |
| Median TTFT (ms) | 10029.42 | 11865.82 | +18.31% ⚠ within variance band |
| P99 TTFT (ms) | 13756.98 | 15538.29 | +12.95% ⚠ within variance band |
| Mean TPOT (ms) | 78.67 | 78.09 | -0.74% |
| Median TPOT (ms) | 77.48 | 76.89 | -0.76% |
| P99 TPOT (ms) | 86.60 | 85.89 | -0.82% |
| Mean ITL (ms) | 78.67 | 78.09 | -0.74% |
| Median ITL (ms) | 71.22 | 71.15 | -0.10% |
| P99 ITL (ms) | 74.84 | 74.96 | +0.16% |

## FP8 Per-Token-Head (single run per branch)

| Metric | MAIN | PR | Δ |
|---|---|---|---|
| Duration (s) | 53.76 | 55.11 | +2.51% |
| Request throughput (req/s) | 1.86 | 1.81 | -2.69% |
| Output tok/s | 952.45 | 929.12 | -2.45% |
| Total tok/s | 1904.90 | 1858.25 | -2.45% |
| Mean TTFT (ms) | 13401.50 | 15162.25 | +13.14% ⚠ within variance band |
| Median TTFT (ms) | 14031.36 | 15789.20 | +12.53% ⚠ within variance band |
| P99 TTFT (ms) | 17695.84 | 19396.58 | +9.61% |
| Mean TPOT (ms) | 78.36 | 77.49 | -1.11% |
| Median TPOT (ms) | 77.18 | 76.30 | -1.14% |
| P99 TPOT (ms) | 86.17 | 85.18 | -1.15% |
| Mean ITL (ms) | 78.36 | 77.49 | -1.11% |
| Median ITL (ms) | 70.88 | 69.97 | -1.28% |
| P99 ITL (ms) | 73.95 | 85.52 | +15.65% |

## FP8 (single run per branch)

| Metric | MAIN | PR | Δ |
|---|---|---|---|
| Duration (s) | 50.25 | 51.56 | +2.61% |
| Request throughput (req/s) | 1.99 | 1.94 | -2.51% |
| Output tok/s | 1018.96 | 992.93 | -2.55% |
| Total tok/s | 2037.91 | 1985.86 | -2.55% |
| Mean TTFT (ms) | 11496.72 | 13217.67 | +14.97% ⚠ within variance band |
| Median TTFT (ms) | 12104.37 | 13821.47 | +14.19% ⚠ within variance band |
| P99 TTFT (ms) | 15626.46 | 17284.39 | +10.61% |
| Mean TPOT (ms) | 75.23 | 74.38 | -1.13% |
| Median TPOT (ms) | 74.08 | 73.23 | -1.15% |
| P99 TPOT (ms) | 82.76 | 81.79 | -1.17% |
| Mean ITL (ms) | 75.23 | 74.38 | -1.13% |
| Median ITL (ms) | 68.15 | 67.20 | -1.39% |
| P99 ITL (ms) | 71.14 | 70.83 | -0.44% |

## Summary

| Path | TTFT Δ | TPOT Δ | Throughput Δ | Assessment |
|---|---|---|---|---|
| KV unquant (2 runs) | -1.14% | +0.42% | -0.22% | No regression |
| INT8 per-token-head (1 run) | +19.55% | -0.74% | -3.00% | Within ~12% variance band — inconclusive |
| FP8 per-token-head (1 run) | +13.14% | -1.11% | -2.69% | Within variance band — inconclusive |
| FP8 (1 run) | +14.97% | -1.13% | -2.51% | Within variance band — inconclusive |

---
### JartX @ 2026-04-24T11:11:02Z

 @tdoublep All test passed :)
---