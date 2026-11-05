# PR #38990: [Bugfix][MoE] Fix 6-8% decode regression: prefer multi-stream shared expert overlap

- field: parallelism  | area: model_executor/layers
- author: voipmonitor
- state: MERGED  created: 2026-04-04T17:59:02Z  merged: 2026-04-05T14:28:32Z
- url: https://github.com/vllm-project/vllm/pull/38990
- labels: bug, ready, ready-run-all-tests

## Body

** NOTE (rob): this regression does not impact any official vLLM release **

## Summary

Fix **6-8% decode throughput regression** on MoE models with TP-only configurations (no EP/EPLB), introduced by #35153.

### Root cause

In `SharedExperts._determine_shared_experts_order()`, the `_has_external_experts` check is evaluated **before** the aux-stream overlap check. For TP-only configs, `_has_external_experts` returns `True` (neither EPLB nor flashinfer two-sided kernels are active), so `EXTERNAL` is selected unconditionally. The `MULTI_STREAM_OVERLAPPED` path is never reached, even when an aux CUDA stream is available.

- **EXTERNAL**: shared experts run sequentially **before** routed experts on the main stream
- **MULTI_STREAM_OVERLAPPED**: shared experts run **in parallel** on an aux stream (what 0.18.x did)

### Fix

Reorder the checks: evaluate `MULTI_STREAM_OVERLAPPED` first (when aux stream is available), fall back to `EXTERNAL` only when it's not.

### Benchmark (GLM-5 MoE, 8×RTX PRO 6000 Blackwell, TP=8, C=1, no MTP)

| ctx | Before (EXTERNAL) | After (MULTI_STREAM) | Improvement |
|-----|-------------------|---------------------|-------------|
| 0   | 36.8 tok/s        | **41.8 tok/s**      | **+14%**    |
| 16k | 34.8 tok/s        | **37.8 tok/s**      | **+9%**     |
| 32k | 33.8 tok/s        | **36.8 tok/s**      | **+9%**     |

After the fix, vLLM 0.19.1 is faster than 0.18.1 and matches SGLang (same cutlass MoE backend, no custom allreduce):

| ctx | vLLM 0.19.1 fixed | vLLM 0.18.1 | SGLang (cutlass MoE, no custom AR) |
|-----|-------------------|-------------|-------------------------------------|
| 0   | **41.8**          | 39.8        | 42.6                                |
| 16k | **37.8**          | 37.7        | 37.7                                |
| 32k | **36.8**          | 35.8        | 36.3                                |

cc @bnellnm — this regression was introduced in #35153. The `_has_external_experts` early return prevents aux-stream overlap for all TP-only deployments. Affects any MoE model (DeepSeek-V3, GLM-5, etc.) on multi-GPU TP without EP.

Related: https://github.com/vllm-project/vllm/issues/37113

## Test plan

- [x] Benchmarked GLM-5 TP=8 decode throughput before/after
- [x] Verified FLASHINFER_MLA + FLASHINFER_CUTLASS MoE backends unchanged
- [x] Compared against vLLM 0.18.1 and SGLang baselines
- [ ] Run existing MoE tests to verify no correctness regression

## Comments

### robertgshaw2-redhat @ 2026-04-04T18:37:01Z

Shoot. My bad
---
### robertgshaw2-redhat @ 2026-04-04T19:08:25Z

to be clear, 0.19.1 is not a release yet. I dont think the offending commit is in a release unless I am mistaken
---
### robertgshaw2-redhat @ 2026-04-04T19:10:14Z

this fix isnt quite right. instead, we need to update the "has_external_experts" logic
---
### robertgshaw2-redhat @ 2026-04-04T19:41:00Z

I pushed a commit to your branch with the right fix. Thanks for bringing this up!
---
### milesial @ 2026-04-05T05:59:24Z

Confirming I saw the same issue, impact is 15 to 20% E2E regression on nemotron nano 3, B200 FP8.

Top trace is from yesterday's nightly, bottom is from mid-March, we see the shared expert in the first half of the top trace.

<img width="1701" height="989" alt="image" src="https://github.com/user-attachments/assets/18116671-fe0e-4edb-939f-cc8d1a455173" />

---
### robertgshaw2-redhat @ 2026-04-05T14:30:19Z

> Confirming I saw the same issue, impact is 15 to 20% E2E regression on nemotron nano 3, B200 FP8.
> 
> Top trace is from yesterday's nightly, bottom is from mid-March, we see the shared expert in the first half of the top trace.
> 
> <img alt="image" width="1701" height="989" src="https://private-user-images.githubusercontent.com/30204471/573884126-18116671-fe0e-4edb-939f-cc8d1a455173.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzUzOTk2MTksIm5iZiI6MTc3NTM5OTMxOSwicGF0aCI6Ii8zMDIwNDQ3MS81NzM4ODQxMjYtMTgxMTY2NzEtZmUwZS00ZWRiLTkzOWYtY2M4ZDFhNDU1MTczLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjA0MDUlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwNDA1VDE0MjgzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAzMTRhODllZDNlNTZkYTcxNjNiNDE5ZjJmMTViMGNhOTU0Njc4ZThjM2IyMDM5MzQ1YzAzZTQ0NDE0OTk2ZjMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.4z7Qy7E011nGTYnWJmYUGyP-O9IW4PmK6yrU6XUq1_w">

Does the fix resolve the issue for you?
---
### milesial @ 2026-04-06T04:12:51Z

Yes, this PR solved it.
---