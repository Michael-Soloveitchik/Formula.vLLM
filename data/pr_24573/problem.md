# Task PR #24573 — [EPLB] Reduce EPLB Inference Overhead (#24573)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **eplb_select_experts_m64** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **eplb_select_experts_m64** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **eplb_select_experts_m1024** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **serving_request_throughput** — `req/s` (requests per second), ↑ higher is better. reported 27.74 -> 28.41 req/s.
- **serving_mean_tpot** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 192.4 -> 188.2 ms.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: FusedMoE.select_experts (static) with enable_eplb=True on synthetic routing: 256 logical / 288 physical experts (32 replicated), top_k 8, M=64 and 1024; baseline = torch.rand replica pick + masks inline, PR's solution = torch.compile'd eplb_map_to_physical_and_record. Wall-clock us per call. Author numbers are DP16 DeepSeek-V3 serving w/ EPLB main vs PR (reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> EPLB (Expert Parallelism Load Balancer) replicates logical experts into multiple physical replicas, requiring per-token replica selection and expert-load recording after MoE routing. The original implementation used `torch.rand_like` to pick a replica, which is slow at inference time and not `torch.compile`-friendly, so this EPLB logic stayed uncompiled and added measurable per-decode overhead. It also masked invalid local expert ids before recording load, adding extra tensor ops.

### Issue — detailed

> In `select_experts` (layer.py), the inline block computed `replica_indices = (torch.rand_like(topk_ids) * logical_replica_count[...]).long()` to pick a replica, then did `masked_fill_` on invalid ids and `scatter_add_` to record load. `torch.rand_like` is non-deterministic, slow, and forces a graph break, so the whole EPLB section ran in eager mode on the hot decode path.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff extracts all EPLB replica-selection and load-recording logic out of `select_experts` into a new standalone function `eplb_map_to_physical_and_record` decorated with `@torch.compile(dynamic=True, backend=...simple_compile_backend)`. It replaces non-deterministic `torch.rand_like` replica selection with a deterministic modulo scheme: `pos_indices % replica_count` using a flattened `torch.arange` token-position index, so the replica is chosen purely from token position and routing rank `k`. It drops the `masked_fill_`/invalid-mask handling and records load with a plain `scatter_add_` of ones (noted as a compilable substitute for `torch.bincount`) now that global physical expert-load metrics are collected. A CPU/non-CUDA fallback `eplb_map_to_physical_and_record` is added that just returns `topk_ids` unchanged. Net effect: the EPLB path becomes compile-friendly and deterministic, cutting per-token inference overhead.

**Author's words:**

> Author reduces EPLB overhead three ways: (1) switch from `torch.rand` to a simple modulo-based pseudo-random replica selection keyed on token index and routing rank `k`; (2) remove unnecessary masking of local experts since global physical expert-load metrics are now collected; (3) extract all EPLB logic from `select_experts` into a `torch.compile`-friendly function. Tested with EPLB enabled, `num_redundant_experts=0`, and no rearrangement to isolate replica-selection and load-recording cost on DeepSeek-V3-0324 DP16.

## Reported numbers (PR author)
- gain: 3.97% (pct)
- evidence: **Summary:** Not accounting for the benefits of improved expert load balancing, on the main branch, EPLB introduces ~3.97% throughput drop.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
