# Task PR #29499 — [EPLB] Optimize EPLB with numpy (#29499)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **rearrange_e2e** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **rearrange_e2e** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 1370 -> 1077 ms. author: profile of eplb_state.py rearrange on one rank incl. P2P + GPU copies (1s370ms -> 1s77ms); ours: CPU index-mapping path for all 8 ranks x 48 layers, no transport -- compare the ratio, not the absolute
- **get_ep_ranks_with_expert** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **move_to_buffer** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **move_from_buffer** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: EPLB rebalance index logic on CPU, Qwen3-Next-80B -dp 8 geometry (8 ranks x 68 local = 544 physical experts, 48 MoE layers) as in the author's server. rearrange_e2e = the arm's real move_to_buffer + move_from_buffer for every layer and every rank with P2P transport stubbed and tiny weights (index mapping only). Three primitives kept as secondary metrics. All units ms (bench.py times in ms).

## The problem (issue)
(from `solution.md`, quoted)

> EPLB rebalancing primitives (`get_ep_ranks_with_expert`, `move_to_buffer`, `move_from_buffer`) ran as torch ops on GPU tensors, incurring repeated GPU-CPU synchronizations and transfers per layer. `move_from_buffer` in particular did multiple gpu-cpu transfers. Additionally, the rebalance algorithm did not preserve slot assignments for experts that stayed on the same GPU, forcing unnecessary GPU memcopies. A config post-processing bug also allowed invalid combinations (async with non-default policy, non-positive log intervals).

### Issue — detailed

> The index-manipulation primitives in `rebalance_execute` operated on torch GPU tensors, so `move_from_buffer`, `move_to_buffer`, and `get_ep_ranks_with_expert` triggered device syncs and multiple gpu-cpu round-trips per layer. `move_from_buffer` passed a `dict[int,int] experts_recv_loc` plus `ep_group`, needing collective/transfer work. The rebalance output lacked a post-processing step to keep same-GPU experts in their existing slots, so `rearrange_expert_weights_inplace` copied weights that didn't need to move.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Rewrites `get_ep_ranks_with_expert`, `move_to_buffer`, and `move_from_buffer` to operate in numpy on CPU, removing per-layer GPU-CPU transfers (~20-25% per primitive). `move_from_buffer`'s signature changes from `(expert_weights_buffer, experts_recv_loc dict, new_indices list, ep_group)` to `(expert_weights_buffers, recv_metadata: RecvMetadata, new_indices tensor, ep_rank)`, where the new `RecvMetadata` struct (recv_primary_mask, recv_count, recv_expert_ids, recv_dst_rows) precomputes receive placement so no collective/dict lookup is needed at copy time. `EplbModelState` fields become numpy arrays (`is_unchanged`, `is_received_locally`) and `recv_metadata`. Adds `preserve_intragpu_slots` post-processing so experts staying on a GPU keep their old physical slots, avoiding needless memcopies. Adds `log_balancedness_interval` config with a `model_validator` rejecting async+non-default policy and non-positive intervals, and switches rearrange timing to CUDA events.

**Author's words:**

> Author removes multiple gpu-cpu transfers in `move_from_buffer` and reimplements `get_ep_ranks_with_expert`, `move_to_buffer`, and `move_from_buffer` in numpy, yielding ~20-25% average gain per primitive. Adds a `preserve_intragpu_slots` primitive that post-processes rebalance results so experts assigned to move within the same GPU stay in their slots, avoiding unnecessary GPU memcopies. Adds a `log_balancedness_interval` config parameter to tune balancedness-logging frequency and reduce logging collectives, and fixes a bug in EPLB config post-processing.

## Reported numbers (PR author)
- gain: 27% (rearrange 1370->1077ms E2E); 20-25% per primitive (pct)
- evidence: ~27% speedup
- reported = numbers from the PR body (author's machine); primitives unquantified by the author ('20-25% on average')

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
