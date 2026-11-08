# Task PR #41163 — [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark (#41163)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **allpool_forward_median** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **allpool_forward_median** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 59.89 -> 39.65 us.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: bench.py is the author's method-level benchmark from the PR body, verbatim (64 reqs x 128 tokens, subgroup 32, hidden 4096, bf16; median of 5 x 4000 iters). /task = tasks/pr_41163 mounted read-only in the task image.

## The problem (issue)
(from `solution.md`, quoted)

> AllPool.forward materializes per-request slices with a Python list comprehension that first calls `.tolist()` on two GPU tensors (`first_token_indices_gpu` and `last_token_indices_gpu`), then builds one slice per request. The two `.tolist()` calls each force a host-visible device sync, stalling the CPU on the GPU, and producing N individual slice ops instead of one bulk operation. This makes the method-level cost dominated by launch/sync overhead rather than useful work.

### Issue — detailed

> In `vllm/model_executor/layers/pooler/tokwise/methods.py`, `AllPool.forward` built `hidden_states_lst` by zipping `first_token_indices_gpu.tolist()` and `last_token_indices_gpu.tolist()` and slicing `hidden_states[first:last+1]` per request. The two `.tolist()` calls are blocking device syncs, and the comprehension issues one slice per request — O(N) ops plus two syncs every forward.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff replaces the two `.tolist()` syncs and the per-request slice comprehension with a single bulk path. It reads `num_scheduled_tokens_cpu.tolist()` (already on CPU, no device sync) to get `split_sizes`, computes the subgroup bounds with two scalar `.item()` calls (`group_start` from `first_token_indices_gpu[0]`, `group_end` from `last_token_indices_gpu[-1]+1`), slices the full tensor to the subgroup once, then calls `hidden_states_group.split(split_sizes)` to produce the per-request list in one op. An empty-`split_sizes` guard returns `[]`. Net effect: fewer host-device syncs and one slice+split instead of N slices, yielding ~51% speedup at the method level.

**Author's words:**

> Slice the subgroup out of the full hidden_states tensor once using its group start/end bounds, then split that contiguous slice by per-request token counts (`num_scheduled_tokens_cpu`) instead of iterating and slicing per request. This avoids the extra host-visible syncs from `.tolist()`/`.item()` in the hot path and cuts method-level time from 59.89 us to 39.65 us (~51% faster).

## Reported numbers (PR author)
- gain: 51% (pct)
- evidence: PR 41163: [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark
- reported = PR body (us, median of the author's benchmark on their GPU)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
