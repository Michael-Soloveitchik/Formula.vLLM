# pr_21222 — measured results (real units: µs per op)

PR: https://github.com/vllm-project/vllm/pull/21222

Workload: free 256 blocks then reclaim 256 blocks = 512 queue ops/call; inner=200, pairs=5, repeats=5 (for variance).
Unit = **microseconds per queue op**, the PR's own unit. Baseline (denominator) = pre-PR `append()`/`popleft()` loop = 0.7467 ± 0.0225 µs/op. Arms = `append_n`/`popleft_n` bulk. Correctness gate PASSES for all arms (see `../results.json`). PR's second reported op (`free_blocks` 33→9 µs, 3.7×) omitted; headline is `get_new_blocks`.

## Results (µs/op, mean ± sd over 5 runs)

Reference = **measured** (our run of the PR's solution) = **1.00×**. `vs measured` is each arm's latency ratio to it (>1 faster, <1 slower).

| row | latency (µs/op) | vs measured | note |
|---|---|---|---|
| reported | 8.000 | 18.8× vs pre-PR | PR author, from 150 µs (GPU box) |
| measured | 0.194 ± 0.010 | 1.00× | our run of PR's solution — reference |
| claude | 0.226 ± 0.033 | 0.86× | counts |
| codex | 0.235 ± 0.020 | 0.82× | counts |

## Advantage metric (idea A)

`Advantage = min(1, (T_base − T_agent)/(T_base − T_expert))`, times in µs/op, correctness gate passed.

| arm | µs/op | Advantage |
|---|---|---|
| baseline | 0.7467 | — |
| PR's solution | 0.1937 | 1.000 |
| codex | 0.2350 | 0.925 |
| claude | 0.2257 | 0.942 |

## Honesty notes

- Units match the PR: µs per queue op. The PR's two reported numbers (`get_new_blocks` 150→8 µs, `free_blocks` 33→9 µs) are full block_pool ops on a GPU box; ours is a CPU slice of the same linked-list primitives — same direction, host-specific magnitude.
- Variance (± sd) is across 5 independent measurement runs on this host.
- codex/claude ≈ PR's solution: the task has essentially one correct bulk-op implementation shape.
