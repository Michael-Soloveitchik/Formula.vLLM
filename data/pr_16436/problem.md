# Task PR #16436 — [Core] Speed up decode by remove synchronizing operation in sampler (#16436)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **tpot** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **tpot** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **itl** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **throughput** — `tok/s` (tokens per second), ↑ higher is better. no number reported in the PR.

Workload command (run per arm): `python3 benchmarks/benchmark_serving.py --backend vllm --model facebook/opt-125m --dataset-name random --random-input-len 1000 --random-output-len 100 --num-prompts 100`
Mechanism / note: Removes CUDA sync in apply_penalties; win via GPU serving + trace. ~8% decode for small models.

## The problem (issue)
(from `solution.md`, quoted)

> `apply_penalties` in vllm/model_executor/layers/utils.py uses boolean advanced indexing (`logits[logits > 0]` and `logits[logits <= 0]`) to selectively scale logits by the repetition penalty. Boolean mask indexing on a GPU tensor invokes `nonzero`, which forces a CUDA device-to-host synchronization. The sync stalls the CPU until the GPU resolves the mask, serializing the sampling phase against the forward pass and blocking overlap.

### Issue — detailed

> `logits[mask]` with a boolean GPU mask triggers `torch.nonzero`, a legitimate sync — CPU needs the nonzero count to size the index tensor and configure subsequent kernel launches (see pytorch/pytorch#12461). Four such masked-index ops per call each force a device sync, preventing CPU/GPU overlap.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Before, the code wrote back into logits via two boolean-masked index assignments (`logits[logits>0] /= ...[logits>0]`, `logits[logits<=0] *= ...[logits<=0]`), each triggering `nonzero` syncs. After, it computes a dense `penalties = torch.where(prompt_mask | output_mask, repetition_penalties, 1.0)`, then builds a single dense `scaling = torch.where(logits > 0, 1.0/penalties, penalties)` and applies it with one elementwise `logits *= scaling`. `torch.where` produces a full-shape tensor with no nonzero/mask-indexing, so no CPU sync occurs. Result: sampling CPU work overlaps the GPU forward pass, ~8% decode speedup on small models.

**Author's words:**

> Author identified that boolean advanced indexing calls `nonzero`, which requires a CPU sync in PyTorch (confirmed via sync debug mode and pytorch/pytorch#12461, where the sync is deemed unavoidable for mask indexing). By refactoring to avoid the indexing, the sync is removed, letting much more sampling-phase CPU work overlap with the GPU forward pass, giving an 8% decode speedup for smaller models.

## Reported numbers (PR author)
- gain: 8% (pct)
- evidence: By refactoring the code to avoid the indexing, we can remove the sync and allow much more of the sampling phase CPU work to overlap with the forward pass on the
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
