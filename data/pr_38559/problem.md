# Task PR #38559 — [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement (#38559)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **mean_pool_e5base_64x512** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **mean_pool_e5base_64x512** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **mean_pool_32x256_h4096** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **serving_request_throughput** — `req/s` (requests per second), ↑ higher is better. reported 238.5 -> 252.7 req/s.
- **serving_mean_e2el** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 265.4 -> 246.5 ms.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: MeanPool.forward via the real PoolingMetadata cursor: 64 x 512 tokens, hidden 768 fp16 (e5-base serving shape) and 32 x 256, hidden 4096 bf16; baseline = fp32 cumsum over all tokens, PR's solution = chunked index_add_. Wall-clock us per forward. Author numbers are e2e embeddings serving (reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> Mean pooling computed a full `torch.cumsum` over the entire `hidden_states` tensor in float32, then gathered per-sequence sums via `cumsum[end] - cumsum[start] + hidden_states[start]`. The cumsum allocates a large temporary float32 tensor the same shape as all tokens stacked together, and computes a prefix sum across every token even though only per-segment totals are needed. This is redundant compute plus a big transient memory spike.

### Issue — detailed

> In `MeanPool.forward` (vllm/model_executor/layers/pooler/seqwise/methods.py), `torch.cumsum(hidden_states, dim=0, dtype=torch.float32)` builds a float32 prefix-sum tensor over the full stacked token tensor, then indexes it with `first_token_indices_gpu`/`last_token_indices_gpu`. Computing a global cumsum is wasteful since only per-segment sums are needed, and the float32 cast of the whole tensor spikes memory.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Replaces the full-tensor `torch.cumsum` + index-difference approach with segment accumulation via `index_add_`. It builds `segment_ids` using `torch.repeat_interleave` (e.g. prompt_lens [2,1,3] → [0,0,1,2,2,2]) mapping each token row to its sequence, then accumulates into a `(num_seqs, hidden_size)` float32 `segment_sums` tensor. Accumulation iterates `hidden_states` in ~16MB chunks, casting each chunk to float32 and calling `segment_sums.index_add_`, bounding the transient float32 cast to one chunk instead of the whole tensor. Finally divides `segment_sums` by `prompt_lens` for the means. Adds an early return `new_empty((0, hidden_size))` for empty batches and casts `prompt_lens` to int64. Yields 5.9% E2E throughput gain by avoiding the redundant global prefix sum and large temp allocation.

**Author's words:**

> Optimize mean pooling using chunks and `index_add`, avoiding redundant calculation from `cumsum` and the large temporary tensor. Accuracy covered by existing MTEB and online pooling unit tests; perf validated with `vllm bench serve` on `intfloat/multilingual-e5-base` showing ~5.9% E2E throughput improvement (238.52 → 252.72 req/s).

## Reported numbers (PR author)
- gain: 5.9% (pct)
- evidence: PR 38559: [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
