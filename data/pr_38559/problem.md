# Task PR #38559 — [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement (#38559)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **mean_pool_e5base_64x512** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **mean_pool_32x256_h4096** (us): no number reported in the PR.
- **serving_request_throughput** (req/s): reported 238.5 -> 252.7 req/s.
- **serving_mean_e2el** (ms): reported 265.4 -> 246.5 ms.
- note: MeanPool.forward via the real PoolingMetadata cursor: 64 x 512 tokens, hidden 768 fp16 (e5-base serving shape) and 32 x 256, hidden 4096 bf16; baseline = fp32 cumsum over all tokens, PR's solution = chunked index_add_. Wall-clock us per forward. Author numbers are e2e embeddings serving (reported-only).

## Reported numbers (PR author)
- gain: 5.9% (pct)
- evidence: PR 38559: [Perf] Optimize mean pooling using chunks and index_add, 5.9% E2E throughput improvement
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
