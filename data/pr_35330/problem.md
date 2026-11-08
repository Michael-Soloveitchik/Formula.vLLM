# Task PR #35330 — [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement (#35330)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **maxsim_64pairs** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **maxsim_64pairs** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **serving_request_throughput** — `req/s` (requests per second), ↑ higher is better. reported 141.4 -> 161.1 req/s.
- **serving_mean_e2el** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 446.3 -> 391.5 ms.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: ColBERT MaxSim over 64 (query 32 x doc 1024..2048, dim 128) CPU fp32 pairs as serving sees pooling outputs; PR's solution imports compute_maxsim_scores (padded GPU bmm batches), baseline tree lacks it so the bench runs serving.py's per-pair compute_maxsim_score loop. Wall-clock ms per 64-pair batch. Author numbers are e2e rerank serving (reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> ColBERT MaxSim score computation for pooling/rerank requests was done one pair at a time on the CPU. For each (query, document) embedding pair the server called compute_maxsim_score, running a separate matmul and amax/sum per pair on CPU tensors. This serialized the work and used slow CPU compute inside the API server's post-processing stage, bottlenecking end-to-end throughput.

### Issue — detailed

> In serving.py _late_interaction_score, a Python for-loop iterated emb pairs calling compute_maxsim_score(q_emb, d_emb) in utils.py, each computing token_scores = q @ d.T then amax(dim=-1).sum() on CPU tensors. Per-pair invocation meant no batching, repeated tiny-tensor ops, and no GPU use.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds compute_maxsim_scores in utils.py that processes pairs in padded mini-batches on GPU. It groups up to max_batch_size (16) pairs, pads queries/docs to max_q/max_d with zero tensors plus boolean masks, and shrinks the batch when (batch*max_q*max_d) exceeds max_score_matrix_elements (16M) to bound allocation. One torch.bmm computes all token-score matrices; padded doc columns are masked with -inf before amax, padded query rows zeroed before sum, then results moved to CPU. serving.py replaces the per-pair compute_maxsim_score loop with a single compute_maxsim_scores call over all embeddings, then zips the precomputed scores into outputs.

**Author's words:**

> Scores were originally computed on CPU per pair; the author moves the computation to GPU and uses a batched version, giving large performance improvement. Memory is bounded via max_score_matrix_elements=16M and batch_size=16, yielding ~40MB-200MB GPU usage, so OOM is not a concern.

## Reported numbers (PR author)
- gain: 13.9% (pct)
- evidence: PR 35330: [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
