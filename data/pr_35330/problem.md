# Task PR #35330 — [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement (#35330)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **maxsim_64pairs** (ms): no number reported in the PR. workload: `python3 /task/bench.py`
- **serving_request_throughput** (req/s): reported 141.4 -> 161.1 req/s.
- **serving_mean_e2el** (ms): reported 446.3 -> 391.5 ms.
- note: ColBERT MaxSim over 64 (query 32 x doc 1024..2048, dim 128) CPU fp32 pairs as serving sees pooling outputs; PR's solution imports compute_maxsim_scores (padded GPU bmm batches), baseline tree lacks it so the bench runs serving.py's per-pair compute_maxsim_score loop. Wall-clock ms per 64-pair batch. Author numbers are e2e rerank serving (reported-only).

## Reported numbers (PR author)
- gain: 13.9% (pct)
- evidence: PR 35330: [Perf] Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
