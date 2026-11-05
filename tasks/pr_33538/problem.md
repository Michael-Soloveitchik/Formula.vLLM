# Task PR #33538 — [Kernel] Triton-based Top-k and Top-p sampler kernels (#33538)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (gold)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
Top-k / top-p filtering of the logits before sampling is done with a full sort of every row of the vocabulary on the GPU. Sorting 32k–128k values per sequence is far more work than needed to find one cutoff value per row, and the cost grows with batch size. Decode steps pay this latency on every token.

### Issue — detailed
`vllm/v1/sample/ops/topk_topp_sampler.py` implements `apply_top_k_top_p` with `logits.sort(dim=-1)` followed by cumulative-sum masking (the "PyTorch sort-based" path). For a batch of 16 at vocab 32768 this takes ~0.28 ms and ~26 MB of scratch memory, and at batch 64 ~0.45 ms / ~105 MB. Only the k-th largest value (top-k) or the pivot where the cumulative probability crosses p (top-p) is needed per row; a search for that pivot does not require sorting. There is no fused GPU kernel for this in the base tree, and no `vllm/v1/sample/ops/topk_topp_triton.py` module exists. The correctness tests expect a function `apply_top_k_top_p_triton(logits, k, p)` in that module that produces the same kept/masked set as the sort-based reference, including rows with `-inf` logits, duplicate logits, extreme k and p values, and mixed partial batches.

## Expert solution (reference — the merged PR)
**General:** Replace the per-row sort with a Triton kernel that finds the top-k / top-p pivot by a pivot search (ternary search on the value range), after first truncating each row to a small "outlier" subset of candidate logits chosen with a stochastic cutoff. Values below the pivot are masked to `-inf`; nothing is sorted.

**Detailed:** New module `vllm/v1/sample/ops/topk_topp_triton.py` (~1000 lines) with `apply_top_k_top_p_triton`. The kernel gathers, per program, a truncated `[num_program, vocab_size]`-shaped buffer of candidate logits (about 80 MiB of extra VRAM), then runs a ternary search over pivots. For top-k it tracks the count of values above the pivot and the min value above it to handle duplicates; for top-p it uses the termination condition `p_over_pivots_sum >= p AND p_over_pivots_sum - (p_over_pivots_min * num_p_over_pivots_min) < p` — the pivot where the sum of probabilities above it reaches p but dropping the smallest of them falls below p. `topk_topp_sampler.py` dispatches to the Triton path on CUDA (with the sort-based path kept as reference / fallback), and `vllm/utils/math_utils.py` gains a small helper used by the kernel. Tests for the Triton path and `benchmarks/benchmark_topk_topp.py` (Triton vs PyTorch, per scenario) are added.

**Author's words:** Re-opening PR #25824 with correctness and benchmark scripts from PR #32558. Passes all correctness tests, faster overall compared to #32558 except for top-p-only cases. This algorithm includes a truncation step, which gathers a small "outlier" subset of the logits to reduce the search space using a stochastic cutoff. Also handles duplicate logits or probabilities. Up to 9.94x vs the PyTorch top-k sampler (batch 4, vocab 32768); ~2–7x at batch 16–64.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline, = the sort-based PyTorch path) and after-merge (gold). Your
edit is timed on the SAME workload. Beat gold on these metrics:
- **topk_whole b16 v32k** (ms): reported 0.280 -> 0.067 ms. workload: `benchmarks/benchmark_topk_topp.py --batch-sizes 1 16 128 2048 --vocab-sizes 32768 131072`
- **topp_whole b16 v32k** (ms): reported 0.361 -> 0.184 ms.
- **topk_topp_whole b16 v32k** (ms): reported 0.242 -> 0.093 ms.
- **mixed_partial b16 v32k** (ms): reported 0.266 -> 0.205 ms.

## Reported numbers (PR author)
- metric: benchmark_topk_topp.py Triton vs PyTorch latency per scenario (plus VRAM)
- value: up to 9.94x vs PyTorch top-k sampler (batch 4, vocab 32768)
- unit: ms

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than gold on the measured experiments wins. Same speed = ties gold.
