# Task PR #28280 — [Kernel] Optimization of the mm_k operator. (#28280)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> The `mm_k` Triton kernel helper (used by LoRA shrink/expand ops) handles the non-`EVEN_K` case inefficiently. On every K-loop iteration it applies element-wise masking on both `tl.load` calls and always issues a `tl.dot()`, even when the K-block is entirely in bounds or entirely out of bounds. Masking is only truly needed for the single partial block at the K boundary, so the per-iteration mask computation and the wasted `tl.dot()` on fully-out-of-range blocks are redundant overhead.

### Issue — detailed

> In `vllm/lora/ops/triton_ops/kernel_utils.py`, the `else` (not `EVEN_K`) branch of the `for k in range(tl.cdiv(K, BLOCK_K*SPLIT_K))` loop computes a mask `offset_k < K - k*(BLOCK_K*SPLIT_K)` on each `tl.load` and unconditionally accumulates `tl.dot(tiled_a, tiled_b)`. This treats every iteration as if it could be partial, so fully-valid blocks pay mask cost and fully-invalid blocks still run a zeroed `tl.dot()`.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff adds a `base_k` parameter and precomputes each iteration's global K offset `iter_k = k*STEP_K + base_k` (with `STEP_K = BLOCK_K*SPLIT_K`) and `block_end = iter_k + BLOCK_K`. For the non-`EVEN_K` case it now branches three ways: if `iter_k >= K` the block is fully out of range and is skipped entirely (no load, no `tl.dot()`); if `block_end <= K` it takes a fast path with unmasked `tl.load`/`tl.dot`; only the remaining partial boundary block builds a `mask = iter_k + arange(0,BLOCK_K) < K` and loads with masking. Callers pass `base_k=0` in `do_expand_kernel` and `base_k=pid_sk*BLOCK_K` in `do_shrink_kernel` to account for SPLIT_K grouping. This removes per-iteration masking from in-bounds blocks and eliminates wasted matmuls on out-of-bounds blocks, speeding up the kernel.

**Author's words:**

> The author observed that in the non-`EVEN_K` case the main branch adds masking and a `tl.dot()` on every loop iteration, but masking is only needed for partially out-of-bound accesses. They split each iteration into three cases: completely in bounds (skip masking, just `tl.dot()`), completely out of bounds (skip entirely — no masking or `tl.dot()`), and partial (mask as before). Reducing redundant masking and `tl.dot()` calls improves speed, particularly on less advanced GPU architectures, with gains shown on Metax C500 and H800.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **output_throughput** (tok/s): reported 267.8 -> 296.9 tok/s. workload: `vllm bench serve --num-prompts 100 --dataset-name sharegpt`
- **tpot** (ms): reported 67.93 -> 61.74 ms.
- note: Triton matmul (EVEN_K masking). PR gives serving numbers only (C500/H800).

## Reported numbers (PR author)
- gain: 8% (pct)
- evidence: Output token throughput increases by approximately 8%, consistent with the intuition that this change is more friendly to older GPU architectures.*

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
