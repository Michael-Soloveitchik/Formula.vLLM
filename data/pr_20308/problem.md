# Task PR #20308 — [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel (#20308)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> The unified Triton attention kernel computed attention over every KV block in the sequence during prefill, iterating `num_blocks = cdiv(seq_len, BLOCK_SIZE)` tiles for each query block. Under causal masking, query tokens early in the prompt can only attend to KV positions at or before themselves, so KV blocks beyond the current query block's maximum reachable position contribute nothing but are still loaded and processed. This wastes compute and memory bandwidth on tiles that are fully masked out.

### Issue — detailed

> In `kernel_unified_attention_2d` in vllm/attention/ops/triton_unified_attention.py, the per-query-block tile loop `for j in range(0, num_blocks)` used `num_blocks` derived from the full `seq_len`. Because the kernel iterated to the end of the sequence regardless of which query positions the current q_block covers, it processed KV tiles that lie entirely in the causally-masked future region, doing useless loads and flops.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Before, `num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)` iterated KV tiles over the entire sequence length. After, the kernel computes `max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1`, the longest sequence prefix any query token in the current q_block can attend to under causal masking, clamps it to `seq_len` via `tl.minimum` to account for padding in the last q_block, and sets `num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)`. This bounds the tile loop to only the KV blocks that can contribute, skipping fully-masked future tiles. The effect is fewer loaded/computed tiles per query block during prefill, lowering TTFT and overall prefill latency without changing results.

**Author's words:**

> The author optimizes prefill attention in the unified Triton kernel by reducing the number of tiles processed during the prefill phase. They leverage the causal mask to skip unnecessary computations—blocks beyond the longest prefix reachable by the current query block—yielding more efficient execution that is especially beneficial for long prompts (~1.75x at 16k tokens, batch size 1).

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **prefill_attn_s2000** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **prefill_attn_s8000** (us): no number reported in the PR.
- **prefill_attn_s16000** (us): no number reported in the PR.
- **serving_mean_ttft** (ms): reported 3874 -> 3588 ms.
- **serving_request_throughput** (req/s): reported 44.37 -> 45.9 req/s.
- note: unified_attention (2D Triton kernel) pure prefill, 1 sequence, Llama-3.1-8B shapes (32/8 heads, head 128, block 16, bf16), prompts 2000/8000/16000; PR = skip KV tiles beyond the causal prefix. CUDA-event us per call. Wheel 0.9.2 pins torch 2.7.0 cu126 (no sm_120): metadata torch_fix swaps the cu128 build of the same torch. Author: ~1.75x kernel-side at 16k (graph) + ShareGPT serving (reported-only).

## Reported numbers (PR author)
- gain: 3%, 5% (pct)
- evidence: Despite the relatively short prompt lengths used in this benchmark, the results still demonstrate a ~3% improvement in throughput, along with over 5% reductions
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
