# Task PR #31984 — [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel (#31984)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **sw128_decode_b1_kv4096** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **sw128_decode_b1_kv4096** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **sw128_decode_b1_kv8192** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **sw128_decode_b1_kv16384** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **sw128_decode_b8_kv8192** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: unified_attention forced onto the 3D (split-softmax decode) Triton kernel with gpt-oss-20b shapes (64/8 heads, head 64, window 128, block 16, bf16): 1 decode query vs KV 4096/8192/16384 and 8 x 8192; PR prunes out-of-window tiles in the 3D kernel. CUDA-event us per call. Author: 3-4% e2e latency at 6.4k-12.8k output tokens (no kernel number).

## The problem (issue)
(from `solution.md`, quoted)

> The 3D Triton unified attention kernel (`kernel_unified_attention_3d`) always iterated over every tile of the KV cache up to `num_tiles`, even when a sliding window was active. For sliding-window attention, each query position can only attend to keys within `SLIDING_WINDOW` positions back, so tiles entirely outside that window contribute nothing after masking. Processing them wasted compute on loads, dot products, and masking that are discarded.

### Issue — detailed

> In `kernel_unified_attention_3d`, the per-segment tile loop ran `for j in range(segm_idx * tiles_per_segment, min((segm_idx+1)*tiles_per_segment, num_tiles))`, covering all tiles regardless of window. The sliding-window constraint was only enforced later via masking, so out-of-window tiles were still loaded and computed, then zeroed — pure wasted work. The 2D kernel already pruned these tiles; the 3D kernel did not.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff adds sliding-window tile pruning to `kernel_unified_attention_3d`. Before, `tile_start=0` and `tile_end=num_tiles` were effectively fixed and the loop spanned the whole segment range. After, when `SLIDING_WINDOW > 0` and not `USE_MM_PREFIX`, it computes the Q-block's query row span (`qpos_lo`/`qpos_hi`), derives the union of allowed key positions `[context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]`, converts those to clamped tile indices `tile_start`/`tile_end`. The loop bounds become `max(segm_idx*tiles_per_segment, tile_start)` to `min((segm_idx+1)*tiles_per_segment, tile_end)`, so only tiles intersecting the window are processed. Effect: skips loads/compute on fully-masked out-of-window tiles, cutting latency ~3-4% at long output lengths.

**Author's words:**

> Author applies the same optimization already used in the 2D kernel to the 3D kernel, ensuring only tiles within the defined sliding window are processed. This reduces computational overhead and improves performance, with gains growing for longer sequences (~3-4% at 6,400 and 12,800 output tokens). Correctness confirmed via existing unit tests (512 passed) and gsm8k lm_eval.

## Reported numbers (PR author)
- gain: 4% (pct)
- evidence: For longer sequence lengths, the performance improvement introduced by this PR becomes more pronounced, reaching approximately 3–4% for output lengths of 6,400
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
