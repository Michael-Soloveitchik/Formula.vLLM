# Task PR #42311 — [Model] [Perf] Use flatten for Qwen3.5's GDN output projection (#42311)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **gdn_output_projection** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **gdn_output_projection** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **merge_heads_op** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 23.26 -> 2.828 us.
- **serving_mean_tpot** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 88.38 -> 67.59 ms.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: GatedDeltaNetAttention._output_projection (unbound, minimal self with real RMSNormGated + Linear out_proj) on Qwen3.5-0.8B shapes (16 x 128 heads, hidden 1024, 8 decode tokens, fp16, eager); PR = einops rearrange -> flatten(-2). Wall-clock us per call. Author numbers are per-op profiler events (reported-only) + e2e serving.

## The problem (issue)
(from `solution.md`, quoted)

> Qwen3.5's GDN (gated delta net) output projection merges the last two tensor dimensions (head and head-dim) using einops `rearrange(core_attn_out, "... h d -> ... (h d)")`. einops parses and validates the pattern string in Python on every call, which is expensive relative to the trivial reshape it performs. This Python-level overhead is incurred on every GDN forward pass.

### Issue — detailed

> In `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, the `_output_projection`, `forward_xpu`, and `forward_cpu` paths call `rearrange(core_attn_out, "... h d -> ... (h d)")` right before `self.out_proj`. einops re-parses the pattern string and validates shapes in Python on each invocation, which dominates the cost of what is just a contiguous merge of the final two dims.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff replaces `core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")` with `core_attn_out = core_attn_out.flatten(-2)` in three code paths (`_output_projection`, `forward_xpu`, `forward_cpu`), keeping the original pattern as an inline comment. `flatten(-2)` is the native PyTorch equivalent that merges the last two dimensions without any Python-level string parsing or validation. The reshape semantics are identical, so model behavior is unchanged, while eliminating einops overhead yields ~8x (CUDA) to ~21x (CPU) per-call speedup on this operation.

**Author's words:**

> Replace einops's `rearrange(core_attn_out, "... h d -> ... (h d)")` with the PyTorch-native equivalent `core_attn_out.flatten(-2)` to merge the last two dimensions, avoiding expensive Python-level pattern string parsing and validation on every GDN forward call. The author verified correctness via PPL comparison on Qwen3.5-0.8B (19.510 flatten vs 19.510 rearrange, both ~0.63% from HF expected), confirming the reshape does not alter tensor ordering or model output.

## Reported numbers (PR author)
- gain: 21.41x (pct)
- evidence: Per-call average speedup: `21.41x`
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
