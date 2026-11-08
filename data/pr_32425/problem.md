# Task PR #32425 — [LoRA] Update LoRA expand kernel heuristic (#32425)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **lora_expand_3slice_h1024_t16** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **lora_expand_3slice_h1024_t16** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 14 -> 10 us.
- **lora_expand_3slice_h1024_t256** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **lora_expand_1slice_h4096_t16** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: lora_expand custom op via LoRAKernelMeta (as tests/lora/test_punica_ops.py): 3 slices x hidden 1024 (QKV-like), rank 16, 1 LoRA, 16 and 256 tokens, plus a 1-slice x 4096 control; PR = default block_n 128 -> max(64, next_pow2(128//num_slices)) (untuned GPU path). CUDA-event us per call. Author: _lora_expand_kernel ~14 -> ~10 us (profiler), e2e +1%.

## The problem (issue)
(from `solution.md`, quoted)

> The LoRA expand Triton kernel `_lora_expand_kernel` used a fixed `block_n=128` tiling heuristic regardless of the layer structure. For fused/sliced layers like `QKVParallelLinearWithLoRA`, the output is split into multiple slices (num_slices > 1), so each slice's output dimension N is small. A large block_n of 128 produces too few blocks per slice, leaving streaming multiprocessors (SMs) underutilized and serializing work the GPU could run in parallel.

### Issue — detailed

> `get_lora_op_configs` in vllm/lora/ops/triton_ops/utils.py returns the default tiling config. The hardcoded `block_n=128` ignored `num_slices`, so expand kernels over sliced layers (N already divided by num_slices) tiled too coarsely, yielding few CTAs and low SM concurrency — the expand kernel ran slower than the shrink kernel.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff changes the default `block_n` in `get_lora_op_configs` from a constant `128` to `max(64, next_power_of_2(128 // num_slices))`. Before: every layer got block_n=128 regardless of slice count. After: block_n scales down with the number of slices (e.g. 128→64 for a 2-way split, clamped to a floor of 64 and rounded to a power of 2). Smaller block_n over sliced layers launches more blocks, raising SM concurrency and cutting `_lora_expand_kernel` time ~30% (~14µs→~10µs).

**Author's words:**

> Author found via profiling that qkv_proj expand kernel was slower than shrink because multi-slice layers (num_slices > 1) have smaller per-slice output size N. Setting a smaller block_n launches more blocks and increases SM concurrency, improving `_lora_expand_kernel` ~30% (up to 2x with tuned config) and e2e output token throughput ~1%.

## Reported numbers (PR author)
- gain: 30% (pct)
- evidence: * Profiling shows `_lora_expand_kernel` improves about 30%.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
