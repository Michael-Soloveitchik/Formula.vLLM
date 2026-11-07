# Task PR #32425 — [LoRA] Update LoRA expand kernel heuristic (#32425)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **lora_expand_3slice_h1024_t16** (us): reported 14 -> 10 us. workload: `python3 /task/bench.py`
- **lora_expand_3slice_h1024_t256** (us): no number reported in the PR.
- **lora_expand_1slice_h4096_t16** (us): no number reported in the PR.
- note: lora_expand custom op via LoRAKernelMeta (as tests/lora/test_punica_ops.py): 3 slices x hidden 1024 (QKV-like), rank 16, 1 LoRA, 16 and 256 tokens, plus a 1-slice x 4096 control; PR = default block_n 128 -> max(64, next_pow2(128//num_slices)) (untuned GPU path). CUDA-event us per call. Author: _lora_expand_kernel ~14 -> ~10 us (profiler), e2e +1%.

## Reported numbers (PR author)
- gain: 30% (pct)
- evidence: * Profiling shows `_lora_expand_kernel` improves about 30%.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
