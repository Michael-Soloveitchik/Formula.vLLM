# Task PR #27402 — [Kernel] LoRA triton kernels support PDL (#27402)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **lora_shrink_expand_1slice_t16** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **lora_shrink_expand_1slice_t16** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **lora_shrink_expand_1slice_t256** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **lora_shrink_expand_3slice_t16** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **lora_shrink_expand_3slice_t256** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **serving_output_throughput** — `tok/s` (tokens per second), ↑ higher is better. reported 1383 -> 1468 tok/s.
- **serving_mean_tpot** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 63.67 -> 59.51 ms.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: lora_shrink + lora_expand back to back (a LoRA linear) via LoRAKernelMeta: hidden 4096, rank 16, 1 LoRA, 1 and 3 slices, 16 and 256 tokens; PR = PDL (griddepcontrol) in the LoRA Triton kernels (sm_90+, sm_120 ok). CUDA-event us per shrink+expand pair. Author: '5-20% kernel runtime speedup' + Qwen3-30B-A3B LoRA serving on H20 (reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> The LoRA triton kernels run as serialized, back-to-back GPU launches (shrink followed by expand, and the fused-MoE shrink/expand pair) with no overlap between a kernel finishing and the next one starting. Each dependent kernel can only begin after the prior kernel's grid fully completes and the runtime schedules the next launch, leaving the GPU idle during that launch/drain gap. This gap is pure overhead on top of the actual matmul work and is proportionally largest when the matmuls themselves are small.

### Issue — detailed

> The shrink output feeds the expand input, forcing strict ordering between the two triton launches (`_lora_shrink`/`_lora_expand` and the `_fused_moe_lora_shrink`/`_fused_moe_lora_expand` pair). The dependent kernel cannot start until the producer kernel's entire grid drains, so the scheduler inserts an idle gap on every launch; `mm_k` and the fused-MoE main loop have no mechanism to begin prefetching the next stage's inputs during that gap.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff adds CUDA Programmatic Dependent Launch (PDL) via Triton GDC primitives, gated by a new cached `supports_pdl()` helper that returns True only on CUDA SM90+. A `USE_GDC` constexpr is threaded through the shrink/expand kernels and `mm_k`/`do_shrink_kernel`/`do_expand_kernel`, and each launch passes `launch_pdl=use_gdc` as kernel metadata. In the K-loop the LoRA weight `b`/`tiled_b` load is hoisted above a `tl.extra.cuda.gdc_wait()` so the weight prefetch overlaps the wait on the prior kernel; the producer kernel (shrink, `IS_PRIMARY=True`) calls `tl.extra.cuda.gdc_launch_dependents()` to let the runtime launch the dependent expand kernel early. The fused-MoE expand runs as the waiter (`IS_PRIMARY=False`), and the split-K write-back is relaxed to `tl.atomic_add(..., sem="relaxed")`, together overlapping the two launches and removing the inter-kernel idle gap.

**Author's words:**

> The author adds PDL support to the LoRA triton shrink/expand kernels so dependent kernels overlap, reporting a slight end-to-end serving gain on H20 (74.05s -> 69.74s, TPOT 63.67ms -> 59.51ms) and 5-20% kernel-runtime speedup for smaller Ms on Qwen3-32B TP=4 shapes. This is part 2 of issue #27371 and requires Triton 3.5.0 (Triton 3.4.0 errored). Adding the same logic to fused_moe_lora shrink/expand was suggested as follow-up.

## Reported numbers (PR author)
- gain: 5%, 20% (pct)
- evidence: This is a good feature to have, see following results, we generally see 5% to 20% kernel runtime speedup for smaller Ms.
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
