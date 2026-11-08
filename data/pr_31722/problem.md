# Task PR #31722 — [PERF] Speed-up of GDN attention decode part (Qwen3-Next) (#31722)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **fused_recurrent_b32_tp2** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **fused_recurrent_b32_tp2** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 30.14 -> 20.29 us.
- **fused_recurrent_b128_tp2** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 93.38 -> 59.17 us.
- **fused_recurrent_b512_tp2** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 341.6 -> 212.7 us.
- **fused_recurrent_b128_tp4** — `us` (microseconds (1e-6 s)), ↓ lower is better. reported 52.06 -> 33.76 us.
- **decode_output_throughput** — `tok/s` (tokens per second), ↑ higher is better. reported 2.389e+04 -> 2.79e+04 tok/s.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Triton fused_recurrent_gated_delta_rule_fwd (GDN decode) with Qwen3-Next shapes (K=V=128; TP2: H=8/HV=16, TP4: H=4/HV=8), T=1, bf16. PR = BV tile 8->32. CUDA-event us per call. Reported = author's H200 kernel table.

## The problem (issue)
(from `solution.md`, quoted)

> The Triton kernel `fused_recurrent_gated_delta_rule_fwd` sets its V-dimension block size `BV = min(next_power_of_2(V), 8)`, capping the value-block at 8 elements. With the head value dimension V used by Qwen3-Next's gated delta-net far larger than 8, this forces `NV = cdiv(V, BV)` into many small blocks, spawning excessive Triton programs that each do tiny amounts of work. The result is poor GPU occupancy and high per-block overhead during the recurrent decode step.

### Issue — detailed

> In `vllm/model_executor/layers/fla/ops/fused_recurrent.py`, the `BV` block size is clamped to 8, so `NV = cdiv(V, BV)` becomes large and the Triton grid launches many narrow V-blocks. Each program processes only 8 value elements, wasting lanes and inflating scheduling overhead for the memory-bound recurrent decode loop.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff raises the V-block-size cap in `fused_recurrent_gated_delta_rule_fwd` from `BV = min(next_power_of_2(V), 8)` to `min(next_power_of_2(V), 32)`. With BV now up to 32, `NV = cdiv(V, BV)` shrinks 4x, so the Triton kernel launches far fewer, wider V-blocks. Each program handles more value elements, improving GPU occupancy and cutting grid/launch overhead. Net effect: the decode kernel runs ~1.4–1.6x faster across batch sizes on H200/B200, yielding ~16.8% higher end-to-end decode throughput.

**Author's words:**

> The author sped up the GDN decode kernel `fused_recurrent_gated_delta_rule_fwd` for Qwen3-Next shapes, benchmarking it in isolation on H200/B200 (TP=2 and TP=4, batch 32–1024) and showing ~1.4–1.6x kernel speedups. End-to-end, serving Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 at TP=2 with expert parallelism and async scheduling, output token throughput rose from 23891 to 27904 tok/s, a 16.8% improvement.

## Reported numbers (PR author)
- gain: 16.8% (pct)
- evidence: **Speedup: 16.8%**
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
