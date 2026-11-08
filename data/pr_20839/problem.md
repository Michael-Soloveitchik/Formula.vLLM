# Task PR #20839 — [Model] Replace Mamba2 RMSNorm Gated with Fused Triton Kernel (#20839)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **rmsnorm_gated_16k** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **rmsnorm_gated_16k** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **rmsnorm_gated_32k** — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **latency_16k** — `s` (seconds), ↓ lower is better. reported 0.6323 -> 0.5812 s.
- **latency_32k** — `s` (seconds), ↓ lower is better. reported 1.28 -> 1.179 s.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Mixer2RMSNormGated.forward_cuda on Bamba-9B-v2 shapes (hidden 8192, bf16, 16k/32k tokens); baseline = python silu-gate + ops.rms_norm, PR's solution = fused Triton rms_norm_gated. TP helpers stubbed to world_size 1. CUDA-event us per call. Author numbers are whole-model benchmark_latency (kept reported-only).

## The problem (issue)
(from `solution.md`, quoted)

> The RMSNormGated layer in the Mamba2 mixer used unfused operations: a separate SiLU gating multiply, a dtype cast, and a standalone `ops.rms_norm` call. Each step launched its own kernel and round-tripped intermediate tensors through HBM. On long prompts this layer consumed as much wall-clock time as the entire Mamba2 SSD scan, far more than expected for a normalization op.

### Issue — detailed

> `RMSNormGated.forward_cuda` in `mamba_mixer2.py` computed `y = x * silu(gate.float())`, cast back to x.dtype, then called `ops.rms_norm` into a fresh `torch.empty_like` buffer. Gating, cast, and normalization were three unfused ops, each a separate kernel with full-tensor HBM traffic — no fusion of the gate-multiply into the norm reduction.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds new file `ops/layernorm_gated.py` with a Triton fused kernel (`_layer_norm_fwd_1pass_kernel` + `_layer_norm_fwd` + `rms_norm_gated` wrapper), adapted from mamba_ssm. The kernel does the SiLU gate multiply (`x *= z*sigmoid(z)`), RMS variance reduction, normalization, and weight scaling in a single pass per row, keeping intermediates in registers instead of HBM. `forward_cuda` now calls `rms_norm_gated(x, weight, z=gate, norm_before_gate=False)` instead of the three-op silu+cast+`ops.rms_norm` sequence. The TP/ngroups guard changes from `tp_size > 1 or n_groups != 1` to `(n_groups % tp_size) != 0 or n_groups != 1`, still falling back to `forward_native` outside the TP=1/ngroups=1 case. Yields ~8–9% end-to-end latency improvement with unchanged gsm8k quality.

**Author's words:**

> Replace the current unfused RMSNormGated implementation with a fused Triton kernel taken from the mamba_ssm repo's `layernorm_gated.py`. The author scopes it to TP=1 and ngroups=1 for now, noting broader use cases likely need similar fixes, and validates output quality e2e via lm_eval gsm8k and performance via benchmark_latency.

## Reported numbers (PR author)
- gain: 9% (pct)
- evidence: **Approximately 8~9% latency improvements observed.**
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
