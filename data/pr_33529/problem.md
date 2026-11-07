# Task PR #33529 — Triton MLA perf fixes (#33529)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> Triton MLA decode uses a fixed `num_kv_splits = 4` regardless of context length or GPU SM count, so at low batch size only ~4 CTAs are launched for the KV reduction, leaving the vast majority of SMs idle. Separately, the grouped stage-1 kernel loads V from the KV buffer as if it were distinct from K, but MLA stores a single combined latent `c_kv` that serves as both K and V, so V is redundantly re-loaded from HBM.

### Issue — detailed

> In `forward_mqa` (triton_mla.py) the split count was hardcoded to 4, so `_fwd_grouped_kernel_stage1` launched a grid too small to saturate SMs when context is long and batch is tiny. Inside that kernel, V was loaded via `offs_buf_v` into `V_Buffer` even though MLA's c_kv already holds the value data as K, doubling the latent reads from the KV cache.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> `num_kv_splits` is now computed dynamically: `ideal_splits = max(1, max_seq_len // 512)` rounded up to the next power of 2, then capped at `sm_count * 2` (occupancy multiplier), replacing the fixed 4 (batch-invariant path still forces 1). The SM count is cached once in `__init__` via `torch.cuda.get_device_properties(0).multi_processor_count`. An `is_mla`/`IS_MLA` flag is threaded through `decode_attention_fwd[_grouped]` into `_fwd_grouped_kernel_stage1`; when set, the kernel skips the separate V load and instead reuses K via `v = tl.trans(k)`, eliminating the redundant c_kv read. Load instructions also gain `cache_modifier` hints (`.ca` for Q/page table, `.cg` for K/KPE) and base offsets are hoisted out of the inner loop with `tl.range` to better overlap load/compute.

**Author's words:**

> The author identified two primary issues: suboptimal KV splitting during low batch count that underutilizes SMs, and unnecessary loading of the Q/V vector since c_kv already contains both. The fix makes KV splitting adaptive to context length and SM count, and avoids the redundant c_kv load in the MLA path. Validated on Kimi 2.5, DeepSeek V2, and Qwen 235B, showing ~53% throughput gain at batch 1 / 80K context.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **mla_decode_b1_kv8192** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **mla_decode_b1_kv32768** (us): no number reported in the PR.
- **mla_decode_b1_kv81920** (us): no number reported in the PR.
- **mla_decode_b8_kv8192** (us): no number reported in the PR.
- **peak_throughput_1req_80k** (tok/s): reported 34 -> 52 tok/s.
- **peak_throughput_8req_80k** (tok/s): reported 176 -> 184 tok/s.
- note: TritonMLAImpl.forward_mqa (unbound, minimal self/metadata/layer stubs) on DeepSeek-V2-Lite-like MLA decode: 16 heads, latent 512+64, bf16 KV, block 64; ctx 8K/32K/80K at batch 1 and 8K at batch 8. PR = SM-aware num_kv_splits (was fixed 4) + kernel load tweaks + is_mla path. CUDA-event us per call. Author (Kimi K2.5 DCP8, sm120) peak tok/s reported-only.

## Reported numbers (PR author)
- gain: 1.05x, 4.36x (pct)
- evidence: | Context | OLD (tok/s) | NEW (tok/s) | Improvement | |---------|-------------|-------------|-------------| | Short (100 in / 400 out) | 69.14 | 72.37 | 1.05x |
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
