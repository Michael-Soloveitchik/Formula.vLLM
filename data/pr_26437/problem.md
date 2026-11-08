# Task PR #26437 — [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h) (#26437)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **chunk_indices_6way_cycle** in `us` (microseconds (1e-6 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **chunk_indices_6way_cycle** (headline) — `us` (microseconds (1e-6 s)), ↓ lower is better. no number reported in the PR.
- **serving_mean_tpot** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 22.89 -> 21.12 ms.
- **serving_output_throughput** — `tok/s` (tokens per second), ↑ higher is better. reported 1.595e+04 -> 1.787e+04 tok/s.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: fla @tensor_cache 4 -> 8 entries (part 2 of the PR): prepare_chunk_indices cycled over 6 distinct cu_seqlens (2 calls x 3 groups per GDN layer) - baseline LRU thrashes and recomputes (incl. .tolist() sync), PR's solution hits. Wall-clock us per call. Parts 1 (bool-mask -> index_select in qwen3_next.py) and 3 are model-level and not measured. Author e2e (Qwen3-Next-80B MTP, B200) reported-only.

## The problem (issue)
(from `solution.md`, quoted)

> Qwen3-next MTP (multi-token prediction) speculative decoding used boolean mask indexing (`tensor[bool_mask]`) in the Gated DeltaNet (GDN) attention path to split tokens into spec and non-spec groups. Boolean mask indexing requires the GPU to count the number of `True` elements to size the output tensor, which forces a device-to-host (d2h) memory transfer and a synchronization stall. These transfers made MTP slower than standard token prediction on some inputs, especially during the prefill phase.

### Issue — detailed

> In `qwen3_next.py` `_forward`, `mixed_qkv[spec_token_masks]`, `g[:, spec_token_masks]`, `beta[:, ...]`, and the `core_attn_out[:, mask] = ...` scatter all used boolean masks built in `gdn_attn.py`'s `GDNAttentionMetadata`. Each boolean index op triggers a d2h copy of the `True` count to allocate the result tensor. Additionally `@tensor_cache` in `fla/ops/utils.py` had cache_size=4, too small for the 2 calls x 3 grouping per GDN attn, causing cache thrashing.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The metadata now carries precomputed integer index tensors `spec_token_indx` and `non_spec_token_indx` (int32) instead of a single boolean `spec_token_masks`; they are built once in `GDNAttentionMetadata.build` via `torch.argsort(spec_token_masks)` splitting at `num_non_spec_tokens`. The forward path replaces boolean indexing with `index_select(dim, indx)` for `mixed_qkv`, `g`, and `beta`, and replaces the masked scatter assignment with `index_copy_`, eliminating the d2h transfers and stream syncs. `cache_size` in `@tensor_cache` is raised from 4 to 8 so the 2x3 grouped GDN calls all stay cached. Finally `rearrange_mixed_qkv` returns `query/key/value` as `.contiguous()` to enable better torch.compile fusion.

**Author's words:**

> Remove boolean indexing from GDN and replace with direct index_select/index_copy indexing to avoid the d2h transfer caused by `tensor[bool_mask]` needing the True-element count. Increase `@tensor_cache` cache size since there are 2 calls per GDN attn grouped by 3, requiring at least 2*3 entries. Make q, k, v `.contiguous()` early in GDN; not helpful now but aids torch.compile fusion.

## Reported numbers (PR author)
- gain: 12% (pct)
- evidence: Speedup of Output token throughput is **12%**
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
