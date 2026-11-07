# Task PR #26437 — [PERF] Qwen3-next MTP speedup (change bool mask indexing to index_select / index_copy to reduce d2h) (#26437)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **chunk_indices_6way_cycle** (us): no number reported in the PR. workload: `python3 /task/bench.py`
- **serving_mean_tpot** (ms): reported 22.89 -> 21.12 ms.
- **serving_output_throughput** (tok/s): reported 1.595e+04 -> 1.787e+04 tok/s.
- note: fla @tensor_cache 4 -> 8 entries (part 2 of the PR): prepare_chunk_indices cycled over 6 distinct cu_seqlens (2 calls x 3 groups per GDN layer) - baseline LRU thrashes and recomputes (incl. .tolist() sync), PR's solution hits. Wall-clock us per call. Parts 1 (bool-mask -> index_select in qwen3_next.py) and 3 are model-level and not measured. Author e2e (Qwen3-Next-80B MTP, B200) reported-only.

## Reported numbers (PR author)
- gain: 12% (pct)
- evidence: Speedup of Output token throughput is **12%**
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
