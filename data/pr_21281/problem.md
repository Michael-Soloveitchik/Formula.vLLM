# Task PR #21281 — [Core] Minimize number of dict lookup in _maybe_evict_cached_block (#21281)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> `_maybe_evict_cached_block` performed redundant dictionary lookups against `self.cached_block_hash_to_block` keyed by the same `block_hash`. The original code did a membership test (`block_hash in ...`), then indexed to delete a sub-entry, then re-indexed to check length, then potentially re-indexed to delete the key — up to 4 lookups for one eviction. After PR #21222 sped up `allocate_new_block`, this redundant-lookup pattern became the new hot-path bottleneck.

### Issue — detailed

> In `vllm/v1/core/block_pool.py`, `_maybe_evict_cached_block` hashed and looked up `block_hash` in `cached_block_hash_to_block` four times: once for `in` check, once for the inner `del`, once for the `len()==0` check, once for the final `del`. Repeated lookups of the same key on a hot path waste CPU cycles.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The diff replaces the membership test + repeated indexing with early returns and a single cached reference. Before: `if block_hash and block_hash in self.cached_block_hash_to_block:` followed by multiple `self.cached_block_hash_to_block[block_hash]` accesses. After: it early-returns `False` when `block_hash is None`, then does one `self.cached_block_hash_to_block.get(block_hash)` into local `blocks_by_id` (returning `False` if `None`), then mutates via `blocks_by_id.pop(block.block_id, None)` and uses the local for the emptiness check, requiring only one additional `del self.cached_block_hash_to_block[block_hash]`. This cuts same-key dict lookups from ~4 to ~2, lowering per-call cost from 0.13ms to 0.03ms.

**Author's words:**

> Author noticed `_maybe_evict_cached_block` did duplicated key lookups against the same dict 4 times and reduced it to 2 by caching the sub-dict in a local variable and using early returns. Verified via trace comparison on top of PR #21222, showing 0.13ms -> 0.03ms improvement.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **allocate_new_block** (us/op): reported 130 -> 30 us/op. workload: `python3 benchmarks/kv_cache/benchmark_block_pool.py --num-gpu-blocks 100000 --num-iteration 1000 --allocate-blocks 10 50 100 500 1000`
- note: Builds on 21222; same block_pool microbench exercises the evict/allocate path. CPU-only.

## Reported numbers (PR author)
- gain: 77% (0.13ms→0.03ms) (unit)
- evidence: Test Result Quite significant improvements 0.13ms -> 0.03ms
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
