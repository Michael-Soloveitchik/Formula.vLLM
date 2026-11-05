# Task PR #41163 — [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark (#41163)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (gold)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
The token-wise "all" pooler returns, for each request in a sub-group of the batch, the slice of the hidden-states tensor that belongs to that request. It builds the list by first copying two index tensors from the GPU to the host (one device sync each) and then slicing the big tensor once per request in a Python loop. With many short requests per batch, these syncs and per-request slice launches dominate the method's time.

### Issue — detailed
In `vllm/model_executor/layers/pooler/tokwise/methods.py`, `AllPool.forward` computes `hidden_states_lst` with a list comprehension that zips `pooling_cursor.first_token_indices_gpu.tolist()` and `pooling_cursor.last_token_indices_gpu.tolist()` and slices `hidden_states[first:last+1]` for each request. The two `.tolist()` calls each force a GPU→CPU transfer and synchronization, and the comprehension issues N separate slice operations. The cursor already knows the sub-group's contiguous token range and the per-request token counts on the CPU (`num_scheduled_tokens_cpu`), so none of that device traffic is necessary. Author's benchmark: 64 requests × 128 tokens, pooling a 32-request sub-group (offset 16), hidden size 4096, bf16 — median `AllPool.forward` 59.89 µs.

## Expert solution (reference — the merged PR)
**General:** Slice the sub-group's contiguous token range out of `hidden_states` once, then split that single slice by the per-request token counts that are already on the CPU. No GPU→CPU index copies, one slice plus one `split` instead of N slices.

**Detailed:** `AllPool.forward` now takes `split_sizes = pooling_cursor.num_scheduled_tokens_cpu.tolist()` (CPU tensor, no device sync), computes the sub-group's start/end token offsets from the cursor, slices `hidden_states[start:end]` once, and calls `.split(split_sizes)` to produce the per-request views. The result list is identical to the old per-request slicing. Only `methods.py` changes.

**Author's words:** Optimize `AllPool.forward` by slicing first — 51% faster in the method-level benchmark: median 59.89 µs → 39.65 µs. Accuracy covered by existing CI.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (gold). Your
edit is timed on the SAME workload. Beat gold on these metrics:
- **allpool_forward_median** (µs): reported 59.89 -> 39.65 µs. workload: the author's method-level benchmark (`tasks/pr_41163/bench.py`): 64 requests × 128 tokens, sub-group of 32, hidden 4096, bf16, median of 5 × 4000 iterations.

## Reported numbers (PR author)
- metric: AllPool.forward median latency (method-level benchmark)
- value: 59.89 -> 39.65
- unit: us

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than gold on the measured experiments wins. Same speed = ties gold.
