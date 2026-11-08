# Task PR #22940 — [Core] direct indexing on self.block_table_np in compute_slot_mapping (#22940)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **throughput_output_tok_s** in `tok/s` (tokens per second); maximize it (higher is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **throughput_output_tok_s** (headline) — `tok/s` (tokens per second), ↑ higher is better. reported 2.07 % gain.
- **throughput_total_tok_s** — `tok/s` (tokens per second), ↑ higher is better. no number reported in the PR.
- **throughput_req_s** — `tok/s` (tokens per second), ↑ higher is better. no number reported in the PR.

Workload command (run per arm): `VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --backend vllm --input-len 800 --output-len 75 --num-prompts 30000 --dtype float16 --max-num-seqs 1024 --max-num-batched-tokens 16384`
Mechanism / note: Author ran --num-prompts 30000 (opt-125m, in 800 / out 75). We run 2000 prompts per repeat so 5 repeats x 4 arms fit in a session; same per-step hot path (compute_slot_mapping runs every step), throughput ratio comparable, absolute tok/s not 1:1 (different GPU, shorter run). Weights come from the HF cache mounted at /root/.cache/huggingface (.weights/hf on the host). Args match the author: vllm 0.10.1 gives >=70 GiB GPUs max_num_seqs=1024/max_num_batched_tokens=16384 by default, we must pass them; 30000 prompts ~16 min/run so repeats=1.

## The problem (issue)
(from `solution.md`, quoted)

> In the V1 worker's `compute_slot_mapping`, block numbers were looked up by first fetching the CPU-side block table as a Torch tensor, calling `.flatten()` on it, fancy-indexing with `block_table_indices`, then converting the result back to NumPy with `.numpy()`. `torch.Tensor.flatten()` forces a full copy of the block-table tensor on every call, and the Torch↔NumPy round-trip adds extra conversion overhead, making a per-step hot-path operation needlessly expensive.

### Issue — detailed

> `block_table.py:compute_slot_mapping` built `block_numbers` via `self.get_cpu_tensor().flatten()[block_table_indices].numpy()`. `flatten()` copies the whole block-table tensor and `.numpy()` adds a conversion step, both redundant since a NumPy view of the same buffer already exists.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Diff drops the Torch path entirely. Before: `block_table_cpu = self.get_cpu_tensor(); block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()`. After: `block_numbers = self.block_table_np.ravel()[block_table_indices]`. `np.ravel()` returns a view (no copy) of the existing NumPy block table, and indexing stays in NumPy so the Torch flatten-copy and `.numpy()` conversion are eliminated. Cuts `compute_slot_mapping` from 400+μs to 15-30μs and lifts opt-125m throughput ~2.07%.

**Author's words:**

> Author replaced Torch tensor flattening and conversion with direct NumPy indexing via `ravel` on `self.block_table_np`, eliminating redundant copies and framework conversions in slot-mapping computation.

## Reported numbers (PR author)
- gain: 2.07% (pct)
- evidence: Throughput improved 2.07% for opt-125m with input=800 and output=75
- reported pct = author's end-to-end throughput gain; dashboard applies it to our measured baseline

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
