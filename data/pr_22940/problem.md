# Task PR #22940 — [Core] direct indexing on self.block_table_np in compute_slot_mapping (#22940)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **throughput_output_tok_s** (tok/s): reported 2.07 % gain. workload: `VLLM_USE_V1=1 python3 benchmarks/benchmark_throughput.py --model facebook/opt-125m --backend vllm --input-len 800 --output-len 75 --num-prompts 30000 --dtype float16 --max-num-seqs 1024 --max-num-batched-tokens 16384`
- **throughput_total_tok_s** (tok/s): no number reported in the PR.
- **throughput_req_s** (tok/s): no number reported in the PR.
- note: Author ran --num-prompts 30000 (opt-125m, in 800 / out 75). We run 2000 prompts per repeat so 5 repeats x 4 arms fit in a session; same per-step hot path (compute_slot_mapping runs every step), throughput ratio comparable, absolute tok/s not 1:1 (different GPU, shorter run). Weights come from the HF cache mounted at /root/.cache/huggingface (.weights/hf on the host). Args match the author: vllm 0.10.1 gives >=70 GiB GPUs max_num_seqs=1024/max_num_batched_tokens=16384 by default, we must pass them; 30000 prompts ~16 min/run so repeats=1.

## Reported numbers (PR author)
- gain: 2.07% (pct)
- evidence: Throughput improved 2.07% for opt-125m with input=800 and output=75
- reported pct = author's end-to-end throughput gain; dashboard applies it to our measured baseline

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
