# Task PR #23928 — [Model loader]: support multi-thread model weight loading (#23928)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **weights_load_1gb_s** in `s` (seconds); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **weights_load_1gb_s** (headline) — `s` (seconds), ↓ lower is better. no number reported in the PR.
- **model_load_time** — `s` (seconds), ↓ lower is better. reported 20 -> 14 s.
- **multi_thread_weight_loading** — `s` (seconds), ↓ lower is better. reported 20 -> 14 s. Diff adds multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator (concurrent.futures.ThreadPoolExecutor over shard files) in weight_utils.py, and DefaultModelLoader selects them when enable_multithread_load is set (num_threads default 8); also validates extra-config keys. Number source: PR body Test Result table - disable 20 s / 3.1 GB/s vs enable 14 s / 4.42 GB/s, 'Load Weight speed up 1.43x'.

Workload command (run per arm): `python3 /task/bench.py`
Mechanism / note: Weight-shard iteration over 8 synthetic 128 MB bf16 safetensors files (1 GB, written to a temp dir, page-cached): PR's solution = multi_thread_safetensors_weights_iterator (8 workers), baseline tree lacks it so the bench runs the single-thread safetensors_weights_iterator the loader uses there. Seconds per full pass, best of 3. Author: QwQ-32B 62 GB load 20 s -> 14 s (reported-only). Opt-in feature (enable_multithread_load).

## The problem (issue)
(from `solution.md`, quoted)

> DefaultModelLoader loads model weight files sequentially, one safetensor or pt/bin shard at a time, via single-threaded iterators. Each file is read and deserialized to CPU before the next starts. Disk/NVMe bandwidth stays underutilized because a single thread cannot saturate I/O. Model initialization time scales linearly with shard count instead of parallel throughput.

### Issue — detailed

> _get_weights_iterator in default_loader.py calls safetensors_weights_iterator / pt_weights_iterator in weight_utils.py, which loop over hf_weights_files sequentially — each load_file/torch.load blocks until done. No concurrency = one shard read at a time = bandwidth-bound serial path.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator to weight_utils.py, which submit each shard's load_file/torch.load to a concurrent.futures.ThreadPoolExecutor and yield state dicts via as_completed. DefaultModelLoader._get_weights_iterator reads model_loader_extra_config: when enable_multithread_load is set, it picks the multi-threaded iterator with max_workers=num_threads (default 8), else keeps the old sequential path. The __init__ previously rejected any extra config outright; now it validates keys against {enable_multithread_load, num_threads} and only raises on unexpected keys. Effect: parallel shard reads saturate NVMe bandwidth — QwQ-32B load dropped 20s→14s (3.1→4.42 GB/s, 1.43x).

**Author's words:**

> Author ports Sglang PR #7277 logic: add multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator using Python ThreadPoolExecutor to parallelize loading safetensor and pt/bin files. DefaultModelLoader uses these when enable_multithread_load extra config is set, with configurable num_threads (default 8). Also adds validation of extra config keys, allowing only enable_multithread_load and num_threads and raising on unexpected keys.

## Reported numbers (PR author)
- gain: 30% (Load Time (s): 20→14) (unit)
- evidence: Load Time (s): before 20 / after 14
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
