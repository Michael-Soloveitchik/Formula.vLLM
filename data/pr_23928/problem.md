# Task PR #23928 — [Model loader]: support multi-thread model weight loading (#23928)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

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

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **weights_load_1gb_s** (s): no number reported in the PR. workload: `python3 /task/bench.py`
- **model_load_time** (s): reported 20 -> 14 s.
- **multi_thread_weight_loading** (s): reported 20 -> 14 s. Diff adds multi_thread_safetensors_weights_iterator and multi_thread_pt_weights_iterator (concurrent.futures.ThreadPoolExecutor over shard files) in weight_utils.py, and DefaultModelLoader selects them when enable_multithread_load is set (num_threads default 8); also validates extra-config keys. Number source: PR body Test Result table - disable 20 s / 3.1 GB/s vs enable 14 s / 4.42 GB/s, 'Load Weight speed up 1.43x'.
- note: Weight-shard iteration over 8 synthetic 128 MB bf16 safetensors files (1 GB, written to a temp dir, page-cached): PR's solution = multi_thread_safetensors_weights_iterator (8 workers), baseline tree lacks it so the bench runs the single-thread safetensors_weights_iterator the loader uses there. Seconds per full pass, best of 3. Author: QwQ-32B 62 GB load 20 s -> 14 s (reported-only). Opt-in feature (enable_multithread_load).

## Reported numbers (PR author)
- gain: 30% (Load Time (s): 20→14) (unit)
- evidence: Load Time (s): before 20 / after 14
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
