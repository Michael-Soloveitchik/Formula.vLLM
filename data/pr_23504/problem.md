# Task PR #23504 — [Perf] Add Triton config for DeepSeek V3 FP8 EP32 H200 (#23504)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> vLLM ships per-shape Triton kernel autotuning configs for FP8 W8A8 block-quantized GEMMs and fused MoE kernels, keyed by matrix dims, device name, dtype, and block_shape. When a config file for a given (N,K)/(E,N) shape on a specific GPU is missing, the kernel falls back to a generic default tiling. That default is not tuned for the actual problem geometry, so the block-FP8 matmul and fused MoE kernels run with suboptimal BLOCK_SIZE/GROUP_SIZE/num_warps/num_stages, leaving GPU throughput on the table.

### Issue — detailed

> fp8_utils.py (block-FP8 matmul config loader, warning at line 581) and fused_moe.py (MoE config loader, warning at line 727) look up a JSON config file by shape+device+dtype+block_shape; a missing file triggers the 'Using default ... kernel config. Performance might be sub-optimal!' path, applying untuned Triton launch params instead of hardware-specific ones.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Adds two new tuned Triton config JSON files for H200: a fused-MoE config at E=8,N=2048,dtype=fp8_w8a8,block_shape=[128,128] with per-M-bucket tilings (M=1..8192), and a block-FP8 GEMM config at N=2112,K=7168,dtype=fp8_w8a8,block_shape=[128,128] with entries for M=2048/3072/4096. With these files present, the config loaders in fp8_utils.py and fused_moe.py find a match and apply the tuned BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, and num_stages instead of the generic default, silencing the sub-optimal-config warnings. Also adds a README documenting that configs are generated via benchmarks/kernels/ scripts, and fixes the tqdm import in benchmark_w8a8_block_fp8.py (from `import tqdm` to `from tqdm import tqdm`).

**Author's words:**

> Add the optimal tuned Triton config for DeepSeek V3 under the DP8TP4EP32 deployment on H200, which resolves the two missing-config warnings (block-FP8 N=2112,K=7168 and fused MoE E=8,N=2048). Validated by benchmarking DeepSeek V3-0324 serving at max-concurrency 2048, 4096 prompts, 2048/1024 input/output len; tuned vs untuned showed comparable-to-slightly-better throughput (16872 vs 16776 tok/s total) and TPOT.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **serving_throughput** (tok/s): reported 1.678e+04 -> 1.687e+04 tok/s. workload: `python3 benchmarks/kernels/benchmark_w8a8_block_fp8.py --tp-size 8 --input-type fp8 --out-dtype float16 --block-n 128 --block-k 128`
- **w8a8_kernel** (us): no number reported in the PR.
- note: W8A8 block FP8 Triton kernel tuning; GPU-only (H200 config).

## Reported numbers (PR author)
- gain: 1.6x (pct)
- evidence: The initial throughput number looked too good (1.6x speedup with tuning).

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
