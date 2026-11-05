# PR #40891: [Core] Avoid using extra thread in `UniProcExecutor`

- field: scheduling  | area: v1/executor
- author: njhill
- state: MERGED  created: 2026-04-26T00:37:17Z  merged: 2026-05-07T22:33:00Z
- url: https://github.com/vllm-project/vllm/pull/40891
- labels: ready, v1

## Body

@zhuohan123's idea

Clear performance benefit, especially in low latency / high concurrency case.

### Benchmark

`--tensor-parallel-size 1 --distributed-executor-backend uni` on a single NVIDIA GB200 GPU.
Model: `Qwen/Qwen3-0.6B`. Each side is mean ± population std across **3 timed runs** sharing one server process; each run uses its own seed (1, 2, 3) and is preceded by a fresh warmup batch.
Δ = relative change of with-mean vs. without-mean (✓ = improvement, ✗ = regression).
256 in / 2048 out; ignore-eos, no prefix cache.

  | Metric | Concurrency 1 (32 prompts) | | | Concurrency 512 (1024 prompts) | | |
  |---|---:|---:|---:|---:|---:|---:|
  | | Without | With | Δ | Without | With | Δ |
  | Output throughput (tok/s) | 631.21 ±0.64 | 648.73 ±14.60 | +2.78% ✓ | 17221.96 ±168.39 | 18974.27 ±176.49 | +10.17% ✓ |
  | P50 TTFT (ms) | 19.00 ±0.16 | 19.21 ±0.19 | +1.10% ✗ | 910.12 ±36.45 | 851.30 ±69.16 | -6.46% ✓ |
  | P90 TTFT (ms) | 19.48 ±0.27 | 19.88 ±0.32 | +2.05% ✗ | 1309.86 ±49.79 | 1113.85 ±85.18 | -14.96% ✓ |
  | P50 TPOT (ms) | 1.572 ±0.001 | 1.530 ±0.032 | -2.66% ✓ | 28.946 ±0.199 | 26.383 ±0.352 | -8.85% ✓ |
  | P90 TPOT (ms) | 1.588 ±0.005 | 1.558 ±0.033 | -1.88% ✓ | 29.512 ±0.214 | 26.935 ±0.070 | -8.73% ✓ |
  | Mean ITL (ms) | 1.576 ±0.002 | 1.534 ±0.035 | -2.68% ✓ | 29.112 ±0.099 | 26.563 ±0.243 | -8.75% ✓ |
  | Mean E2EL (ms) | 3244.52 ±3.29 | 3158.44 ±71.05 | -2.65% ✓ | 60292.61 ±232.44 | 55010.05 ±510.73 | -8.76% ✓ |



## Comments

(no discussion comments)