# Task PR #33538 — [Kernel] Triton-based Top-k and Top-p sampler kernels (#33538)

## Objective
Edit the code under `./vllm` so that:
1. the hidden correctness tests still PASS, and
2. the workload runs **at least as fast as the PR's merged solution** — ideally faster.

The headline metric is **topk_whole_b16_v32768** in `ms` (milliseconds (1e-3 s)); minimize it (lower is better).

## How you are measured
The SAME workload is timed three times, all on this machine:
- **baseline** — the code before the PR (`base_sha`).
- **PR's solution** — the code after the PR landed (`merge_sha`); this is the bar to beat.
- **your edit** — baseline tree + your changes.

Each metric below is parsed from the workload's stdout and averaged over repeated runs.
You win a metric when your edit is faster than the PR's solution on it; equal speed ties.

### Metrics

- **topk_whole_b16_v32768** (headline) — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.28 -> 0.067 ms.
- **topp_whole_b16_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.361 -> 0.184 ms.
- **topk_topp_whole_b16_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.242 -> 0.093 ms.
- **mixed_partial_b16_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.266 -> 0.205 ms.
- **topk_whole_b1_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. reported 0.101 -> 0.065 ms.
- **topk_whole_b128_v32768** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.
- **topk_topp_whole_b2048_v131072** — `ms` (milliseconds (1e-3 s)), ↓ lower is better. no number reported in the PR.

Workload command (run per arm): `python3 benchmarks/benchmark_topk_topp.py --batch-sizes 1 16 128 2048 --vocab-sizes 32768 131072 --warmup-iters 5 --benchmark-iters 20`
Mechanism / note: Adds Triton sampler + bench script. Triton vs PyTorch sort-based. CUDA-only.

## The problem (issue)
(from `solution.md`, quoted)

> Re-opening PR #25824, with correctness and benchmark scripts from @njhill's PR #32558.
>
> Passes all correctness tests, faster overall compared to #32558 except for Top-p only cases.  Compared to #32558, this algorithm includes a truncation step, which gathers a small "outlier" subset of the logits to reduce the search space using a stochastic cutoff. The kernel uses [num_program, vocab_size] shaped buffers to gather the outlier subset, requiring around ~80MiB of extra VRAM.
>
> This implementation also uses `p_over_pivots_sum >= p AND p_over_pivots_sum - (p_over_pivots_min * num_p_over_pivots_min) < p` for its Top-p search termination condition. This condition looks for the pivot where "the sum of probabilities over the pivot is larger than p, but exclusion of the smallest probability over the pivot pushes the sum below p", which should be a more accurate Top-p condition than PR #32558. The algorithm also includes handling of duplicate logits or probabilities.
>
> Below are the execution latency and memory usage comparisons against PR #32558 and PyTorch.
> ```
> Scenario        Batch   Vocab  Ops%  Triton (ms)  PR32558 (ms)  PyTorch (ms)  Speedup  Speedup vs PR32558    Tri Mem  32558 Mem    Pyt Mem
> ------------------------------------------------------------------------------------------------------------------------------------------
> topk_whole          1   32768  100%        0.065         0.162         0.101    1.55x               2.49x  722.00 KB  592.00 KB    1.47 MB
> topk_partial        1   32768    0%        0.039         0.025         0.099    2.54x               0.64x  722.00 KB  592.00 KB    1.47 MB
> topp_whole          1   32768  100%        0.172          0.21         0.112    0.65x               1.22x  722.00 KB  592.00 KB    1.47 MB
> topp_partial        1   32768    0%        0.039         0.024         0.112    2.87x               0.62x  722.00 KB  592.00 KB    1.47 MB
> topk_topp_whole     1   32768  200%        0.089          0.35         0.134    1.51x               3.93x  722.00 KB  592.00 KB    1.47 MB
> mixed_partial       1   32768  200%        0.087          0.35         0.137    1.57x               4.02x  722.00 KB  592.00 KB    1.47 MB
> topk_whole          4   32768  100%        0.067         0.162         0.666    9.94x               2.42x    2.21 MB    1.70 MB    6.74 MB
> topk_partial        4   32768   50%         0.07         0.162          0.24    3.43x               2.31x    2.21 MB    1.70 MB    6.74 MB
> topp_whole          4   32768  100%        0.181          0.21         0.201    1.11x               1.16x    2.21 MB    1.70 MB    6.74 MB
> topp_partial        4   32768   50%        0.181         0.209         0.482    2.66x               1.15x    2.21 MB    1.70 MB    6.74 MB
> topk_topp_whole     4   32768  200%        0.098         0.351         0.311    3.17x               3.58x    2.21 MB    1.70 MB    6.74 MB
> mixed_partial       4   32768  150%        0.191          0.35         0.752    3.94x               1.83x    2.21 MB    1.70 MB    6.74 MB
> topk_whole         16   32768  100%        0.067         0.163          0.28    4.18x               2.43x    8.21 MB    6.20 MB   26.31 MB
> topk_partial       16   32768   50%        0.064         0.163         0.169    2.64x               2.55x    8.21 MB    6.20 MB   26.31 MB
> topp_whole         16   32768  100%        0.184          0.21         0.361    1.96x               1.14x    8.21 MB    6.20 MB   26.31 MB
> topp_partial       16   32768   50%        0.178          0.21         0.305    1.71x               1.18x    8.21 MB    6.20 MB   26.31 MB
> topk_topp_whole    16   32768  200%        0.093         0.351         0.242    2.60x               3.77x    8.21 MB    6.20 MB   26.31 MB
> mixed_partial      16   32768  138%        0.205         0.351         0.266    1.30x               1.71x    8.21 MB    6.20 MB   26.31 MB
> topk_whole         24   32768  100%        0.065         0.165          0.32    4.92x               2.54x   12.21 MB    9.20 MB   39.35 MB
> topk_partial       24   32768   50%        0.064         0.163         0.217    3.39x               2.55x   12.21 MB    9.20 MB   39.35 MB
> topp_whole         24   32768  100%        0.185         0.211         0.588    3.18x               1.14x   12.21 MB    9.20 MB   39.35 MB
> topp_partial       24   32768   50%        0.177          0.21         0.262    1.48x               1.19x   12.21 MB    9.20 MB   39.35 MB
> topk_topp_whole    24   32768  200%        0.092         0.351         0.438    4.76x               3.82x   12.21 MB    9.20 MB   39.35 MB
> mixed_partial      24   32768  133%        0.212         0.351         0.389    1.83x               1.66x   12.21 MB    9.20 MB   39.35 MB
> topk_whole         32   32768  100%        0.064         0.163         0.478    7.47x               2.55x   16.21 MB   12.20 MB   52.40 MB
> topk_partial       32   32768   50%        0.064         0.163         0.231    3.61x               2.55x   16.21 MB   12.20 MB   52.40 MB
> topp_whole         32   32768  100%        0.181          0.21         0.544    3.01x               1.16x   16.21 MB   12.20 MB   52.40 MB
> topp_partial       32   32768   50%        0.181          0.21         1.631    9.01x               1.16x   16.21 MB   12.20 MB   52.40 MB
> topk_topp_whole    32   32768  200%        0.089         0.351         0.467    5.25x               3.94x   16.21 MB   12.20 MB   52.40 MB
> mixed_partial      32   32768  138%        0.219         0.351         0.304    1.39x               1.60x   16.21 MB   12.20 MB   52.40 MB
> topk_whole         48   32768  100%        0.067         0.163         0.885   13.21x               2.43x   24.21 MB   18.20 MB   78.50 MB
> topk_partial       48   32768   50%         0.07         0.163         0.584    8.34x               2.33x   24.21 MB   18.20 MB   78.50 MB
> topp_whole         48   32768  100%        0.182          0.21         0.986    5.42x               1.15x   24.21 MB   18.20 MB   78.50 MB
> topp_partial       48   32768   50%        0.186         0.211          0.35    1.88x               1.13x   24.21 MB   18.20 MB   78.50 MB
> topk_topp_whole    48   32768  200%        0.096         0.352         0.481    5.01x               3.67x   24.21 MB   18.20 MB   78.50 MB
> mixed_partial      48   32768  133%        0.207         0.351         0.374    1.81x               1.70x   24.21 MB   18.20 MB   78.50 MB
> topk_whole         56   32768  100%        0.066         0.164         0.498    7.55x               2.48x   28.21 MB   21.20 MB   92.20 MB
> topk_partial       56   32768   50%        0.066         0.163         1.999   30.29x               2.47x   28.21 MB   21.20 MB   92.20 MB
> topp_whole         56   32768  100%        0.181         0.211         0.393    2.17x               1.17x   28.21 MB   21.20 MB   92.20 MB
> topp_partial       56   32768   50%        0.181         0.211         0.392    2.17x               1.17x   28.21 MB   21.20 MB   92.20 MB
> topk_topp_whole    56   32768  200%        0.092         0.351         0.594    6.46x               3.82x   28.21 MB   21.20 MB   92.20 MB
> mixed_partial      56   32768  136%         0.21         0.351         3.027   14.41x               1.67x   28.21 MB   21.20 MB   92.20 MB
> topk_whole         64   32768  100%        0.067         0.163         0.447    6.67x               2.43x   32.21 MB   24.20 MB  104.59 MB
> topk_partial       64   32768   50%        0.066         0.163         0.547    8.29x               2.47x   32.21 MB   24.20 MB  104.59 MB
> topp_whole         64   32768  100%        0.185         0.211         0.752    4.06x               1.14x   32.21 MB   24.20 MB  104.59 MB
> topp_partial       64   32768   50%        0.182          0.21         1.383    7.60x               1.15x   32.21 MB   24.20 MB  104.59 MB
> topk_topp_whole    64   32768  200%          0.1         0.352         0.782    7.82x               3.52x   32.21 MB   24.20 MB  104.59 MB
> mixed_partial      64   32768  134%        0.225         0.351         0.741    3.29x               1.56x   32.21 MB   24.20 MB  104.59 MB
> topk_whole         96   32768  100%        0.069         0.164         0.788   11.42x               2.38x   48.21 MB   36.20 MB  156.78 MB
> topk_partial       96   32768   50%        0.067         0.163         0.535    7.99x               2.43x   48.21 MB   36.20 MB  156.78 MB
> topp_whole         96   32768  100%        0.191         0.211         1.008    5.28x               1.10x   48.21 MB   36.20 MB  156.78 MB
> topp_partial       96   32768   50%        0.189         0.212         0.652    3.45x               1.12x   48.21 MB   36.20 MB  156.78 MB
> topk_topp_whole    96   32768  200%        0.103         0.352         1.488   14.45x               3.42x   48.21 MB   36.20 MB  156.78 MB
> mixed_partial      96   32768  133%        0.238         0.352         0.624    2.62x               1.48x   48.21 MB   36.20 MB  156.78 MB
> topk_whole        128   32768  100%        0.071         0.168         0.843   11.87x               2.37x   64.21 MB   48.20 MB  160.20 MB
> topk_partial      128   32768   50%        0.069         0.168          0.85   12.32x               2.43x   64.21 MB   48.20 MB  160.20 MB
> topp_whole        128   32768  100%         0.19         0.216         0.777    4.09x               1.14x   64.21 MB   48.20 MB  160.20 MB
> topp_partial      128   32768   50%         0.19         0.216          0.73    3.84x               1.14x   64.21 MB   48.20 MB  160.20 MB
> topk_topp_whole   128   32768  200%        0.102         0.357         1.075   10.54x               3.50x   64.21 MB   48.20 MB  160.20 MB
> mixed_partial     128   32768  134%        0.235         0.352         0.792    3.37x               1.50x   64.21 MB   48.20 MB  160.20 MB
> topk_whole        192   32768  100%         0.12         0.256         0.847    7.06x               2.13x   88.71 MB   72.20 MB  240.20 MB
> topk_partial      192   32768   50%        0.071         0.172         1.394   19.63x               2.42x   88.71 MB   72.20 MB  240.20 MB
> topp_whole        192   32768  100%         0.36         0.303         1.613    4.48x               0.84x   88.71 MB   72.20 MB  240.20 MB
> topp_partial      192   32768   50%         0.19          0.22         2.167   11.41x               1.16x   88.71 MB   72.20 MB  240.20 MB
> topk_topp_whole   192   32768  200%        0.172         0.537         1.113    6.47x               3.12x   88.71 MB   72.20 MB  240.20 MB
> mixed_partial     192   32768  133%         0.24         0.489         1.233    5.14x               2.04x   88.71 MB   72.20 MB  240.20 MB
> topk_whole        256   32768  100%        0.124         0.263         1.606   12.95x               2.12x  112.71 MB   96.20 MB  320.20 MB
> topk_partial      256   32768   50%        0.072         0.176         1.347   18.71x               2.44x  112.71 MB   96.20 MB  320.20 MB
> topp_whole        256   32768  100%        0.367         0.311         1.371    3.74x               0.85x  112.71 MB   96.20 MB  320.20 MB
> topp_partial      256   32768   50%        0.195         0.223         1.459    7.48x               1.14x  112.71 MB   96.20 MB  320.20 MB
> topk_topp_whole   256   32768  200%         0.18         0.542         1.219    6.77x               3.01x  112.71 MB   96.20 MB  320.20 MB
> mixed_partial     256   32768  134%        0.326           0.4          1.57    4.82x               1.23x  112.71 MB   96.20 MB  320.20 MB
> topk_whole        512   32768  100%        0.228         0.557         1.985    8.71x               2.44x  208.71 MB  192.20 MB  640.20 MB
> topk_partial      512   32768   50%        0.125         0.285         1.981   15.85x               2.28x  208.71 MB  192.20 MB  640.20 MB
> topp_whole        512   32768  100%        0.703         0.548         2.252    3.20x               0.78x  208.71 MB  192.20 MB  640.20 MB
> topp_partial      512   32768   50%        0.363         0.335         2.255    6.21x               0.92x  208.71 MB  192.20 MB  640.20 MB
> topk_topp_whole   512   32768  200%        0.333         1.048         2.572    7.72x               3.15x  208.71 MB  192.20 MB  640.20 MB
> mixed_partial     512   32768  134%        0.602         0.835         2.367    3.93x               1.39x  208.71 MB  192.20 MB  640.20 MB
> topk_whole       1024   32768  100%         0.43           0.9         3.844    8.94x               2.09x  400.71 MB  384.20 MB    1.25 GB
> topk_partial     1024   32768   50%        0.231         0.563         5.988   25.92x               2.44x  400.71 MB  384.20 MB    1.25 GB
> topp_whole       1024   32768  100%         1.38         0.794         4.236    3.07x               0.58x  400.71 MB  384.20 MB    1.25 GB
> topp_partial     1024   32768   50%        0.704         0.565         4.298    6.11x               0.80x  400.71 MB  384.20 MB    1.25 GB
> topk_topp_whole  1024   32768  200%        0.629         1.593          5.04    8.01x               2.53x  400.71 MB  384.20 MB    1.25 GB
> mixed_partial    1024   32768  133%        1.023         1.306         5.996    5.86x               1.28x  400.71 MB  384.20 MB    1.25 GB
> topk_whole          1  131072  100%        0.133         0.941         0.567    4.26x               7.08x    2.21 MB    1.70 MB    5.24 MB
> topk_partial        1  131072    0%        0.044         0.024         0.193    4.39x               0.55x    2.21 MB    1.70 MB    5.24 MB
> topp_whole          1  131072  100%        0.895         1.116         0.155    0.17x               1.25x    2.21 MB    1.70 MB    5.24 MB
> topp_partial        1  131072    0%        0.043         0.024         0.252    5.86x               0.56x    2.21 MB    1.70 MB    5.24 MB
> topk_topp_whole     1  131072  200%        0.155         2.018         0.405    2.61x              13.02x    2.21 MB    1.70 MB    5.24 MB
> mixed_partial       1  131072  200%        0.156         2.017         0.332    2.13x              12.93x    2.21 MB    1.70 MB    5.24 MB
> topk_whole          4  131072  100%        0.132         0.947         0.153    1.16x               7.17x    8.21 MB    6.20 MB   26.31 MB
> topk_partial        4  131072   50%        0.128         0.933         0.154    1.20x               7.29x    8.21 MB    6.20 MB   26.31 MB
> topp_whole          4  131072  100%        0.947          1.12          0.35    0.37x               1.18x    8.21 MB    6.20 MB   26.31 MB
> topp_partial        4  131072   50%        0.987          1.11          0.35    0.35x               1.12x    8.21 MB    6.20 MB   26.31 MB
> topk_topp_whole     4  131072  200%        0.164         2.028         0.392    2.39x              12.37x    8.21 MB    6.20 MB   26.31 MB
> mixed_partial       4  131072  150%        0.896         2.004         0.389    0.43x               2.24x    8.21 MB    6.20 MB   26.31 MB
> topk_whole         16  131072  100%        0.134         0.952         3.252   24.27x               7.10x   32.21 MB   24.20 MB  104.59 MB
> topk_partial       16  131072   50%         0.13         0.946         0.992    7.63x               7.28x   32.21 MB   24.20 MB  104.59 MB
> topp_whole         16  131072  100%         0.95         1.126         0.575    0.61x               1.19x   32.21 MB   24.20 MB  104.59 MB
> topp_partial       16  131072   50%        1.007         1.121             1    0.99x               1.11x   32.21 MB   24.20 MB  104.59 MB
> topk_topp_whole    16  131072  200%        0.173         2.037         0.611    3.53x              11.77x   32.21 MB   24.20 MB  104.59 MB
> mixed_partial      16  131072  138%        1.135         2.036         1.098    0.97x               1.79x   32.21 MB   24.20 MB  104.59 MB
> topk_whole         24  131072  100%        0.133         0.962         0.503    3.78x               7.23x   48.21 MB   36.20 MB  156.78 MB
> topk_partial       24  131072   50%        0.133         0.958         0.803    6.04x               7.20x   48.21 MB   36.20 MB  156.78 MB
> topp_whole         24  131072  100%        1.017         1.134         0.729    0.72x               1.12x   48.21 MB   36.20 MB  156.78 MB
> topp_partial       24  131072   50%        1.018         1.133         1.146    1.13x               1.11x   48.21 MB   36.20 MB  156.78 MB
> topk_topp_whole    24  131072  200%         0.17         2.056         1.917   11.28x              12.09x   48.21 MB   36.20 MB  156.78 MB
> mixed_partial      24  131072  133%        1.153         2.054         0.754    0.65x               1.78x   48.21 MB   36.20 MB  156.78 MB
> topk_whole         32  131072  100%        0.138         0.974         0.714    5.17x               7.06x   64.21 MB   48.20 MB  208.97 MB
> topk_partial       32  131072   50%        0.136         0.968         1.041    7.65x               7.12x   64.21 MB   48.20 MB  208.97 MB
> topp_whole         32  131072  100%        1.049         1.145         0.872    0.83x               1.09x   64.21 MB   48.20 MB  208.97 MB
> topp_partial       32  131072   50%        1.023         1.142         0.983    0.96x               1.12x   64.21 MB   48.20 MB  208.97 MB
> topk_topp_whole    32  131072  200%        0.177         2.077         1.462    8.26x              11.73x   64.21 MB   48.20 MB  208.97 MB
> mixed_partial      32  131072  138%        1.235         2.073         0.909    0.74x               1.68x   64.21 MB   48.20 MB  208.97 MB
> topk_whole         48  131072  100%         0.14         0.991         4.286   30.61x               7.08x   96.21 MB   72.20 MB  314.20 MB
> topk_partial       48  131072   50%        0.138          0.99         1.198    8.68x               7.17x   96.21 MB   72.20 MB  314.20 MB
> topp_whole         48  131072  100%         1.09         1.162         1.222    1.12x               1.07x   96.21 MB   72.20 MB  314.20 MB
> topp_partial       48  131072   50%        1.041         1.161         1.197    1.15x               1.12x   96.21 MB   72.20 MB  314.20 MB
> topk_topp_whole    48  131072  200%        0.179         2.095         1.228    6.86x              11.70x   96.21 MB   72.20 MB  314.20 MB
> mixed_partial      48  131072  133%         1.19         2.063         1.178    0.99x               1.73x   96.21 MB   72.20 MB  314.20 MB
> topk_whole         56  131072  100%        0.144             1         2.586   17.96x               6.94x  112.21 MB   84.20 MB  366.20 MB
> topk_partial       56  131072   50%        0.139         0.997         1.129    8.12x               7.17x  112.21 MB   84.20 MB  366.20 MB
> topp_whole         56  131072  100%        1.099         1.169         2.427    2.21x               1.06x  112.21 MB   84.20 MB  366.20 MB
> topp_partial       56  131072   50%        1.031         1.168         1.914    1.86x               1.13x  112.21 MB   84.20 MB  366.20 MB
> topk_topp_whole    56  131072  200%        0.181         2.101           2.1   11.60x              11.61x  112.21 MB   84.20 MB  366.20 MB
> mixed_partial      56  131072  136%        1.293         2.072         1.362    1.05x               1.60x  112.21 MB   84.20 MB  366.20 MB
> topk_whole         64  131072  100%        0.148         1.034         1.136    7.68x               6.99x  128.21 MB   96.20 MB  418.20 MB
> topk_partial       64  131072   50%        0.141         1.003         1.133    8.04x               7.11x  128.21 MB   96.20 MB  418.20 MB
> topp_whole         64  131072  100%        1.115         1.202         1.688    1.51x               1.08x  128.21 MB   96.20 MB  418.20 MB
> topp_partial       64  131072   50%        1.047         1.173         1.401    1.34x               1.12x  128.21 MB   96.20 MB  418.20 MB
> topk_topp_whole    64  131072  200%        0.187         2.121         1.854    9.91x              11.34x  128.21 MB   96.20 MB  418.20 MB
> mixed_partial      64  131072  134%        1.311         2.098         2.007    1.53x               1.60x  128.21 MB   96.20 MB  418.20 MB
> topk_whole         96  131072  100%        0.158         1.642         4.722   29.89x              10.39x  192.21 MB  144.20 MB  626.50 MB
> topk_partial       96  131072   50%        0.144         1.011         1.895   13.16x               7.02x  192.21 MB  144.20 MB  626.50 MB
> topp_whole         96  131072  100%        1.505         1.717         2.048    1.36x               1.14x  192.21 MB  144.20 MB  626.50 MB
> topp_partial       96  131072   50%        1.099         1.181         1.916    1.74x               1.07x  192.21 MB  144.20 MB  626.50 MB
> topk_topp_whole    96  131072  200%        0.195         3.285         3.479   17.84x              16.85x  192.21 MB  144.20 MB  626.50 MB
> mixed_partial      96  131072  133%        1.407         2.754          4.44    3.16x               1.96x  192.21 MB  144.20 MB  626.50 MB
> topk_whole        128  131072  100%        0.165         1.661         2.609   15.81x              10.07x  256.21 MB  192.20 MB  640.20 MB
> topk_partial      128  131072   50%        0.149         1.153         2.388   16.03x               7.74x  256.21 MB  192.20 MB  640.20 MB
> topp_whole        128  131072  100%         1.85         1.732         2.903    1.57x               0.94x  256.21 MB  192.20 MB  640.20 MB
> topp_partial      128  131072   50%        1.213          1.32          2.73    2.25x               1.09x  256.21 MB  192.20 MB  640.20 MB
> topk_topp_whole   128  131072  200%        0.206         3.315         2.982   14.48x              16.09x  256.21 MB  192.20 MB  640.20 MB
> mixed_partial     128  131072  134%        1.429         2.787         2.812    1.97x               1.95x  256.21 MB  192.20 MB  640.20 MB
> topk_whole        192  131072  100%        0.298          1.72         3.142   10.54x               5.77x  354.21 MB  288.20 MB  960.20 MB
> topk_partial      192  131072   50%        0.159         1.643         5.631   35.42x              10.33x  354.21 MB  288.20 MB  960.20 MB
> topp_whole        192  131072  100%        2.946         1.797         3.537    1.20x               0.61x  354.21 MB  288.20 MB  960.20 MB
> topp_partial      192  131072   50%         1.58         1.718         3.874    2.45x               1.09x  354.21 MB  288.20 MB  960.20 MB
> topk_topp_whole   192  131072  200%        0.375         3.423         3.791   10.11x               9.13x  354.21 MB  288.20 MB  960.20 MB
> mixed_partial     192  131072  133%        1.597         2.874         3.695    2.31x               1.80x  354.21 MB  288.20 MB  960.20 MB
> topk_whole        256  131072  100%        0.314         1.778         4.156   13.24x               5.66x  450.21 MB  384.20 MB    1.25 GB
> topk_partial      256  131072   50%        0.166          1.66         6.882   41.46x              10.00x  450.21 MB  384.20 MB    1.25 GB
> topp_whole        256  131072  100%         3.55         1.836         4.341    1.22x               0.52x  450.21 MB  384.20 MB    1.25 GB
> topp_partial      256  131072   50%        1.861         1.733         4.244    2.28x               0.93x  450.21 MB  384.20 MB    1.25 GB
> topk_topp_whole   256  131072  200%        0.395         3.502         8.431   21.34x               8.87x  450.21 MB  384.20 MB    1.25 GB
> mixed_partial     256  131072  134%        2.241         3.399         5.953    2.66x               1.52x  450.21 MB  384.20 MB    1.25 GB
> topk_whole        512  131072  100%        0.604         2.177         7.488   12.40x               3.60x  834.21 MB  768.20 MB    2.50 GB
> topk_partial      512  131072   50%         0.31         1.777         8.979   28.96x               5.73x  834.21 MB  768.20 MB    2.50 GB
> topp_whole        512  131072  100%        6.894         2.155         8.787    1.27x               0.31x  834.21 MB  768.20 MB    2.50 GB
> topp_partial      512  131072   50%        3.552         1.864         8.343    2.35x               0.52x  834.21 MB  768.20 MB    2.50 GB
> topk_topp_whole   512  131072  200%         0.76         4.142        10.756   14.15x               5.45x  834.21 MB  768.20 MB    2.50 GB
> mixed_partial     512  131072  134%        4.129          3.83         9.383    2.27x               0.93x  834.21 MB  768.20 MB    2.50 GB
> topk_whole       1024  131072  100%        1.177          3.58        14.857   12.62x               3.04x    1.56 GB    1.50 GB    5.00 GB
> topk_partial     1024  131072   50%        0.602         2.202        14.997   24.91x               3.66x    1.56 GB    1.50 GB    5.00 GB
> topp_whole       1024  131072  100%       13.691         3.172        16.312    1.19x               0.23x    1.56 GB    1.50 GB    5.00 GB
> topp_partial     1024  131072   50%        6.933         2.243        16.313    2.35x               0.32x    1.56 GB    1.50 GB    5.00 GB
> topk_topp_whole  1024  131072  200%        1.488         6.381        17.452   11.73x               4.29x    1.56 GB    1.50 GB    5.00 GB
> mixed_partial    1024  131072  133%        7.268         5.226        17.183    2.36x               0.72x    1.56 GB    1.50 GB    5.00 GB
> ```

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.
>
> - Original PR: https://github.com/vllm-project/vllm/pull/33538
> - Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/33538.diff
> - Retrieved at: 2026-11-03T13:50:46.346007+00:00
> - Diff SHA-256: `c84ec01b8ea42d5a0a2655a0dbb2ba297b07840ec640f348b6007dad6643065a`
>
> Existing semantic notes in [33538.sem.json](./33538.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Reported numbers (PR author)
- gain: up to 9.94x vs PyTorch topk sampler (batch 4, vocab 32768) (unit)
- evidence: faster triton topk/topp sampler kernels
- reported = PR author's table (ms), same scenario key as benchmark_topk_topp.py SUMMARY TABLE: <scenario>_b<batch>_v<vocab>

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Beating the PR's solution on the headline metric wins; equal speed ties it.
