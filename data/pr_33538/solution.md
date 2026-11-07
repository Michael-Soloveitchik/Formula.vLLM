# PR #33538: [Kernel] Triton-based Top-k and Top-p sampler kernels (#33538)

# Issue

## Original PR description (captured source)

Re-opening PR #25824, with correctness and benchmark scripts from @njhill's PR #32558.

Passes all correctness tests, faster overall compared to #32558 except for Top-p only cases.  Compared to #32558, this algorithm includes a truncation step, which gathers a small "outlier" subset of the logits to reduce the search space using a stochastic cutoff. The kernel uses [num_program, vocab_size] shaped buffers to gather the outlier subset, requiring around ~80MiB of extra VRAM.

This implementation also uses `p_over_pivots_sum >= p AND p_over_pivots_sum - (p_over_pivots_min * num_p_over_pivots_min) < p` for its Top-p search termination condition. This condition looks for the pivot where "the sum of probabilities over the pivot is larger than p, but exclusion of the smallest probability over the pivot pushes the sum below p", which should be a more accurate Top-p condition than PR #32558. The algorithm also includes handling of duplicate logits or probabilities.

Below are the execution latency and memory usage comparisons against PR #32558 and PyTorch.
```
Scenario        Batch   Vocab  Ops%  Triton (ms)  PR32558 (ms)  PyTorch (ms)  Speedup  Speedup vs PR32558    Tri Mem  32558 Mem    Pyt Mem
------------------------------------------------------------------------------------------------------------------------------------------
topk_whole          1   32768  100%        0.065         0.162         0.101    1.55x               2.49x  722.00 KB  592.00 KB    1.47 MB
topk_partial        1   32768    0%        0.039         0.025         0.099    2.54x               0.64x  722.00 KB  592.00 KB    1.47 MB
topp_whole          1   32768  100%        0.172          0.21         0.112    0.65x               1.22x  722.00 KB  592.00 KB    1.47 MB
topp_partial        1   32768    0%        0.039         0.024         0.112    2.87x               0.62x  722.00 KB  592.00 KB    1.47 MB
topk_topp_whole     1   32768  200%        0.089          0.35         0.134    1.51x               3.93x  722.00 KB  592.00 KB    1.47 MB
mixed_partial       1   32768  200%        0.087          0.35         0.137    1.57x               4.02x  722.00 KB  592.00 KB    1.47 MB
topk_whole          4   32768  100%        0.067         0.162         0.666    9.94x               2.42x    2.21 MB    1.70 MB    6.74 MB
topk_partial        4   32768   50%         0.07         0.162          0.24    3.43x               2.31x    2.21 MB    1.70 MB    6.74 MB
topp_whole          4   32768  100%        0.181          0.21         0.201    1.11x               1.16x    2.21 MB    1.70 MB    6.74 MB
topp_partial        4   32768   50%        0.181         0.209         0.482    2.66x               1.15x    2.21 MB    1.70 MB    6.74 MB
topk_topp_whole     4   32768  200%        0.098         0.351         0.311    3.17x               3.58x    2.21 MB    1.70 MB    6.74 MB
mixed_partial       4   32768  150%        0.191          0.35         0.752    3.94x               1.83x    2.21 MB    1.70 MB    6.74 MB
topk_whole         16   32768  100%        0.067         0.163          0.28    4.18x               2.43x    8.21 MB    6.20 MB   26.31 MB
topk_partial       16   32768   50%        0.064         0.163         0.169    2.64x               2.55x    8.21 MB    6.20 MB   26.31 MB
topp_whole         16   32768  100%        0.184          0.21         0.361    1.96x               1.14x    8.21 MB    6.20 MB   26.31 MB
topp_partial       16   32768   50%        0.178          0.21         0.305    1.71x               1.18x    8.21 MB    6.20 MB   26.31 MB
topk_topp_whole    16   32768  200%        0.093         0.351         0.242    2.60x               3.77x    8.21 MB    6.20 MB   26.31 MB
mixed_partial      16   32768  138%        0.205         0.351         0.266    1.30x               1.71x    8.21 MB    6.20 MB   26.31 MB
topk_whole         24   32768  100%        0.065         0.165          0.32    4.92x               2.54x   12.21 MB    9.20 MB   39.35 MB
topk_partial       24   32768   50%        0.064         0.163         0.217    3.39x               2.55x   12.21 MB    9.20 MB   39.35 MB
topp_whole         24   32768  100%        0.185         0.211         0.588    3.18x               1.14x   12.21 MB    9.20 MB   39.35 MB
topp_partial       24   32768   50%        0.177          0.21         0.262    1.48x               1.19x   12.21 MB    9.20 MB   39.35 MB
topk_topp_whole    24   32768  200%        0.092         0.351         0.438    4.76x               3.82x   12.21 MB    9.20 MB   39.35 MB
mixed_partial      24   32768  133%        0.212         0.351         0.389    1.83x               1.66x   12.21 MB    9.20 MB   39.35 MB
topk_whole         32   32768  100%        0.064         0.163         0.478    7.47x               2.55x   16.21 MB   12.20 MB   52.40 MB
topk_partial       32   32768   50%        0.064         0.163         0.231    3.61x               2.55x   16.21 MB   12.20 MB   52.40 MB
topp_whole         32   32768  100%        0.181          0.21         0.544    3.01x               1.16x   16.21 MB   12.20 MB   52.40 MB
topp_partial       32   32768   50%        0.181          0.21         1.631    9.01x               1.16x   16.21 MB   12.20 MB   52.40 MB
topk_topp_whole    32   32768  200%        0.089         0.351         0.467    5.25x               3.94x   16.21 MB   12.20 MB   52.40 MB
mixed_partial      32   32768  138%        0.219         0.351         0.304    1.39x               1.60x   16.21 MB   12.20 MB   52.40 MB
topk_whole         48   32768  100%        0.067         0.163         0.885   13.21x               2.43x   24.21 MB   18.20 MB   78.50 MB
topk_partial       48   32768   50%         0.07         0.163         0.584    8.34x               2.33x   24.21 MB   18.20 MB   78.50 MB
topp_whole         48   32768  100%        0.182          0.21         0.986    5.42x               1.15x   24.21 MB   18.20 MB   78.50 MB
topp_partial       48   32768   50%        0.186         0.211          0.35    1.88x               1.13x   24.21 MB   18.20 MB   78.50 MB
topk_topp_whole    48   32768  200%        0.096         0.352         0.481    5.01x               3.67x   24.21 MB   18.20 MB   78.50 MB
mixed_partial      48   32768  133%        0.207         0.351         0.374    1.81x               1.70x   24.21 MB   18.20 MB   78.50 MB
topk_whole         56   32768  100%        0.066         0.164         0.498    7.55x               2.48x   28.21 MB   21.20 MB   92.20 MB
topk_partial       56   32768   50%        0.066         0.163         1.999   30.29x               2.47x   28.21 MB   21.20 MB   92.20 MB
topp_whole         56   32768  100%        0.181         0.211         0.393    2.17x               1.17x   28.21 MB   21.20 MB   92.20 MB
topp_partial       56   32768   50%        0.181         0.211         0.392    2.17x               1.17x   28.21 MB   21.20 MB   92.20 MB
topk_topp_whole    56   32768  200%        0.092         0.351         0.594    6.46x               3.82x   28.21 MB   21.20 MB   92.20 MB
mixed_partial      56   32768  136%         0.21         0.351         3.027   14.41x               1.67x   28.21 MB   21.20 MB   92.20 MB
topk_whole         64   32768  100%        0.067         0.163         0.447    6.67x               2.43x   32.21 MB   24.20 MB  104.59 MB
topk_partial       64   32768   50%        0.066         0.163         0.547    8.29x               2.47x   32.21 MB   24.20 MB  104.59 MB
topp_whole         64   32768  100%        0.185         0.211         0.752    4.06x               1.14x   32.21 MB   24.20 MB  104.59 MB
topp_partial       64   32768   50%        0.182          0.21         1.383    7.60x               1.15x   32.21 MB   24.20 MB  104.59 MB
topk_topp_whole    64   32768  200%          0.1         0.352         0.782    7.82x               3.52x   32.21 MB   24.20 MB  104.59 MB
mixed_partial      64   32768  134%        0.225         0.351         0.741    3.29x               1.56x   32.21 MB   24.20 MB  104.59 MB
topk_whole         96   32768  100%        0.069         0.164         0.788   11.42x               2.38x   48.21 MB   36.20 MB  156.78 MB
topk_partial       96   32768   50%        0.067         0.163         0.535    7.99x               2.43x   48.21 MB   36.20 MB  156.78 MB
topp_whole         96   32768  100%        0.191         0.211         1.008    5.28x               1.10x   48.21 MB   36.20 MB  156.78 MB
topp_partial       96   32768   50%        0.189         0.212         0.652    3.45x               1.12x   48.21 MB   36.20 MB  156.78 MB
topk_topp_whole    96   32768  200%        0.103         0.352         1.488   14.45x               3.42x   48.21 MB   36.20 MB  156.78 MB
mixed_partial      96   32768  133%        0.238         0.352         0.624    2.62x               1.48x   48.21 MB   36.20 MB  156.78 MB
topk_whole        128   32768  100%        0.071         0.168         0.843   11.87x               2.37x   64.21 MB   48.20 MB  160.20 MB
topk_partial      128   32768   50%        0.069         0.168          0.85   12.32x               2.43x   64.21 MB   48.20 MB  160.20 MB
topp_whole        128   32768  100%         0.19         0.216         0.777    4.09x               1.14x   64.21 MB   48.20 MB  160.20 MB
topp_partial      128   32768   50%         0.19         0.216          0.73    3.84x               1.14x   64.21 MB   48.20 MB  160.20 MB
topk_topp_whole   128   32768  200%        0.102         0.357         1.075   10.54x               3.50x   64.21 MB   48.20 MB  160.20 MB
mixed_partial     128   32768  134%        0.235         0.352         0.792    3.37x               1.50x   64.21 MB   48.20 MB  160.20 MB
topk_whole        192   32768  100%         0.12         0.256         0.847    7.06x               2.13x   88.71 MB   72.20 MB  240.20 MB
topk_partial      192   32768   50%        0.071         0.172         1.394   19.63x               2.42x   88.71 MB   72.20 MB  240.20 MB
topp_whole        192   32768  100%         0.36         0.303         1.613    4.48x               0.84x   88.71 MB   72.20 MB  240.20 MB
topp_partial      192   32768   50%         0.19          0.22         2.167   11.41x               1.16x   88.71 MB   72.20 MB  240.20 MB
topk_topp_whole   192   32768  200%        0.172         0.537         1.113    6.47x               3.12x   88.71 MB   72.20 MB  240.20 MB
mixed_partial     192   32768  133%         0.24         0.489         1.233    5.14x               2.04x   88.71 MB   72.20 MB  240.20 MB
topk_whole        256   32768  100%        0.124         0.263         1.606   12.95x               2.12x  112.71 MB   96.20 MB  320.20 MB
topk_partial      256   32768   50%        0.072         0.176         1.347   18.71x               2.44x  112.71 MB   96.20 MB  320.20 MB
topp_whole        256   32768  100%        0.367         0.311         1.371    3.74x               0.85x  112.71 MB   96.20 MB  320.20 MB
topp_partial      256   32768   50%        0.195         0.223         1.459    7.48x               1.14x  112.71 MB   96.20 MB  320.20 MB
topk_topp_whole   256   32768  200%         0.18         0.542         1.219    6.77x               3.01x  112.71 MB   96.20 MB  320.20 MB
mixed_partial     256   32768  134%        0.326           0.4          1.57    4.82x               1.23x  112.71 MB   96.20 MB  320.20 MB
topk_whole        512   32768  100%        0.228         0.557         1.985    8.71x               2.44x  208.71 MB  192.20 MB  640.20 MB
topk_partial      512   32768   50%        0.125         0.285         1.981   15.85x               2.28x  208.71 MB  192.20 MB  640.20 MB
topp_whole        512   32768  100%        0.703         0.548         2.252    3.20x               0.78x  208.71 MB  192.20 MB  640.20 MB
topp_partial      512   32768   50%        0.363         0.335         2.255    6.21x               0.92x  208.71 MB  192.20 MB  640.20 MB
topk_topp_whole   512   32768  200%        0.333         1.048         2.572    7.72x               3.15x  208.71 MB  192.20 MB  640.20 MB
mixed_partial     512   32768  134%        0.602         0.835         2.367    3.93x               1.39x  208.71 MB  192.20 MB  640.20 MB
topk_whole       1024   32768  100%         0.43           0.9         3.844    8.94x               2.09x  400.71 MB  384.20 MB    1.25 GB
topk_partial     1024   32768   50%        0.231         0.563         5.988   25.92x               2.44x  400.71 MB  384.20 MB    1.25 GB
topp_whole       1024   32768  100%         1.38         0.794         4.236    3.07x               0.58x  400.71 MB  384.20 MB    1.25 GB
topp_partial     1024   32768   50%        0.704         0.565         4.298    6.11x               0.80x  400.71 MB  384.20 MB    1.25 GB
topk_topp_whole  1024   32768  200%        0.629         1.593          5.04    8.01x               2.53x  400.71 MB  384.20 MB    1.25 GB
mixed_partial    1024   32768  133%        1.023         1.306         5.996    5.86x               1.28x  400.71 MB  384.20 MB    1.25 GB
topk_whole          1  131072  100%        0.133         0.941         0.567    4.26x               7.08x    2.21 MB    1.70 MB    5.24 MB
topk_partial        1  131072    0%        0.044         0.024         0.193    4.39x               0.55x    2.21 MB    1.70 MB    5.24 MB
topp_whole          1  131072  100%        0.895         1.116         0.155    0.17x               1.25x    2.21 MB    1.70 MB    5.24 MB
topp_partial        1  131072    0%        0.043         0.024         0.252    5.86x               0.56x    2.21 MB    1.70 MB    5.24 MB
topk_topp_whole     1  131072  200%        0.155         2.018         0.405    2.61x              13.02x    2.21 MB    1.70 MB    5.24 MB
mixed_partial       1  131072  200%        0.156         2.017         0.332    2.13x              12.93x    2.21 MB    1.70 MB    5.24 MB
topk_whole          4  131072  100%        0.132         0.947         0.153    1.16x               7.17x    8.21 MB    6.20 MB   26.31 MB
topk_partial        4  131072   50%        0.128         0.933         0.154    1.20x               7.29x    8.21 MB    6.20 MB   26.31 MB
topp_whole          4  131072  100%        0.947          1.12          0.35    0.37x               1.18x    8.21 MB    6.20 MB   26.31 MB
topp_partial        4  131072   50%        0.987          1.11          0.35    0.35x               1.12x    8.21 MB    6.20 MB   26.31 MB
topk_topp_whole     4  131072  200%        0.164         2.028         0.392    2.39x              12.37x    8.21 MB    6.20 MB   26.31 MB
mixed_partial       4  131072  150%        0.896         2.004         0.389    0.43x               2.24x    8.21 MB    6.20 MB   26.31 MB
topk_whole         16  131072  100%        0.134         0.952         3.252   24.27x               7.10x   32.21 MB   24.20 MB  104.59 MB
topk_partial       16  131072   50%         0.13         0.946         0.992    7.63x               7.28x   32.21 MB   24.20 MB  104.59 MB
topp_whole         16  131072  100%         0.95         1.126         0.575    0.61x               1.19x   32.21 MB   24.20 MB  104.59 MB
topp_partial       16  131072   50%        1.007         1.121             1    0.99x               1.11x   32.21 MB   24.20 MB  104.59 MB
topk_topp_whole    16  131072  200%        0.173         2.037         0.611    3.53x              11.77x   32.21 MB   24.20 MB  104.59 MB
mixed_partial      16  131072  138%        1.135         2.036         1.098    0.97x               1.79x   32.21 MB   24.20 MB  104.59 MB
topk_whole         24  131072  100%        0.133         0.962         0.503    3.78x               7.23x   48.21 MB   36.20 MB  156.78 MB
topk_partial       24  131072   50%        0.133         0.958         0.803    6.04x               7.20x   48.21 MB   36.20 MB  156.78 MB
topp_whole         24  131072  100%        1.017         1.134         0.729    0.72x               1.12x   48.21 MB   36.20 MB  156.78 MB
topp_partial       24  131072   50%        1.018         1.133         1.146    1.13x               1.11x   48.21 MB   36.20 MB  156.78 MB
topk_topp_whole    24  131072  200%         0.17         2.056         1.917   11.28x              12.09x   48.21 MB   36.20 MB  156.78 MB
mixed_partial      24  131072  133%        1.153         2.054         0.754    0.65x               1.78x   48.21 MB   36.20 MB  156.78 MB
topk_whole         32  131072  100%        0.138         0.974         0.714    5.17x               7.06x   64.21 MB   48.20 MB  208.97 MB
topk_partial       32  131072   50%        0.136         0.968         1.041    7.65x               7.12x   64.21 MB   48.20 MB  208.97 MB
topp_whole         32  131072  100%        1.049         1.145         0.872    0.83x               1.09x   64.21 MB   48.20 MB  208.97 MB
topp_partial       32  131072   50%        1.023         1.142         0.983    0.96x               1.12x   64.21 MB   48.20 MB  208.97 MB
topk_topp_whole    32  131072  200%        0.177         2.077         1.462    8.26x              11.73x   64.21 MB   48.20 MB  208.97 MB
mixed_partial      32  131072  138%        1.235         2.073         0.909    0.74x               1.68x   64.21 MB   48.20 MB  208.97 MB
topk_whole         48  131072  100%         0.14         0.991         4.286   30.61x               7.08x   96.21 MB   72.20 MB  314.20 MB
topk_partial       48  131072   50%        0.138          0.99         1.198    8.68x               7.17x   96.21 MB   72.20 MB  314.20 MB
topp_whole         48  131072  100%         1.09         1.162         1.222    1.12x               1.07x   96.21 MB   72.20 MB  314.20 MB
topp_partial       48  131072   50%        1.041         1.161         1.197    1.15x               1.12x   96.21 MB   72.20 MB  314.20 MB
topk_topp_whole    48  131072  200%        0.179         2.095         1.228    6.86x              11.70x   96.21 MB   72.20 MB  314.20 MB
mixed_partial      48  131072  133%         1.19         2.063         1.178    0.99x               1.73x   96.21 MB   72.20 MB  314.20 MB
topk_whole         56  131072  100%        0.144             1         2.586   17.96x               6.94x  112.21 MB   84.20 MB  366.20 MB
topk_partial       56  131072   50%        0.139         0.997         1.129    8.12x               7.17x  112.21 MB   84.20 MB  366.20 MB
topp_whole         56  131072  100%        1.099         1.169         2.427    2.21x               1.06x  112.21 MB   84.20 MB  366.20 MB
topp_partial       56  131072   50%        1.031         1.168         1.914    1.86x               1.13x  112.21 MB   84.20 MB  366.20 MB
topk_topp_whole    56  131072  200%        0.181         2.101           2.1   11.60x              11.61x  112.21 MB   84.20 MB  366.20 MB
mixed_partial      56  131072  136%        1.293         2.072         1.362    1.05x               1.60x  112.21 MB   84.20 MB  366.20 MB
topk_whole         64  131072  100%        0.148         1.034         1.136    7.68x               6.99x  128.21 MB   96.20 MB  418.20 MB
topk_partial       64  131072   50%        0.141         1.003         1.133    8.04x               7.11x  128.21 MB   96.20 MB  418.20 MB
topp_whole         64  131072  100%        1.115         1.202         1.688    1.51x               1.08x  128.21 MB   96.20 MB  418.20 MB
topp_partial       64  131072   50%        1.047         1.173         1.401    1.34x               1.12x  128.21 MB   96.20 MB  418.20 MB
topk_topp_whole    64  131072  200%        0.187         2.121         1.854    9.91x              11.34x  128.21 MB   96.20 MB  418.20 MB
mixed_partial      64  131072  134%        1.311         2.098         2.007    1.53x               1.60x  128.21 MB   96.20 MB  418.20 MB
topk_whole         96  131072  100%        0.158         1.642         4.722   29.89x              10.39x  192.21 MB  144.20 MB  626.50 MB
topk_partial       96  131072   50%        0.144         1.011         1.895   13.16x               7.02x  192.21 MB  144.20 MB  626.50 MB
topp_whole         96  131072  100%        1.505         1.717         2.048    1.36x               1.14x  192.21 MB  144.20 MB  626.50 MB
topp_partial       96  131072   50%        1.099         1.181         1.916    1.74x               1.07x  192.21 MB  144.20 MB  626.50 MB
topk_topp_whole    96  131072  200%        0.195         3.285         3.479   17.84x              16.85x  192.21 MB  144.20 MB  626.50 MB
mixed_partial      96  131072  133%        1.407         2.754          4.44    3.16x               1.96x  192.21 MB  144.20 MB  626.50 MB
topk_whole        128  131072  100%        0.165         1.661         2.609   15.81x              10.07x  256.21 MB  192.20 MB  640.20 MB
topk_partial      128  131072   50%        0.149         1.153         2.388   16.03x               7.74x  256.21 MB  192.20 MB  640.20 MB
topp_whole        128  131072  100%         1.85         1.732         2.903    1.57x               0.94x  256.21 MB  192.20 MB  640.20 MB
topp_partial      128  131072   50%        1.213          1.32          2.73    2.25x               1.09x  256.21 MB  192.20 MB  640.20 MB
topk_topp_whole   128  131072  200%        0.206         3.315         2.982   14.48x              16.09x  256.21 MB  192.20 MB  640.20 MB
mixed_partial     128  131072  134%        1.429         2.787         2.812    1.97x               1.95x  256.21 MB  192.20 MB  640.20 MB
topk_whole        192  131072  100%        0.298          1.72         3.142   10.54x               5.77x  354.21 MB  288.20 MB  960.20 MB
topk_partial      192  131072   50%        0.159         1.643         5.631   35.42x              10.33x  354.21 MB  288.20 MB  960.20 MB
topp_whole        192  131072  100%        2.946         1.797         3.537    1.20x               0.61x  354.21 MB  288.20 MB  960.20 MB
topp_partial      192  131072   50%         1.58         1.718         3.874    2.45x               1.09x  354.21 MB  288.20 MB  960.20 MB
topk_topp_whole   192  131072  200%        0.375         3.423         3.791   10.11x               9.13x  354.21 MB  288.20 MB  960.20 MB
mixed_partial     192  131072  133%        1.597         2.874         3.695    2.31x               1.80x  354.21 MB  288.20 MB  960.20 MB
topk_whole        256  131072  100%        0.314         1.778         4.156   13.24x               5.66x  450.21 MB  384.20 MB    1.25 GB
topk_partial      256  131072   50%        0.166          1.66         6.882   41.46x              10.00x  450.21 MB  384.20 MB    1.25 GB
topp_whole        256  131072  100%         3.55         1.836         4.341    1.22x               0.52x  450.21 MB  384.20 MB    1.25 GB
topp_partial      256  131072   50%        1.861         1.733         4.244    2.28x               0.93x  450.21 MB  384.20 MB    1.25 GB
topk_topp_whole   256  131072  200%        0.395         3.502         8.431   21.34x               8.87x  450.21 MB  384.20 MB    1.25 GB
mixed_partial     256  131072  134%        2.241         3.399         5.953    2.66x               1.52x  450.21 MB  384.20 MB    1.25 GB
topk_whole        512  131072  100%        0.604         2.177         7.488   12.40x               3.60x  834.21 MB  768.20 MB    2.50 GB
topk_partial      512  131072   50%         0.31         1.777         8.979   28.96x               5.73x  834.21 MB  768.20 MB    2.50 GB
topp_whole        512  131072  100%        6.894         2.155         8.787    1.27x               0.31x  834.21 MB  768.20 MB    2.50 GB
topp_partial      512  131072   50%        3.552         1.864         8.343    2.35x               0.52x  834.21 MB  768.20 MB    2.50 GB
topk_topp_whole   512  131072  200%         0.76         4.142        10.756   14.15x               5.45x  834.21 MB  768.20 MB    2.50 GB
mixed_partial     512  131072  134%        4.129          3.83         9.383    2.27x               0.93x  834.21 MB  768.20 MB    2.50 GB
topk_whole       1024  131072  100%        1.177          3.58        14.857   12.62x               3.04x    1.56 GB    1.50 GB    5.00 GB
topk_partial     1024  131072   50%        0.602         2.202        14.997   24.91x               3.66x    1.56 GB    1.50 GB    5.00 GB
topp_whole       1024  131072  100%       13.691         3.172        16.312    1.19x               0.23x    1.56 GB    1.50 GB    5.00 GB
topp_partial     1024  131072   50%        6.933         2.243        16.313    2.35x               0.32x    1.56 GB    1.50 GB    5.00 GB
topk_topp_whole  1024  131072  200%        1.488         6.381        17.452   11.73x               4.29x    1.56 GB    1.50 GB    5.00 GB
mixed_partial    1024  131072  133%        7.268         5.226        17.183    2.36x               0.72x    1.56 GB    1.50 GB    5.00 GB
```

## Full discussion

See [33538.md](./33538.md) and [33538.json](./33538.json) for the complete locally captured PR discussion. Later comments may correct claims in the description.

# Solution

## Source-backed resolution

The exact PR diff below was recovered from the original GitHub PR. It records the human change; merge status does not establish correctness or a reproducible performance gain.

- Original PR: https://github.com/vllm-project/vllm/pull/33538
- Diff retrieved from: https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/33538.diff
- Retrieved at: 2026-11-03T13:50:46.346007+00:00
- Diff SHA-256: `c84ec01b8ea42d5a0a2655a0dbb2ba297b07840ec640f348b6007dad6643065a`

Existing semantic notes in [33538.sem.json](./33538.sem.json) are derived annotations. They are not a substitute for this patch or the original discussion.

## Ground-truth diff

### Files changed

- `benchmarks/benchmark_topk_topp.py`
- `tests/entrypoints/instrumentator/test_basic.py`
- `tests/v1/sample/test_topk_topp_sampler.py`
- `vllm/utils/math_utils.py`
- `vllm/v1/sample/ops/topk_topp_sampler.py`
- `vllm/v1/sample/ops/topk_topp_triton.py`

### Diff

```diff
diff --git a/benchmarks/benchmark_topk_topp.py b/benchmarks/benchmark_topk_topp.py
new file mode 100644
index 000000000000..cac332a099d8
--- /dev/null
+++ b/benchmarks/benchmark_topk_topp.py
@@ -0,0 +1,471 @@
+#!/usr/bin/env python3
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Benchmark comparing Triton vs PyTorch sort-based top-k/top-p implementations.
+
+Compares:
+- apply_top_k_top_p_triton (Triton binary search)
+- apply_top_k_top_p (PyTorch sort-based)
+
+Scenarios:
+- top_k only (whole batch, partial batch)
+- top_p only (whole batch, partial batch)
+- mix of top_k and top_p
+"""
+
+import argparse
+import gc
+from dataclasses import dataclass
+
+import torch
+
+from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
+from vllm.v1.sample.ops.topk_topp_triton import (
+    apply_top_k_top_p_triton,
+    reset_buffer_cache,
+)
+
+
+@dataclass
+class BenchmarkConfig:
+    """Configuration for a benchmark run."""
+
+    name: str
+    batch_size: int
+    vocab_size: int
+    # k and p can be tensors or None
+    k_values: torch.Tensor | None  # [batch_size] or None
+    p_values: torch.Tensor | None  # [batch_size] or None
+    description: str
+    ops_pct: float = 0.0  # Percentage of ops relative to batch size
+
+
+def calculate_ops_pct(
+    k_values: torch.Tensor | None,
+    p_values: torch.Tensor | None,
+    vocab_size: int,
+    batch_size: int,
+) -> float:
+    """
+    Calculate the percentage of active top-k and top-p operations.
+
+    Returns percentage where 100% = batch_size ops.
+    E.g., if all rows have both top-k and top-p active, returns 200%.
+    """
+    active_ops = 0
+
+    if k_values is not None:
+        # Count rows where k < vocab_size (active top-k filtering)
+        active_ops += (k_values < vocab_size).sum().item()
+
+    if p_values is not None:
+        # Count rows where p < 1.0 (active top-p filtering)
+        active_ops += (p_values < 1.0).sum().item()
+
+    return (active_ops / batch_size) * 100 if batch_size > 0 else 0.0
+
+
+def create_logits(
+    batch_size: int, vocab_size: int, device: str = "cuda"
+) -> torch.Tensor:
+    """Create random logits mimicking a realistic LLM distribution.
+
+    Uses a Zipf-like probability distribution (rank^-1.1) converted to logits
+    via log, then randomly permuted per row. This produces a peaked distribution
+    where a small number of tokens capture most probability mass, similar to
+    real model outputs.
+    """
+    # Create Zipf-like probabilities: p(rank) ~ rank^(-alpha)
+    ranks = torch.arange(1, vocab_size + 1, dtype=torch.float32, device=device)
+    probs = ranks.pow(-1.1)
+    probs = probs / probs.sum()
+
+    # Convert to logits (log-probabilities, unnormalized is fine)
+    base_logits = probs.log()
+
+    # Broadcast to batch and randomly permute each row
+    logits = base_logits.unsqueeze(0).expand(batch_size, -1).clone()
+    for i in range(batch_size):
+        logits[i] = logits[i, torch.randperm(vocab_size, device=device)]
+
+    return logits
+
+
+def measure_memory() -> tuple[int, int]:
+    """Return (allocated, reserved) memory in bytes."""
+    torch.cuda.synchronize()
+    return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
+
+
+def reset_memory_stats():
+    """Reset peak memory statistics."""
+    reset_buffer_cache()
+    torch.cuda.reset_peak_memory_stats()
+    torch.cuda.empty_cache()
+    gc.collect()
+
+
+def benchmark_function(
+    func,
+    logits: torch.Tensor,
+    k: torch.Tensor | None,
+    p: torch.Tensor | None,
+    warmup_iters: int = 5,
+    benchmark_iters: int = 20,
+) -> tuple[float, int]:
+    """
+    Benchmark a function and return (avg_time_ms, peak_memory_bytes).
+
+    Returns average time in milliseconds and peak memory usage.
+    """
+    # Warmup
+    for _ in range(warmup_iters):
+        logits_copy = logits.clone()
+        func(logits_copy, k, p)
+    torch.cuda.synchronize()
+
+    # Reset memory stats before benchmark
+    reset_memory_stats()
+
+    # Benchmark
+    start_events = [
+        torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)
+    ]
+    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
+
+    for i in range(benchmark_iters):
+        logits_copy = logits.clone()
+        start_events[i].record()
+        func(logits_copy, k, p)
+        end_events[i].record()
+
+    torch.cuda.synchronize()
+
+    # Calculate timing
+    times = [
+        start_events[i].elapsed_time(end_events[i]) for i in range(benchmark_iters)
+    ]
+    avg_time = sum(times) / len(times)
+
+    # Get peak memory
+    _, peak_memory = measure_memory()
+
+    return avg_time, peak_memory
+
+
+def create_benchmark_configs(
+    batch_sizes: list[int],
+    vocab_sizes: list[int],
+    device: str = "cuda",
+) -> list[BenchmarkConfig]:
+    """Create all benchmark configurations."""
+    configs = []
+
+    for vocab_size in vocab_sizes:
+        for batch_size in batch_sizes:
+            # 1. Top-k only - whole batch (all rows have k < vocab_size)
+            k_all = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
+            configs.append(
+                BenchmarkConfig(
+                    name=f"topk_whole_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=k_all,
+                    p_values=None,
+                    description=f"Top-k only (whole batch, k=50), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(k_all, None, vocab_size, batch_size),
+                )
+            )
+
+            # 2. Top-k only - partial batch (half have k=50, half have k=vocab_size)
+            k_partial = torch.full((batch_size,), 50, dtype=torch.int32, device=device)
+            k_partial[batch_size // 2 :] = vocab_size  # No filtering for second half
+            configs.append(
+                BenchmarkConfig(
+                    name=f"topk_partial_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=k_partial,
+                    p_values=None,
+                    description=f"Top-k only (partial batch, 50% k=50, 50% k=vocab), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(k_partial, None, vocab_size, batch_size),
+                )
+            )
+
+            # 3. Top-p only - whole batch (all rows have p < 1.0)
+            p_all = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
+            configs.append(
+                BenchmarkConfig(
+                    name=f"topp_whole_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=None,
+                    p_values=p_all,
+                    description=f"Top-p only (whole batch, p=0.9), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(None, p_all, vocab_size, batch_size),
+                )
+            )
+
+            # 4. Top-p only - partial batch (half have p=0.9, half have p=1.0)
+            p_partial = torch.full(
+                (batch_size,), 0.9, dtype=torch.float32, device=device
+            )
+            p_partial[batch_size // 2 :] = 1.0  # No filtering for second half
+            configs.append(
+                BenchmarkConfig(
+                    name=f"topp_partial_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=None,
+                    p_values=p_partial,
+                    description=f"Top-p only (partial batch, 50% p=0.9, 50% p=1.0), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(None, p_partial, vocab_size, batch_size),
+                )
+            )
+
+            # 5. Mix of top-k and top-p (both applied to whole batch)
+            k_mix = torch.full((batch_size,), 100, dtype=torch.int32, device=device)
+            p_mix = torch.full((batch_size,), 0.9, dtype=torch.float32, device=device)
+            configs.append(
+                BenchmarkConfig(
+                    name=f"topk_topp_whole_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=k_mix,
+                    p_values=p_mix,
+                    description=f"Top-k + Top-p (whole batch, k=100, p=0.9), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(k_mix, p_mix, vocab_size, batch_size),
+                )
+            )
+
+            # 6. Mix with partial application (some rows k only, some p only, some both)
+            k_mixed = torch.full(
+                (batch_size,), vocab_size, dtype=torch.int32, device=device
+            )
+            p_mixed = torch.full((batch_size,), 1.0, dtype=torch.float32, device=device)
+            # First third: k only
+            third = batch_size // 3
+            k_mixed[:third] = 50
+            # Second third: p only
+            p_mixed[third : 2 * third] = 0.5
+            # Last third: both k and p
+            k_mixed[2 * third :] = 100
+            p_mixed[2 * third :] = 0.9
+            configs.append(
+                BenchmarkConfig(
+                    name=f"mixed_partial_b{batch_size}_v{vocab_size // 1000}k",
+                    batch_size=batch_size,
+                    vocab_size=vocab_size,
+                    k_values=k_mixed,
+                    p_values=p_mixed,
+                    description=f"Mixed partial (1/3 k=50, 1/3 p=0.9, 1/3 both), "
+                    f"batch={batch_size}, vocab={vocab_size}",
+                    ops_pct=calculate_ops_pct(k_mixed, p_mixed, vocab_size, batch_size),
+                )
+            )
+
+    return configs
+
+
+def format_memory(bytes_val: int) -> str:
+    """Format memory in human-readable form."""
+    if bytes_val >= 1024**3:
+        return f"{bytes_val / (1024**3):.2f} GB"
+    elif bytes_val >= 1024**2:
+        return f"{bytes_val / (1024**2):.2f} MB"
+    elif bytes_val >= 1024:
+        return f"{bytes_val / 1024:.2f} KB"
+    return f"{bytes_val} B"
+
+
+def run_benchmark(
+    configs: list[BenchmarkConfig],
+    warmup_iters: int = 5,
+    benchmark_iters: int = 20,
+    verbose: bool = True,
+):
+    """Run all benchmarks and print results."""
+    results = []
+
+    print("=" * 100)
+    print("Top-k/Top-p Benchmark: Triton vs PyTorch Sort-based")
+    print("=" * 100)
+    print()
+
+    for config in configs:
+        if verbose:
+            print(f"Running: {config.description}")
+
+        # Create fresh logits for this config
+        logits = create_logits(config.batch_size, config.vocab_size)
+
+        # Benchmark Triton
+        reset_memory_stats()
+        triton_time, triton_mem = benchmark_function(
+            apply_top_k_top_p_triton,
+            logits,
+            config.k_values,
+            config.p_values,
+            warmup_iters,
+            benchmark_iters,
+        )
+
+        # Benchmark PyTorch
+        reset_memory_stats()
+        pytorch_time, pytorch_mem = benchmark_function(
+            apply_top_k_top_p_pytorch,
+            logits,
+            config.k_values,
+            config.p_values,
+            warmup_iters,
+            benchmark_iters,
+        )
+
+        speedup = pytorch_time / triton_time if triton_time > 0 else float("inf")
+        mem_ratio = pytorch_mem / triton_mem if triton_mem > 0 else float("inf")
+
+        result = {
+            "config": config,
+            "triton_time_ms": triton_time,
+            "pytorch_time_ms": pytorch_time,
+            "triton_mem": triton_mem,
+            "pytorch_mem": pytorch_mem,
+            "speedup": speedup,
+            "mem_ratio": mem_ratio,
+        }
+        results.append(result)
+
+        if verbose:
+            print(f"  Triton:  {triton_time:.3f} ms, {format_memory(triton_mem)}")
+            print(f"  PyTorch: {pytorch_time:.3f} ms, {format_memory(pytorch_mem)}")
+            print(f"  Speedup: {speedup:.2f}x, Memory ratio: {mem_ratio:.2f}x")
+            print()
+
+        # Clean up
+        del logits
+        reset_memory_stats()
+
+    return results
+
+
+def print_summary_table(results: list[dict]):
+    """Print a summary table of results."""
+    print()
+    print("=" * 130)
+    print("SUMMARY TABLE")
+    print("=" * 130)
+    print()
+
+    # Header
+    header = (
+        f"{'Scenario':<40} {'Batch':>6} {'Vocab':>7} {'Ops%':>6} "
+        f"{'Triton (ms)':>12} {'PyTorch (ms)':>13} {'Speedup':>8} "
+        f"{'Tri Mem':>10} {'Pyt Mem':>10}"
+    )
+    print(header)
+    print("-" * 130)
+
+    # Group by scenario type
+    current_vocab = None
+    for result in results:
+        config = result["config"]
+
+        # Add separator between vocab sizes
+        if current_vocab != config.vocab_size:
+            if current_vocab is not None:
+                print("-" * 130)
+            current_vocab = config.vocab_size
+
+        scenario = config.name.split("_b")[0]  # Extract scenario name
+        print(
+            f"{scenario:<40} {config.batch_size:>6} {config.vocab_size:>7} "
+            f"{config.ops_pct:>5.0f}% "
+            f"{result['triton_time_ms']:>12.3f} {result['pytorch_time_ms']:>13.3f} "
+            f"{result['speedup']:>7.2f}x "
+            f"{format_memory(result['triton_mem']):>10} "
+            f"{format_memory(result['pytorch_mem']):>10}"
+        )
+
+    print("=" * 130)
+
+
+def main():
+    parser = argparse.ArgumentParser(
+        description="Benchmark Triton vs PyTorch sort-based top-k/top-p implementations"
+    )
+    parser.add_argument(
+        "--batch-sizes",
+        type=int,
+        nargs="+",
+        default=[1, 4, 16, 64, 128, 512, 1024, 2048],
+        help="Batch sizes to test (default: 1 4 16 64)",
+    )
+    parser.add_argument(
+        "--vocab-sizes",
+        type=int,
+        nargs="+",
+        default=[32768, 131072],  # 32k, 128k
+        help="Vocabulary sizes to test (default: 32768 131072)",
+    )
+    parser.add_argument(
+        "--warmup-iters",
+        type=int,
+        default=5,
+        help="Number of warmup iterations (default: 5)",
+    )
+    parser.add_argument(
+        "--benchmark-iters",
+        type=int,
+        default=20,
+        help="Number of benchmark iterations (default: 20)",
+    )
+    parser.add_argument(
+        "--quiet",
+        action="store_true",
+        help="Only print summary table",
+    )
+
+    args = parser.parse_args()
+
+    # Print configuration
+    print(f"Batch sizes: {args.batch_sizes}")
+    print(f"Vocab sizes: {args.vocab_sizes}")
+    print(f"Warmup iterations: {args.warmup_iters}")
+    print(f"Benchmark iterations: {args.benchmark_iters}")
+    print()
+
+    # Check CUDA
+    if not torch.cuda.is_available():
+        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
+        return
+
+    device_name = torch.cuda.get_device_name(0)
+    print(f"GPU: {device_name}")
+    print()
+
+    # Create configs
+    configs = create_benchmark_configs(
+        args.batch_sizes,
+        args.vocab_sizes,
+    )
+
+    # Run benchmarks
+    results = run_benchmark(
+        configs,
+        warmup_iters=args.warmup_iters,
+        benchmark_iters=args.benchmark_iters,
+        verbose=not args.quiet,
+    )
+
+    # Print summary
+    print_summary_table(results)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tests/entrypoints/instrumentator/test_basic.py b/tests/entrypoints/instrumentator/test_basic.py
index 1ff30de31bbe..9c2986ebe6c9 100644
--- a/tests/entrypoints/instrumentator/test_basic.py
+++ b/tests/entrypoints/instrumentator/test_basic.py
@@ -145,6 +145,7 @@ async def test_request_cancellation(server: RemoteOpenAIServer):
                 model=MODEL_NAME,
                 max_tokens=10000,
                 extra_body={"min_tokens": 10000},
+                temperature=0.0,
             )
         )
         tasks.append(task)
@@ -163,7 +164,7 @@ async def test_request_cancellation(server: RemoteOpenAIServer):
     # be able to respond to this one within the timeout
     client = server.get_async_client(timeout=5)
     response = await client.chat.completions.create(
-        messages=chat_input, model=MODEL_NAME, max_tokens=10
+        messages=chat_input, model=MODEL_NAME, max_tokens=10, temperature=0.0
     )
 
     assert len(response.choices) == 1
diff --git a/tests/v1/sample/test_topk_topp_sampler.py b/tests/v1/sample/test_topk_topp_sampler.py
index 6a3ec704b157..ce1e288a2418 100644
--- a/tests/v1/sample/test_topk_topp_sampler.py
+++ b/tests/v1/sample/test_topk_topp_sampler.py
@@ -5,8 +5,9 @@
 from torch import Generator
 
 from vllm.platforms import current_platform
-from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
+from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
 
+CUDA_DEVICE = "cuda" if current_platform.is_cuda() else None
 DEVICE = current_platform.device_type
 
 BATCH_SIZE = 1024
@@ -39,11 +40,11 @@ def test_topk_impl_equivalence():
     )
 
     # Top-k only implementation
-    result1 = apply_top_k_top_p(logits=logits.clone(), k=k, p=None)
+    result1 = apply_top_k_top_p_pytorch(logits=logits.clone(), k=k, p=None)
 
     # Top-p + top-k
     no_op_top_p = torch.tensor([1.0])
-    result2 = apply_top_k_top_p(logits=logits.clone(), k=k, p=no_op_top_p)
+    result2 = apply_top_k_top_p_pytorch(logits=logits.clone(), k=k, p=no_op_top_p)
 
     assert torch.allclose(result1, result2)
 
@@ -98,7 +99,7 @@ def test_flashinfer_sampler():
         torch.randint(0, 2, (BATCH_SIZE,), generator=generator, dtype=torch.bool), 1.0
     )
 
-    python_logits = apply_top_k_top_p(
+    python_logits = apply_top_k_top_p_pytorch(
         logits=logits.clone(),
         k=k_values,
         p=p_values,
@@ -120,3 +121,451 @@ def test_flashinfer_sampler():
     assert torch.allclose(python_probs, flashinfer_probs, atol=2e-2), (
         "FlashInfer and Python sampling implementations do not match!"
     )
+
+
+# =============================================================================
+# Triton kernel tests
+# =============================================================================
+
+
+@pytest.mark.skipif(CUDA_DEVICE is None, reason="CUDA not available")
+class TestTritonTopkTopp:
+    """Tests for the Triton top-k/top-p kernel."""
+
+    @pytest.fixture(autouse=True)
+    def setup(self):
+        """Set up test fixtures."""
+        torch.set_default_device(CUDA_DEVICE)
+        self.generator = Generator(device=CUDA_DEVICE).manual_seed(42)
+
+    def _compare_results(
+        self,
+        logits: torch.Tensor,
+        k: torch.Tensor | None,
+        p: torch.Tensor | None,
+    ):
+        """Compare Triton kernel results with PyTorch sorting implementation.
+
+        For top-k only, we expect exact match.
+        For top-p (with or without top-k), we allow small differences due to
+        floating-point precision in probability sum calculations.
+        """
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        # Clone logits for both implementations
+        logits_pytorch = logits.clone()
+        logits_triton = logits.clone().to(torch.float32)
+
+        # Apply PyTorch sorting implementation
+        result_pytorch = apply_top_k_top_p_pytorch(logits_pytorch, k, p)
+
+        # Apply Triton kernel
+        k_i32 = k.to(torch.int32) if k is not None else None
+        p_f32 = p.to(torch.float32) if p is not None else None
+        result_triton = apply_top_k_top_p_triton(logits_triton, k_i32, p_f32)
+
+        # Compare kept counts per row
+        pytorch_kept = (result_pytorch != float("-inf")).sum(dim=-1)
+        triton_kept = (result_triton != float("-inf")).sum(dim=-1)
+
+        if p is None:
+            # Top-k only: expect exact match
+            assert torch.equal(pytorch_kept, triton_kept), (
+                f"Top-k mask mismatch: PyTorch kept {pytorch_kept.tolist()}, "
+                f"Triton kept {triton_kept.tolist()}"
+            )
+        else:
+            # Top-p involved: allow small differences
+            # Either < 1% of kept values OR < 5 values absolute
+            max_diff = (pytorch_kept - triton_kept).abs().max().item()
+            max_kept = pytorch_kept.max().item()
+            if max_kept > 0 and max_diff > 3:
+                diff_pct = max_diff / max_kept * 100
+                assert diff_pct < 0.5, (
+                    f"Top-p mask difference too large: {diff_pct:.2f}% "
+                    f"(max diff {max_diff} values out of {max_kept})"
+                )
+
+    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
+    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
+    def test_topk_only(self, batch_size: int, vocab_size: int):
+        """Test top-k only (p=None)."""
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        k = torch.randint(
+            1, min(100, vocab_size), (batch_size,), generator=self.generator
+        )
+        # Randomly disable top-k for some rows (~25%)
+        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
+        k.masked_fill_(disable_mask, vocab_size)
+
+        self._compare_results(logits, k, p=None)
+
+    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
+    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
+    def test_topp_only(self, batch_size: int, vocab_size: int):
+        """Test top-p only (k=None)."""
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]
+        # Randomly disable top-p for some rows (~25%)
+        disable_mask = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
+        p.masked_fill_(disable_mask, 1.0)
+
+        self._compare_results(logits, k=None, p=p)
+
+    @pytest.mark.parametrize("batch_size", [1, 8, 32, 128, 512, 1024])
+    @pytest.mark.parametrize("vocab_size", [1024, 32000, 128256])
+    def test_topk_and_topp(self, batch_size: int, vocab_size: int):
+        """Test combined top-k and top-p."""
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        k = torch.randint(
+            1, min(100, vocab_size), (batch_size,), generator=self.generator
+        )
+        p = torch.rand(batch_size, generator=self.generator) * 0.9 + 0.1  # [0.1, 1.0]
+
+        # Randomly disable top-k for some rows (~25%)
+        disable_k = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
+        k.masked_fill_(disable_k, vocab_size)
+        # Randomly disable top-p for some rows (~25%)
+        disable_p = torch.randint(0, 4, (batch_size,), generator=self.generator) == 0
+        p.masked_fill_(disable_p, 1.0)
+
+        self._compare_results(logits, k, p)
+
+    def test_both_disabled(self):
+        """Test when both k and p are None (should be no-op)."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        logits = torch.randn(32, 1024, generator=self.generator, dtype=torch.float32)
+        logits_clone = logits.clone()
+
+        result = apply_top_k_top_p_triton(logits_clone, k=None, p=None)
+
+        assert torch.equal(result, logits), "Should be no-op when both k and p are None"
+
+    def test_extreme_k_values(self):
+        """Test edge cases for k values."""
+        batch_size, vocab_size = 16, 1024
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+
+        # k=1 (keep only top 1)
+        k = torch.ones(batch_size, dtype=torch.int32)
+        self._compare_results(logits.clone(), k, p=None)
+
+        # k=vocab_size (keep all)
+        k = torch.full((batch_size,), vocab_size, dtype=torch.int32)
+        self._compare_results(logits.clone(), k, p=None)
+
+        # Mixed extreme values
+        k = torch.tensor([1, vocab_size, 2, vocab_size - 1] * 4, dtype=torch.int32)
+        self._compare_results(logits.clone(), k, p=None)
+
+    def test_extreme_p_values(self):
+        """Test edge cases for p values."""
+        batch_size, vocab_size = 16, 1024
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+
+        # p close to 0 (very restrictive)
+        p = torch.full((batch_size,), 0.01, dtype=torch.float32)
+        self._compare_results(logits.clone(), k=None, p=p)
+
+        # p=1.0 (keep all)
+        p = torch.ones(batch_size, dtype=torch.float32)
+        self._compare_results(logits.clone(), k=None, p=p)
+
+        # Mixed values
+        p = torch.tensor([0.1, 0.5, 0.9, 1.0] * 4, dtype=torch.float32)
+        self._compare_results(logits.clone(), k=None, p=p)
+
+    def test_large_batch(self):
+        """Test with a large batch size."""
+        batch_size, vocab_size = 512, 32000
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        k = torch.randint(1, 50, (batch_size,), generator=self.generator)
+        p = torch.rand(batch_size, generator=self.generator) * 0.5 + 0.5
+
+        self._compare_results(logits, k, p)
+
+    # -----------------------------------------------------------------
+    # Tests for -inf logits (e.g. from grammar / structured output masks)
+    # -----------------------------------------------------------------
+
+    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
+    def test_topk_with_neginf_logits(self, inf_fraction: float):
+        """Top-k with many -inf logits (simulating grammar bitmask).
+
+        The kernel must not produce NaN when most logits are -inf, which
+        can happen when structured-output grammar masks are applied before
+        sampling.
+        """
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        # Mask a fraction of logits to -inf.
+        mask = (
+            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
+        )
+        logits[mask] = float("-inf")
+
+        k = torch.randint(
+            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
+        )
+        result = apply_top_k_top_p_triton(logits.clone(), k, None)
+
+        assert not result.isnan().any(), "NaN found in top-k result with -inf logits"
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"
+            # At least one value should survive unless the row was all -inf.
+            finite_in = (logits[i] > float("-inf")).sum().item()
+            if finite_in > 0:
+                assert kept > 0, f"Row {i}: no tokens kept despite finite input"
+
+    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
+    def test_topp_with_neginf_logits(self, inf_fraction: float):
+        """Top-p with many -inf logits."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        mask = (
+            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
+        )
+        logits[mask] = float("-inf")
+
+        p = (
+            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
+            + 0.1
+        )
+        result = apply_top_k_top_p_triton(logits.clone(), None, p)
+
+        assert not result.isnan().any(), "NaN found in top-p result with -inf logits"
+        for i in range(batch_size):
+            finite_in = (logits[i] > float("-inf")).sum().item()
+            kept = (result[i] > float("-inf")).sum().item()
+            if finite_in > 0:
+                assert kept > 0, f"Row {i}: no tokens kept despite finite input"
+
+    @pytest.mark.parametrize("inf_fraction", [0.5, 0.9, 0.99])
+    def test_topk_topp_with_neginf_logits(self, inf_fraction: float):
+        """Combined top-k + top-p with many -inf logits."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        mask = (
+            torch.rand(batch_size, vocab_size, generator=self.generator) < inf_fraction
+        )
+        logits[mask] = float("-inf")
+
+        k = torch.randint(
+            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
+        )
+        p = (
+            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
+            + 0.1
+        )
+        result = apply_top_k_top_p_triton(logits.clone(), k, p)
+
+        assert not result.isnan().any(), (
+            "NaN found in top-k+top-p result with -inf logits"
+        )
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept <= k[i].item(), f"Row {i}: kept {kept} > k={k[i].item()}"
+
+    def test_all_neginf_logits(self):
+        """All logits are -inf (fully masked). Kernel should be a no-op."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 16, 128256
+        logits = torch.full(
+            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
+        )
+
+        k = torch.randint(
+            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
+        )
+        p = torch.full((batch_size,), 0.9, dtype=torch.float32)
+
+        # top-k only
+        result = apply_top_k_top_p_triton(logits.clone(), k, None)
+        assert not result.isnan().any(), "NaN from all-inf top-k"
+        assert (result == float("-inf")).all(), "Expected all -inf unchanged"
+
+        # top-p only
+        result = apply_top_k_top_p_triton(logits.clone(), None, p)
+        assert not result.isnan().any(), "NaN from all-inf top-p"
+        assert (result == float("-inf")).all(), "Expected all -inf unchanged"
+
+        # top-k + top-p
+        result = apply_top_k_top_p_triton(logits.clone(), k, p)
+        assert not result.isnan().any(), "NaN from all-inf top-k+top-p"
+        assert (result == float("-inf")).all(), "Expected all -inf unchanged"
+
+    def test_few_valid_tokens_with_neginf(self):
+        """Only a handful of tokens are finite per row (strict grammar)."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.full(
+            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
+        )
+        # Allow only 5 random tokens per row to be finite.
+        for i in range(batch_size):
+            indices = torch.randperm(vocab_size, generator=self.generator)[:5]
+            logits[i, indices] = torch.randn(
+                5, generator=self.generator, dtype=torch.float32
+            )
+
+        k = torch.full((batch_size,), 50, dtype=torch.int32)
+        p = torch.full((batch_size,), 0.9, dtype=torch.float32)
+
+        # top-k only (k=50 but only 5 finite → keep all 5)
+        result = apply_top_k_top_p_triton(logits.clone(), k, None)
+        assert not result.isnan().any()
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept == 5, f"Row {i}: expected 5 kept, got {kept}"
+
+        # top-k with k < num_finite
+        k_small = torch.full((batch_size,), 3, dtype=torch.int32)
+        result = apply_top_k_top_p_triton(logits.clone(), k_small, None)
+        assert not result.isnan().any()
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept <= 3, f"Row {i}: expected <=3 kept, got {kept}"
+
+        # top-p only
+        result = apply_top_k_top_p_triton(logits.clone(), None, p)
+        assert not result.isnan().any()
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept > 0, f"Row {i}: no tokens kept"
+
+    @pytest.mark.parametrize("num_valid", [1, 2, 5, 10, 50])
+    @pytest.mark.parametrize(
+        "mode",
+        ["topk_only", "topp_only", "topk_and_topp"],
+    )
+    def test_equal_logits_few_valid(self, num_valid: int, mode: str):
+        """Few valid tokens all sharing the same logit value.
+
+        This is the pattern produced by grammar bitmask filtering when
+        the model assigns similar scores to the few allowed tokens.
+        The ternary search can converge to a pivot equal to max_logit,
+        causing the strict `>` keep_mask to exclude everything.
+        Regression test for the `final_pivot >= max_logit` guard.
+        """
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.full(
+            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
+        )
+        # Set exactly `num_valid` tokens per row to the SAME finite value.
+        for i in range(batch_size):
+            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
+            logits[i, indices] = 1.0  # all equal
+
+        k: torch.Tensor | None = None
+        p: torch.Tensor | None = None
+        if mode in ("topk_only", "topk_and_topp"):
+            k = torch.full((batch_size,), max(1, num_valid - 1), dtype=torch.int32)
+        if mode in ("topp_only", "topk_and_topp"):
+            p = torch.full((batch_size,), 0.95, dtype=torch.float32)
+
+        result = apply_top_k_top_p_triton(logits.clone(), k, p)
+
+        assert not result.isnan().any(), "NaN in equal-logit result"
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            # The key invariant: at least one token must survive.
+            # With all-equal logits the pivot search can't differentiate
+            # tokens, so the guard may keep more than k — that is the
+            # intended safe fallback.
+            assert kept > 0, (
+                f"Row {i}: all tokens masked with {num_valid} equal-valued "
+                f"finite logits ({mode})"
+            )
+
+    @pytest.mark.parametrize("num_valid", [2, 5, 10])
+    def test_nearly_equal_logits_topp(self, num_valid: int):
+        """Few valid tokens with very similar (but not identical) logits.
+
+        Ensures the kernel handles near-degenerate probability
+        distributions where the ternary search range collapses.
+        """
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 128256
+        logits = torch.full(
+            (batch_size, vocab_size), float("-inf"), dtype=torch.float32
+        )
+        for i in range(batch_size):
+            indices = torch.randperm(vocab_size, generator=self.generator)[:num_valid]
+            # Tiny spread: values in [1.0, 1.0 + 1e-6]
+            logits[i, indices] = (
+                1.0
+                + torch.rand(num_valid, generator=self.generator, dtype=torch.float32)
+                * 1e-6
+            )
+
+        p = torch.full((batch_size,), 0.95, dtype=torch.float32)
+        result = apply_top_k_top_p_triton(logits.clone(), None, p)
+
+        assert not result.isnan().any(), "NaN in nearly-equal-logit result"
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept > 0, (
+                f"Row {i}: all tokens masked with {num_valid} "
+                f"nearly-equal finite logits"
+            )
+
+    def test_mixed_neginf_and_normal_rows(self):
+        """Batch with a mix of normal rows and heavily-masked rows."""
+        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
+
+        batch_size, vocab_size = 32, 32000
+        logits = torch.randn(
+            batch_size, vocab_size, generator=self.generator, dtype=torch.float32
+        )
+        # Mask even rows heavily (99% -inf), leave odd rows normal.
+        for i in range(0, batch_size, 2):
+            mask = torch.rand(vocab_size, generator=self.generator) < 0.99
+            logits[i][mask] = float("-inf")
+
+        k = torch.randint(
+            1, 50, (batch_size,), generator=self.generator, dtype=torch.int32
+        )
+        p = (
+            torch.rand(batch_size, generator=self.generator, dtype=torch.float32) * 0.9
+            + 0.1
+        )
+
+        result = apply_top_k_top_p_triton(logits.clone(), k, p)
+        assert not result.isnan().any(), "NaN in mixed normal/-inf batch"
+        for i in range(batch_size):
+            kept = (result[i] > float("-inf")).sum().item()
+            assert kept <= k[i].item()
+            finite_in = (logits[i] > float("-inf")).sum().item()
+            if finite_in > 0:
+                assert kept > 0, f"Row {i}: no tokens kept"
diff --git a/vllm/utils/math_utils.py b/vllm/utils/math_utils.py
index 5fc6c3d664f0..a0e301af471f 100644
--- a/vllm/utils/math_utils.py
+++ b/vllm/utils/math_utils.py
@@ -14,16 +14,12 @@ def cdiv(a: int, b: int) -> int:
 
 def next_power_of_2(n: int) -> int:
     """The next power of 2 (inclusive)"""
-    if n < 1:
-        return 1
-    return 1 << (n - 1).bit_length()
+    return 1 if n < 1 else 1 << (n - 1).bit_length()
 
 
 def prev_power_of_2(n: int) -> int:
     """The previous power of 2 (inclusive)"""
-    if n <= 0:
-        return 0
-    return 1 << (n.bit_length() - 1)
+    return 0 if n <= 0 else 1 << (n.bit_length() - 1)
 
 
 def round_up(x: int, y: int) -> int:
diff --git a/vllm/v1/sample/ops/topk_topp_sampler.py b/vllm/v1/sample/ops/topk_topp_sampler.py
index 03da3e565e49..33f7090e4e3d 100644
--- a/vllm/v1/sample/ops/topk_topp_sampler.py
+++ b/vllm/v1/sample/ops/topk_topp_sampler.py
@@ -11,6 +11,10 @@
 from vllm.config.model import LogprobsMode
 from vllm.logger import init_logger
 from vllm.platforms import CpuArchEnum, current_platform
+from vllm.triton_utils import HAS_TRITON
+
+if HAS_TRITON:
+    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
 
 logger = init_logger(__name__)
 
@@ -87,8 +91,6 @@ def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None:
         else:
             self.forward = self.forward_native
 
-        self.apply_top_k_top_p = apply_top_k_top_p
-
     def forward_native(
         self,
         logits: torch.Tensor,
@@ -101,7 +103,7 @@ def forward_native(
 
         The logits tensor may be updated in-place.
         """
-        logits = self.apply_top_k_top_p(logits, k, p)
+        logits = apply_top_k_top_p(logits, k, p)
         logits_to_return = None
         if self.logprobs_mode == "processed_logits":
             logits_to_return = logits
@@ -149,7 +151,7 @@ def forward_cpu(
 
         The logits tensor may be updated in-place.
         """
-        logits = self.apply_top_k_top_p(logits, k, p)
+        logits = apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
         logits_to_return = None
         if self.logprobs_mode == "processed_logits":
             logits_to_return = logits
@@ -158,14 +160,14 @@ def forward_cpu(
 
         if len(generators) != logits.shape[0]:
             return compiled_random_sample(logits), logits_to_return
-        else:
-            probs = logits.softmax(dim=-1, dtype=torch.float32)
-            q = torch.empty_like(probs)
-            q.exponential_()
-            for i, generator in generators.items():
-                q[i].exponential_(generator=generator)
 
-            return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return
+        probs = logits.softmax(dim=-1, dtype=torch.float32)
+        q = torch.empty_like(probs)
+        q.exponential_()
+        for i, generator in generators.items():
+            q[i].exponential_(generator=generator)
+
+        return probs.div_(q).argmax(dim=-1).view(-1), logits_to_return
 
     def forward_hip(
         self,
@@ -241,9 +243,23 @@ def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
 
 
 def apply_top_k_top_p(
+    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
+) -> torch.Tensor:
+    if p is None and k is None:
+        return logits
+
+    if HAS_TRITON and logits.shape[0] >= 8:
+        return apply_top_k_top_p_triton(logits, k, p)
+
+    # Use pytorch sort implementation for small batch sizes.
+    return apply_top_k_top_p_pytorch(logits, k, p)
+
+
+def apply_top_k_top_p_pytorch(
     logits: torch.Tensor,
     k: torch.Tensor | None,
     p: torch.Tensor | None,
+    allow_cpu_sync: bool = False,
 ) -> torch.Tensor:
     """Apply top-k and top-p masks to the logits.
 
@@ -256,8 +272,9 @@ def apply_top_k_top_p(
         if k is None:
             return logits
 
-        # Avoid sorting vocab for top-k only case.
-        return apply_top_k_only(logits, k)
+        if allow_cpu_sync:
+            # Avoid sorting vocab for top-k only case.
+            return apply_top_k_only(logits, k)
 
     logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
 
@@ -279,18 +296,16 @@ def apply_top_k_top_p(
         logits_sort.masked_fill_(top_p_mask, -float("inf"))
 
     # Re-sort the probabilities.
-    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
-    return logits
+    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
 
 
-def apply_top_k_only(
-    logits: torch.Tensor,
-    k: torch.Tensor,
-) -> torch.Tensor:
+def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
     """
     Apply top-k mask to the logits.
 
     This implementation doesn't involve sorting the entire vocab.
+    Note however that it involves a GPU->CPU sync which can be detrimental for
+    async scheduling performance.
 
     The logits tensor may be updated in-place.
     """
@@ -304,8 +319,7 @@ def apply_top_k_only(
     top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
     # Handle non-topk rows.
     top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
-    logits.masked_fill_(logits < top_k_mask, -float("inf"))
-    return logits
+    return logits.masked_fill_(logits < top_k_mask, -float("inf"))
 
 
 def random_sample(
diff --git a/vllm/v1/sample/ops/topk_topp_triton.py b/vllm/v1/sample/ops/topk_topp_triton.py
new file mode 100644
index 000000000000..f776e94d6680
--- /dev/null
+++ b/vllm/v1/sample/ops/topk_topp_triton.py
@@ -0,0 +1,1039 @@
+# SPDX-License-Identifier: Apache-2.0
+# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+"""
+Combined Top-K and Top-P Triton kernels.
+
+Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
+using Pivot-based Truncation and Selection" By Park et al.
+(https://arxiv.org/abs/2602.01518)
+
+"""
+
+import torch
+
+from vllm.triton_utils import tl, triton
+from vllm.utils.math_utils import next_power_of_2
+
+_TRITON_TABLE_CACHE: dict[tuple[torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
+_TRITON_BUFFER_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
+
+# fmt: off
+_NORMAL_CDF_TO_SIGMA_TABLE = [
+  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503, 
+  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373, 
+  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198, 
+  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113, 
+  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970, 
+  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844, 
+  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733, 
+  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595, 
+  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485, 
+  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364, 
+  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250, 
+  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147, 
+  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016, 
+  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897, 
+  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751, 
+  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603, 
+  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422, 
+  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163, 
+  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806, 
+  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813 
+]
+
+_PERCENTILE_TO_STD_TABLE = [
+  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659, 
+  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280, 
+  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030, 
+  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837, 
+  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675, 
+  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521, 
+  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383, 
+  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247, 
+  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124, 
+  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003, 
+ -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122, 
+ -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248, 
+ -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382, 
+ -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521, 
+ -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673, 
+ -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, 
+ -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, 
+ -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, 
+ -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658, 
+ -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813 
+]
+# fmt: on
+
+
+@triton.jit
+def _topk_topp_kernel(
+    LOGITS,
+    BUFFER,
+    PERCENTILE_TO_STD_TABLE,
+    NORMAL_CDF_TO_SIGMA_TABLE,
+    K,
+    P,
+    BATCH_SIZE,
+    VOCAB_SIZE: tl.constexpr,
+    MASK_VALUE: tl.constexpr,
+    BLOCK_SIZE: tl.constexpr,
+    BLOCK_SIZE_TRUNC: tl.constexpr,
+    TOPK_ENABLED: tl.constexpr,
+    TOPP_ENABLED: tl.constexpr,
+):
+    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
+    pid = tl.program_id(0)
+    num_programs = tl.num_programs(0)
+    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
+        LOGITS_ROW = LOGITS + row_id * VOCAB_SIZE
+        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE
+
+        final_pivot = -float("inf")
+        duplicate_logit = float("inf")
+        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
+        num_keep = tl.zeros((), dtype=tl.uint32)
+        num_kept = tl.zeros((), dtype=tl.uint32)
+
+        max_logit = -float("inf")
+        min_logit = float("inf")
+
+        if TOPK_ENABLED:
+            k = tl.load(K + row_id)
+            if k < VOCAB_SIZE:
+                # Zeroth pass: Compute avg and std from a sample block
+                offs = tl.arange(0, BLOCK_SIZE)
+                mask_n = offs < VOCAB_SIZE
+                logits_blk0 = tl.load(
+                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
+                )
+                # Exclude -inf values (e.g. from grammar bitmasks) from
+                # statistics to avoid NaN in pivot computation.
+                finite_mask = (logits_blk0 > -float("inf")) & mask_n
+                num_finite = tl.sum(finite_mask)
+                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
+                avg_logit = tl.where(
+                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
+                )
+                sq_avg_logit = tl.where(
+                    num_finite > 0,
+                    tl.sum(finite_logits * finite_logits) / num_finite,
+                    0.0,
+                )
+                std_logit = tl.sqrt(
+                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
+                )
+
+                # Calculate outlier pivot t for Gaussian sigma-truncation
+                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
+                percentile = tl.minimum(percentile, 199)
+                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
+                sigma = sigma + tl.abs(sigma) * -0.15
+                outlier_pivot = avg_logit + std_logit * sigma
+                num_outliers = tl.zeros((), dtype=tl.uint32)
+
+                # First pass: compute max and min logits and gather outliers
+                num_finite_total = tl.zeros((), dtype=tl.uint32)
+                for i in range(0, NUM_TILES):
+                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                    mask_n = offs_n < VOCAB_SIZE
+                    logits_blk = tl.load(
+                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                    )
+
+                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
+                    # Exclude -inf from min to keep binary search bounds
+                    # finite (avoids NaN pivots).
+                    finite_blk_mask = logits_blk > -float("inf")
+                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
+                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
+                    num_finite_total += tl.sum(finite_blk_mask & mask_n)
+
+                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
+                    cumulative_pos = tl.cast(
+                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
+                    )
+                    num_outliers += tl.sum(outlier_mask)
+                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
+                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)
+
+                # If no finite logits exist (all -inf), clamp min to
+                # max so the search converges to -inf (no masking).
+                min_logit = tl.minimum(min_logit, max_logit)
+
+                # Second passes: Ternary search for pivots
+                num_iters = 0
+                k_pivot = float("inf")
+                k_pivots_num = tl.zeros((), dtype=tl.uint32)
+                min_larger = float("inf")
+                num_min_larger = tl.zeros((), dtype=tl.uint32)
+                if num_outliers > k:
+                    max_range = max_logit
+                    min_range = outlier_pivot
+                    search_range = tl.cast(num_outliers, tl.int32)
+                    search_iters = tl.cast(
+                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
+                        tl.int32,
+                    )
+                    found_pivot = 0
+                    while found_pivot == 0:
+                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
+                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
+                        min_larger_0 = float("inf")
+                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
+
+                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
+                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
+                        min_larger_1 = float("inf")
+                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
+
+                        # First pass: Calculate k_pivots_num and min_larger
+                        for i in range(0, search_iters):
+                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                0, BLOCK_SIZE_TRUNC
+                            )
+                            mask_n_2 = offs_n < search_range
+                            logits_blk2 = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
+                            )
+
+                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
+                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
+
+                            min_larger_0 = tl.minimum(min_larger_0, tl.min(logits_blk2))
+                            min_larger_1 = tl.minimum(min_larger_1, tl.min(logits_blk2))
+
+                        # Second pass: Calculate num_min_larger
+                        for i in range(0, search_iters):
+                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                0, BLOCK_SIZE_TRUNC
+                            )
+                            mask_n_2 = offs_n < search_range
+                            logits_blk2 = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
+                            )
+
+                            num_min_larger_0 += tl.sum(
+                                tl.abs(logits_blk2 - min_larger_0) < 1e-9
+                            )
+                            num_min_larger_1 += tl.sum(
+                                tl.abs(logits_blk2 - min_larger_1) < 1e-9
+                            )
+
+                        # Check if any of the pivots satisfy termination condition
+                        if (
+                            k_pivots_num_0 >= k
+                            and k_pivots_num_0 - num_min_larger_0 < k
+                        ):
+                            k_pivot = k_pivot_0
+                            k_pivots_num = k_pivots_num_0
+                            min_larger = min_larger_0
+                            num_min_larger = num_min_larger_0
+                            found_pivot = 1
+                        if (
+                            k_pivots_num_1 >= k
+                            and k_pivots_num_1 - num_min_larger_1 < k
+                        ):
+                            k_pivot = k_pivot_1
+                            k_pivots_num = k_pivots_num_1
+                            min_larger = min_larger_1
+                            num_min_larger = num_min_larger_1
+                            found_pivot = 1
+
+                        # Update range
+                        if k_pivots_num_1 > k:
+                            min_range = k_pivot_1
+                        elif k_pivots_num_0 > k:
+                            min_range = k_pivot_0
+
+                        if k_pivots_num_0 < k:
+                            max_range = k_pivot_0
+                        elif k_pivots_num_1 < k:
+                            max_range = k_pivot_1
+
+                        num_iters += 1
+                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
+                            k_pivot = (max_range + min_range) / 2.0
+                            found_pivot = 1
+                else:
+                    # If top-k outlier gathering failed, search whole logit space
+                    max_range = max_logit
+                    min_range = min_logit
+                    found_pivot = 0
+                    while found_pivot == 0:
+                        k_pivot_0 = (max_range - min_range) * 1.0 / 4.0 + min_range
+                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
+                        min_larger_0 = float("inf")
+                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
+
+                        k_pivot_1 = (max_range - min_range) * 2.0 / 4.0 + min_range
+                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
+                        min_larger_1 = float("inf")
+                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
+
+                        # First pass: Calculate k_pivots_num and min_larger
+                        for i in range(0, NUM_TILES):
+                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                            mask_n = offs_n < VOCAB_SIZE
+                            logits_blk2 = tl.load(
+                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                            )
+
+                            k_pivots_num_0 += tl.sum(logits_blk2 > k_pivot_0)
+                            k_pivots_num_1 += tl.sum(logits_blk2 > k_pivot_1)
+
+                            # Exclude -inf from min_larger to avoid
+                            # poisoning the convergence check.
+                            finite_blk2 = tl.where(
+                                logits_blk2 > -float("inf"), logits_blk2, float("inf")
+                            )
+                            min_larger_0 = tl.minimum(min_larger_0, tl.min(finite_blk2))
+                            min_larger_1 = tl.minimum(min_larger_1, tl.min(finite_blk2))
+
+                        # Second pass: Calculate num_min_larger
+                        for i in range(0, NUM_TILES):
+                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                            mask_n = offs_n < VOCAB_SIZE
+                            logits_blk2 = tl.load(
+                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                            )
+
+                            num_min_larger_0 += tl.sum(
+                                tl.abs(logits_blk2 - min_larger_0) < 1e-9
+                            )
+                            num_min_larger_1 += tl.sum(
+                                tl.abs(logits_blk2 - min_larger_1) < 1e-9
+                            )
+
+                        # Check if any of the pivots satisfy termination condition
+                        if (
+                            k_pivots_num_0 >= k
+                            and k_pivots_num_0 - num_min_larger_0 < k
+                        ):
+                            k_pivot = k_pivot_0
+                            k_pivots_num = k_pivots_num_0
+                            min_larger = min_larger_0
+                            num_min_larger = num_min_larger_0
+                            found_pivot = 1
+                        if (
+                            k_pivots_num_1 >= k
+                            and k_pivots_num_1 - num_min_larger_1 < k
+                        ):
+                            k_pivot = k_pivot_1
+                            k_pivots_num = k_pivots_num_1
+                            min_larger = min_larger_1
+                            num_min_larger = num_min_larger_1
+                            found_pivot = 1
+
+                        # Update range
+                        if k_pivots_num_1 > k:
+                            min_range = k_pivot_1
+                        elif k_pivots_num_0 > k:
+                            min_range = k_pivot_0
+
+                        if k_pivots_num_0 < k:
+                            max_range = k_pivot_0
+                        elif k_pivots_num_1 < k:
+                            max_range = k_pivot_1
+
+                        num_iters += 1
+                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
+                            k_pivot = (max_range + min_range) / 2.0
+                            found_pivot = 1
+
+                duplicate_logit = min_larger
+                num_duplicate_logit = num_min_larger
+                num_keep = num_duplicate_logit - (k_pivots_num - k)
+                num_kept = tl.zeros((), dtype=tl.uint32)
+
+                # Top-k only path.  If there are fewer finite values
+                # than k (e.g. grammar mask), keep everything.
+                final_pivot = k_pivot if num_finite_total > k else -float("inf")
+
+                if TOPP_ENABLED and num_finite_total > k:
+                    #### TOP-P SAMPLING AFTER TOP-K ####
+                    p = tl.load(P + row_id)
+                    if p < 1.0:
+                        min_logit = k_pivot
+                        sum_exp_logits = 0.0
+                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
+                        search_range = tl.cast(num_outliers, tl.int32)
+                        search_iters = tl.cast(
+                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
+                            tl.int32,
+                        )
+
+                        # Third pass: Calculate exp logits and sum, gather outliers
+                        if num_outliers > k:
+                            for i in range(0, search_iters):
+                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                    0, BLOCK_SIZE_TRUNC
+                                )
+                                mask_n_2 = offs_n < search_range
+
+                                probs_blk = tl.load(
+                                    BUFFER_ROW + offs_n,
+                                    mask=mask_n_2,
+                                    other=-float("inf"),
+                                )
+
+                                outlier_mask = (probs_blk > min_logit) & mask_n_2
+
+                                # Duplicate logit handling for Top-k
+                                if num_keep < num_duplicate_logit:
+                                    duplicate_mask = (
+                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
+                                    )
+                                    duplicate_count = (
+                                        tl.cumsum(duplicate_mask) + num_kept
+                                    )
+                                    duplicate_keep_mask = (
+                                        duplicate_count <= num_keep
+                                    ) & duplicate_mask
+                                    duplicate_remove_mask = (
+                                        duplicate_mask & ~duplicate_keep_mask
+                                    )
+                                    outlier_mask = outlier_mask & (
+                                        ~duplicate_remove_mask
+                                    )
+                                    num_kept += tl.sum(duplicate_keep_mask)
+
+                                probs_blk = tl.where(
+                                    outlier_mask, probs_blk, -float("inf")
+                                )
+                                probs_blk = probs_blk - max_logit
+                                probs_blk = tl.exp(probs_blk)
+                                sum_exp_logits += tl.sum(probs_blk)
+
+                            # Fourth pass: Calculate BUFFER and get outliers
+                            for i in range(0, search_iters):
+                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                    0, BLOCK_SIZE_TRUNC
+                                )
+                                mask_n_2 = offs_n < search_range
+
+                                probs_blk = tl.load(
+                                    BUFFER_ROW + offs_n,
+                                    mask=mask_n_2,
+                                    other=-float("inf"),
+                                )
+
+                                probs_blk = probs_blk - max_logit
+                                probs_blk = tl.exp(probs_blk)
+                                probs_blk = probs_blk / sum_exp_logits
+                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
+                        else:
+                            # If top-k outlier gathering failed,
+                            # retry gathering using top-k pivot
+                            for i in range(0, NUM_TILES):
+                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                                mask_n = offs_n < VOCAB_SIZE
+
+                                probs_blk = tl.load(
+                                    LOGITS_ROW + offs_n,
+                                    mask=mask_n,
+                                    other=-float("inf"),
+                                )
+
+                                outlier_mask = (probs_blk > min_logit) & mask_n
+
+                                # Duplicate logit handling for Top-k
+                                duplicate_mask = (
+                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
+                                )
+                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
+                                duplicate_keep_mask = (
+                                    duplicate_count <= num_keep
+                                ) & duplicate_mask
+                                duplicate_remove_mask = (
+                                    duplicate_mask & ~duplicate_keep_mask
+                                )
+                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
+                                num_kept += tl.sum(duplicate_keep_mask)
+
+                                probs_blk = tl.where(
+                                    outlier_mask, probs_blk, -float("inf")
+                                )
+                                probs_blk = probs_blk - max_logit
+                                probs_blk = tl.exp(probs_blk)
+                                sum_exp_logits += tl.sum(probs_blk)
+
+                                cumulative_pos = tl.cast(
+                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
+                                    tl.int32,
+                                )
+                                num_outliers_2 += tl.sum(outlier_mask)
+                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
+                                tl.store(
+                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
+                                )
+
+                            search_range = tl.cast(num_outliers_2, tl.int32)
+                            search_iters = tl.cast(
+                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
+                                // BLOCK_SIZE_TRUNC,
+                                tl.int32,
+                            )
+
+                            # Fourth pass: Calculate BUFFER and get outliers
+                            for i in range(0, search_iters):
+                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                    0, BLOCK_SIZE_TRUNC
+                                )
+                                mask_n_2 = offs_n < search_range
+
+                                probs_blk = tl.load(
+                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
+                                )
+                                probs_blk = probs_blk / sum_exp_logits
+                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
+
+                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
+                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits
+
+                        p_pivot = 1.0
+                        num_iters = 0
+                        min_larger_prob = 1.0
+                        num_min_larger = tl.zeros((), dtype=tl.uint32)
+                        p_pivots_sum = 0.0
+
+                        # Fifth passes: Search for p_pivot
+                        found_pivot = 0
+                        while found_pivot == 0:
+                            p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
+                            p_pivots_sum_0 = 0.0
+                            min_larger_0 = 1.0
+                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
+
+                            p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
+                            p_pivots_sum_1 = 0.0
+                            min_larger_1 = 1.0
+                            num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
+
+                            # First pass: Calculate p_pivots_sum and min_larger
+                            for i in range(0, search_iters):
+                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                    0, BLOCK_SIZE_TRUNC
+                                )
+                                mask_n_2 = offs_n < search_range
+                                probs_blk = tl.load(
+                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
+                                )
+
+                                p_pivots_sum_0 += tl.sum(
+                                    probs_blk * (probs_blk > p_pivot_0)
+                                )
+                                masked_larger_0 = tl.where(
+                                    probs_blk > p_pivot_0, probs_blk, 1.0
+                                )
+                                min_larger_0 = tl.minimum(
+                                    min_larger_0, tl.min(masked_larger_0)
+                                )
+
+                                p_pivots_sum_1 += tl.sum(
+                                    probs_blk * (probs_blk > p_pivot_1)
+                                )
+                                masked_larger_1 = tl.where(
+                                    probs_blk > p_pivot_1, probs_blk, 1.0
+                                )
+                                min_larger_1 = tl.minimum(
+                                    min_larger_1, tl.min(masked_larger_1)
+                                )
+
+                            # Second pass: Calculate num_min_larger
+                            for i in range(0, search_iters):
+                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                    0, BLOCK_SIZE_TRUNC
+                                )
+                                mask_n_2 = offs_n < search_range
+                                probs_blk = tl.load(
+                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
+                                )
+
+                                num_min_larger_0 += tl.sum(
+                                    tl.abs(probs_blk - min_larger_0) < 1e-9
+                                )
+                                num_min_larger_1 += tl.sum(
+                                    tl.abs(probs_blk - min_larger_1) < 1e-9
+                                )
+
+                            # Check if any of the pivots satisfy termination condition
+                            if p_pivots_sum_1 >= p and (
+                                p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
+                            ):
+                                p_pivot = p_pivot_1
+                                min_larger_prob = min_larger_1
+                                num_min_larger = num_min_larger_1
+                                p_pivots_sum = p_pivots_sum_1
+                                found_pivot = 1
+                            if p_pivots_sum_0 >= p and (
+                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
+                            ):
+                                p_pivot = p_pivot_0
+                                min_larger_prob = min_larger_0
+                                num_min_larger = num_min_larger_0
+                                p_pivots_sum = p_pivots_sum_0
+                                found_pivot = 1
+
+                            # Update range
+                            if p_pivots_sum_1 > p:
+                                min_range = p_pivot_1
+                            elif p_pivots_sum_0 > p:
+                                min_range = p_pivot_0
+
+                            if p_pivots_sum_0 < p:
+                                max_range = p_pivot_0
+                            elif p_pivots_sum_1 < p:
+                                max_range = p_pivot_1
+
+                            num_iters += 1
+                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
+                                p_pivot = (max_range + min_range) / 2.0
+                                found_pivot = 1
+
+                        duplicate_logit = (
+                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
+                        )
+                        num_duplicate_logit = num_min_larger
+                        num_keep = num_duplicate_logit - tl.cast(
+                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
+                        )
+                        num_kept = tl.zeros((), dtype=tl.uint32)
+
+                        # Top-k + Top-p path
+                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit
+
+        if TOPP_ENABLED and final_pivot == -float("inf"):
+            #### STANDALONE TOP-P SAMPLING ####
+            p = tl.load(P + row_id)
+            if p < 1.0:
+                # Zeroth pass: Compute avg and std from a sample block
+                offs = tl.arange(0, BLOCK_SIZE)
+                mask_n = offs < VOCAB_SIZE
+                logits_blk0 = tl.load(
+                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
+                )
+                # Exclude -inf values (e.g. from grammar bitmasks) from
+                # statistics to avoid NaN in pivot computation.
+                finite_mask = (logits_blk0 > -float("inf")) & mask_n
+                num_finite = tl.sum(finite_mask)
+                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
+                avg_logit = tl.where(
+                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
+                )
+                sq_avg_logit = tl.where(
+                    num_finite > 0,
+                    tl.sum(finite_logits * finite_logits) / num_finite,
+                    0.0,
+                )
+                std_logit = tl.sqrt(
+                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
+                )
+                max_sample = avg_logit + std_logit * 10.0
+                sum_exp_logits = 0.0
+
+                # First pass: compute max and min logits and sum_exp_logits
+                for i in range(0, NUM_TILES):
+                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                    mask_n = offs_n < VOCAB_SIZE
+                    logits_blk = tl.load(
+                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                    )
+                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
+                    # Exclude -inf from min to keep binary search bounds
+                    # finite (avoids NaN pivots).
+                    finite_blk = tl.where(
+                        logits_blk > -float("inf"), logits_blk, float("inf")
+                    )
+                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
+
+                    probs_blk = tl.exp(logits_blk - max_sample)
+                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
+                    sum_exp_logits += tl.sum(probs_blk)
+
+                # If no finite logits exist (all -inf), clamp min to
+                # max so the search converges to -inf (no masking).
+                min_logit = tl.minimum(min_logit, max_logit)
+
+                idx = tl.cast(p * 200, tl.int32)
+                idx = tl.maximum(0, tl.minimum(idx, 199))
+                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
+                sigma = sigma + tl.abs(sigma) * -0.25
+                outlier_pivot = avg_logit + std_logit * sigma
+
+                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
+                sum_outlier_probs = 0.0
+                num_outliers = tl.zeros((), dtype=tl.uint32)
+
+                # Second pass: Calculate softmax and gather outliers
+                for i in range(0, NUM_TILES):
+                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                    mask_n = offs_n < VOCAB_SIZE
+
+                    probs_blk = tl.load(
+                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                    )
+                    probs_blk = tl.exp(probs_blk - max_sample)
+                    probs_blk = probs_blk / sum_exp_logits
+
+                    outlier_mask = (probs_blk > outlier_prob) & mask_n
+                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
+                    cumulative_pos = tl.cast(
+                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
+                    )
+                    num_outliers += tl.sum(outlier_mask)
+                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
+                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)
+
+                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
+                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits
+
+                p_pivot = 1.0
+                num_iters = 0
+                min_larger_prob = 1.0
+                num_min_larger = tl.zeros((), dtype=tl.uint32)
+                p_pivots_sum = 0.0
+
+                # Third pass: Search for p_pivot
+                if sum_outlier_probs > p:
+                    min_range = outlier_prob
+                    search_range = tl.cast(num_outliers, tl.int32)
+                    search_iters = tl.cast(
+                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
+                        tl.int32,
+                    )
+
+                    found_pivot = 0
+                    while found_pivot == 0:
+                        p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
+                        p_pivots_sum_0 = 0.0
+                        min_larger_0 = 1.0
+                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
+
+                        p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
+                        p_pivots_sum_1 = 0.0
+                        min_larger_1 = 1.0
+                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
+
+                        # First pass: Calculate p_pivots_sum and min_larger
+                        for i in range(0, search_iters):
+                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                0, BLOCK_SIZE_TRUNC
+                            )
+                            mask_n_2 = offs_n < search_range
+                            probs_blk = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
+                            )
+
+                            p_pivots_sum_0 += tl.sum(
+                                probs_blk * (probs_blk > p_pivot_0)
+                            )
+                            masked_larger_0 = tl.where(
+                                probs_blk > p_pivot_0, probs_blk, 1.0
+                            )
+                            min_larger_0 = tl.minimum(
+                                min_larger_0, tl.min(masked_larger_0)
+                            )
+
+                            p_pivots_sum_1 += tl.sum(
+                                probs_blk * (probs_blk > p_pivot_1)
+                            )
+                            masked_larger_1 = tl.where(
+                                probs_blk > p_pivot_1, probs_blk, 1.0
+                            )
+                            min_larger_1 = tl.minimum(
+                                min_larger_1, tl.min(masked_larger_1)
+                            )
+
+                        # Second pass: Calculate num_min_larger
+                        for i in range(0, search_iters):
+                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
+                                0, BLOCK_SIZE_TRUNC
+                            )
+                            mask_n_2 = offs_n < search_range
+                            probs_blk = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
+                            )
+
+                            num_min_larger_0 += tl.sum(
+                                tl.abs(probs_blk - min_larger_0) < 1e-9
+                            )
+                            num_min_larger_1 += tl.sum(
+                                tl.abs(probs_blk - min_larger_1) < 1e-9
+                            )
+
+                        # Check if any of the pivots satisfy termination condition
+                        if (
+                            p_pivots_sum_1 >= p
+                            and p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
+                        ):
+                            p_pivot = p_pivot_1
+                            min_larger_prob = min_larger_1
+                            num_min_larger = num_min_larger_1
+                            p_pivots_sum = p_pivots_sum_1
+                            found_pivot = 1
+                        if (
+                            p_pivots_sum_0 >= p
+                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
+                        ):
+                            p_pivot = p_pivot_0
+                            min_larger_prob = min_larger_0
+                            num_min_larger = num_min_larger_0
+                            p_pivots_sum = p_pivots_sum_0
+                            found_pivot = 1
+
+                        # Update range
+                        if p_pivots_sum_1 > p:
+                            min_range = p_pivot_1
+                        elif p_pivots_sum_0 > p:
+                            min_range = p_pivot_0
+
+                        if p_pivots_sum_0 < p:
+                            max_range = p_pivot_0
+                        elif p_pivots_sum_1 < p:
+                            max_range = p_pivot_1
+
+                        num_iters += 1
+                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
+                            p_pivot = (max_range + min_range) / 2.0
+                            found_pivot = 1
+                else:
+                    # Re-populate the buffer with full softmax probabilities
+                    for i in range(0, NUM_TILES):
+                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                        mask_n = offs_n < VOCAB_SIZE
+
+                        probs_blk = tl.load(
+                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                        )
+                        probs_blk = tl.exp(probs_blk - max_sample)
+                        probs_blk = probs_blk / sum_exp_logits
+                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)
+
+                    found_pivot = 0
+                    while found_pivot == 0:
+                        p_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
+                        p_pivots_sum_0 = 0.0
+                        min_larger_0 = 1.0
+                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)
+
+                        p_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
+                        p_pivots_sum_1 = 0.0
+                        min_larger_1 = 1.0
+                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)
+
+                        # First pass: Calculate p_pivots_sum and min_larger
+                        for i in range(0, NUM_TILES):
+                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                            mask_n = offs_n < VOCAB_SIZE
+                            probs_blk = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
+                            )
+
+                            p_pivots_sum_0 += tl.sum(
+                                probs_blk * (probs_blk > p_pivot_0)
+                            )
+                            masked_larger_0 = tl.where(
+                                probs_blk > p_pivot_0, probs_blk, 1.0
+                            )
+                            min_larger_0 = tl.minimum(
+                                min_larger_0, tl.min(masked_larger_0)
+                            )
+
+                            p_pivots_sum_1 += tl.sum(
+                                probs_blk * (probs_blk > p_pivot_1)
+                            )
+                            masked_larger_1 = tl.where(
+                                probs_blk > p_pivot_1, probs_blk, 1.0
+                            )
+                            min_larger_1 = tl.minimum(
+                                min_larger_1, tl.min(masked_larger_1)
+                            )
+
+                        # Second pass: Calculate num_min_larger
+                        for i in range(0, NUM_TILES):
+                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                            mask_n = offs_n < VOCAB_SIZE
+                            probs_blk = tl.load(
+                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
+                            )
+
+                            num_min_larger_0 += tl.sum(
+                                tl.abs(probs_blk - min_larger_0) < 1e-9
+                            )
+                            num_min_larger_1 += tl.sum(
+                                tl.abs(probs_blk - min_larger_1) < 1e-9
+                            )
+
+                        # Check if any of the pivots satisfy termination condition
+                        if (
+                            p_pivots_sum_1 >= p
+                            and p_pivots_sum_1 - (min_larger_1 * num_min_larger_1) < p
+                        ):
+                            p_pivot = p_pivot_1
+                            min_larger_prob = min_larger_1
+                            num_min_larger = num_min_larger_1
+                            p_pivots_sum = p_pivots_sum_1
+                            found_pivot = 1
+                        if (
+                            p_pivots_sum_0 >= p
+                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
+                        ):
+                            p_pivot = p_pivot_0
+                            min_larger_prob = min_larger_0
+                            num_min_larger = num_min_larger_0
+                            p_pivots_sum = p_pivots_sum_0
+                            found_pivot = 1
+
+                        # Update range
+                        if p_pivots_sum_1 > p:
+                            min_range = p_pivot_1
+                        elif p_pivots_sum_0 > p:
+                            min_range = p_pivot_0
+
+                        if p_pivots_sum_0 < p:
+                            max_range = p_pivot_0
+                        elif p_pivots_sum_1 < p:
+                            max_range = p_pivot_1
+
+                        num_iters += 1
+                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
+                            p_pivot = (max_range + min_range) / 2.0
+                            found_pivot = 1
+
+                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_logit
+                num_duplicate_logit = num_min_larger
+                num_keep = num_duplicate_logit - tl.cast(
+                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
+                )
+                num_kept = tl.zeros((), dtype=tl.uint32)
+
+                # Top-p only path
+                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample
+
+        # Sixth pass: Apply mask and store final output.
+        # If the pivot >= max logit (or is NaN), no token would
+        # survive the strict `>` keep_mask.  Skip masking.
+        # Using `not <` instead of `>=` so that NaN is also caught.
+        if not (final_pivot < max_logit):
+            final_pivot = -float("inf")
+        elif final_pivot != -float("inf"):
+            for i in range(0, NUM_TILES):
+                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
+                mask_n = offs_n < VOCAB_SIZE
+                logits_blk = tl.load(
+                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
+                )
+                keep_mask = (logits_blk > final_pivot) & mask_n
+
+                # Duplicate logit handling
+                if num_keep < num_duplicate_logit:
+                    duplicate_mask = (
+                        tl.abs(logits_blk - duplicate_logit) < 1e-9
+                    ) & mask_n
+                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
+                    duplicate_keep_mask = (
+                        duplicate_count <= num_duplicate_logit
+                    ) & duplicate_mask
+                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
+                    num_kept += tl.sum(duplicate_keep_mask)
+                    keep_mask = keep_mask & (~duplicate_remove_mask)
+
+                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
+                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)
+
+
+def apply_top_k_top_p_triton(
+    logits: torch.Tensor,
+    k: torch.Tensor | None,
+    p: torch.Tensor | None,
+    mask_value: float = float("-inf"),
+) -> torch.Tensor:
+    """
+    Apply combined top-k and top-p masking using Triton.
+
+    Top-k is applied first (by logit value), then top-p is applied
+    to the remaining k values (by probability).
+
+    Args:
+        logits: [batch_size, vocab_size] float32 tensor, modified in-place
+        k: [batch_size] int32 tensor of top-k values per row, or None to disable top-k
+        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
+            or None to disable top-p
+        mask_value: Value for masked positions (default: -inf)
+
+    Returns:
+        The logits tensor (modified in-place)
+    """
+    assert logits.ndim == 2
+    assert logits.dtype == torch.float32
+    assert logits.is_cuda
+
+    batch_size, vocab_size = logits.shape
+
+    topk_enabled = k is not None
+    topp_enabled = p is not None
+
+    if batch_size == 0 or not (topk_enabled or topp_enabled):
+        return logits
+
+    if k is not None:
+        assert k.ndim == 1 and k.shape[0] == batch_size and k.is_cuda
+        k_ptr = k.to(torch.int32)
+    else:
+        k_ptr = logits  # Dummy pointer (won't be read)
+
+    if p is not None:
+        assert p.ndim == 1 and p.shape[0] == batch_size and p.is_cuda
+        p_ptr = p.to(torch.float32)
+    else:
+        p_ptr = logits  # Dummy pointer (won't be read)
+
+    num_sm = torch.cuda.get_device_properties(logits.device).multi_processor_count
+    NUM_PROGRAMS = min(num_sm, batch_size)
+
+    # Cache per-Triton Program buffer on each device.
+    buf_key = (logits.device, logits.dtype, vocab_size)
+    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
+    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
+        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
+        buffer = logits.new_empty((size, vocab_size))
+        _TRITON_BUFFER_CACHE[buf_key] = buffer
+    if buffer.shape[0] > NUM_PROGRAMS:
+        buffer = buffer[:NUM_PROGRAMS]
+
+    # Cache lookup table entries on each device.
+    tables = _TRITON_TABLE_CACHE.get(logits.device)
+    if tables is None:
+        normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
+        percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
+        _TRITON_TABLE_CACHE[logits.device] = (
+            normal_cdf_to_sigma_table,
+            percentile_to_std_table,
+        )
+    else:
+        normal_cdf_to_sigma_table, percentile_to_std_table = tables
+
+    _topk_topp_kernel[(NUM_PROGRAMS,)](
+        logits,
+        buffer,
+        percentile_to_std_table,
+        normal_cdf_to_sigma_table,
+        k_ptr,
+        p_ptr,
+        BATCH_SIZE=batch_size,
+        MASK_VALUE=mask_value,
+        VOCAB_SIZE=vocab_size,
+        BLOCK_SIZE=8192,
+        BLOCK_SIZE_TRUNC=4096,
+        TOPK_ENABLED=topk_enabled,
+        TOPP_ENABLED=topp_enabled,
+    )
+
+    return logits
+
+
+def reset_buffer_cache():
+    _TRITON_BUFFER_CACHE.clear()
+    _TRITON_TABLE_CACHE.clear()
+    torch.cuda.empty_cache()
```
