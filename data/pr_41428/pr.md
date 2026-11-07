# PR #41428: [DSv4] Improved fused Indexer Q quant kernel

- field: gpu_kernels  | area: v1/attention
- author: gau-nernst
- state: MERGED  created: 2026-04-30T23:59:24Z  merged: 2026-05-09T08:20:33Z
- url: https://github.com/vllm-project/vllm/pull/41428
- labels: ready, v1

## Body


## Purpose

Replace `_fused_indexer_q_rope_mxfp4_kernel` Triton kernel with a CuteDSL version to utilize 256-bit loads. Initially I wrote this in CUDA C++, but couldn't build vLLM from source, so asked Codex to port it over to CuteDSL. Hopefully this will be the first of many CuteDSL kernels to come in vLLM.

Update: I keep the original Triton implementation for fallback (potentially for ROCm). Also put CuteDSL kernel in a separate file and add import guards for platform doesn't have CuteDSL.

### Microbenchmarks

<details>
<summary>Benchmark script</summary>

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)
from flashinfer.testing import bench_gpu_time_with_cupti
import statistics

NUM_HEADS = 64
HEAD_DIM = 128
ROPE_DIM = 64
MAX_POS = 100_000
TOKENS = [1, 8, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
ROPE_DTYPE = torch.float32


def make_inputs(num_tokens: int):
    positions = torch.randint(MAX_POS, (num_tokens,), dtype=torch.int64)
    query = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16)
    cos_sin_cache = torch.randn(MAX_POS, ROPE_DIM, dtype=ROPE_DTYPE)
    weights = torch.randn(num_tokens, NUM_HEADS, dtype=torch.bfloat16)
    return (
        positions,
        query,
        cos_sin_cache,
        weights,
        HEAD_DIM**-0.5,
        NUM_HEADS**-0.5,
        True,
    )


def benchmark(num_tokens: int):
    torch.set_default_device("cuda")

    kernel_args = make_inputs(num_tokens)
    timings = bench_gpu_time_with_cupti(lambda: fused_indexer_q_rope_quant(*kernel_args))
    median_ms = statistics.median(timings)

    bytes_per_token = 8  # position int64
    bytes_per_token += NUM_HEADS * HEAD_DIM * 2  # q in bf16
    bytes_per_token += ROPE_DIM * torch.empty((), dtype=ROPE_DTYPE).element_size()
    bytes_per_token += NUM_HEADS * 2  # weights in bf16
    bytes_per_token += NUM_HEADS * HEAD_DIM // 2  # q out fp4
    bytes_per_token += NUM_HEADS * HEAD_DIM // 32  # q_scale uint8
    bytes_per_token += NUM_HEADS * 4  # weights out fp32
    total_bytes = bytes_per_token * num_tokens

    return median_ms, total_bytes


if __name__ == "__main__":
    for num_tokens in TOKENS:
        median_ms, moved_bytes = benchmark(num_tokens)
        bandwidth_gb_s = moved_bytes / (median_ms * 1e-3) * 1e-9
        print(
            f"T={num_tokens:6d}  "
            f"{median_ms * 1e3:7.2f} us  "
            f"BW {bandwidth_gb_s:7.1f} GB/s  "
        )
```

</details>

Result on GB200

Couldn't quite get to SOL yet (8TB/s), but still should be a good improvement for now.

T (tokens) | Before (us) | After (us) | Before BW (GB/s) | After BW (GB/s) | Speedup
-- | -- | -- | -- | -- | --
1 | 3.71 | 2.69 | 5.7 | 8.0 | 1.38×
8 | 3.90 | 2.50 | 43.6 | 68.5 | 1.56×
32 | 4.35 | 3.01 | 156.3 | 227.5 | 1.44×
128 | 8.48 | 3.46 | 320.8 | 792.0 | 2.45×
256 | 13.41 | 4.00 | 405.8 | 1368.6 | 3.35×
512 | 22.37 | 5.66 | 486.5 | 1933.0 | 3.95×
1024 | 40.06 | 7.78 | 543.3 | 2816.0 | 5.15×
2048 | 75.71 | 12.83 | 575.0 | 3412.9 | 5.90×
4096 | 147.58 | 21.50 | 589.9 | 4073.1 | 6.86×
8192 | 290.43 | 36.06 | 599.6 | 4857.4 | 8.05×
16384 | 576.05 | 63.14 | 604.6 | 5549.2 | 9.12×

### E2E benchmarks

DSv4-Flash, 4xGB200, 8k-1k, concurrency 256

Setting | Throughput (tok/s) | TTFT (P50) | TPOT (P50)
---|---|---|---
Before c7aa186d | 66877 | 1.39s | 32.62 ms
After | 67910 | 1.35s | 32.55ms

## Test Plan

```
pytest tests/kernels/test_fused_indexer_q_rope_quant.py
```

## Test Result

Existing tests pass. GSM8k 0.9484

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### gau-nernst @ 2026-05-06T06:42:44Z

Pending #41603 investigation, since this PR introduces even a bigger change (completely new kernel)
---