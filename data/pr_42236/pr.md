# PR #42236: [DSv4] Improved dequant gather K cache kernel

- field: gpu_kernels  | area: v1/attention
- author: gau-nernst
- state: MERGED  created: 2026-05-10T15:29:10Z  merged: 2026-05-11T14:41:13Z
- url: https://github.com/vllm-project/vllm/pull/42236
- labels: ready, v1, verified, DSv4

## Body



## Purpose

When I did an nsys profile of DSv4 prefill, several memory-bound kernels show up as taking quite a bit of time. So I did microbenchmarks on some of them, finding out that `dequantize_and_gather_k_cache()` is currently having very low bandwidth utilization (it can only reach ~60GB/s for a single request). Hence, this PR introduces a CuteDSL version to improve it.

This kernel somewhat follows the Triton kernel design: launch `(num_requests, num_workers)`, where `num_workers` is a large number to have multiple CTAs working on a sequence. To keep things simple, 1 warp handles 1 token. The first 448 FP8 elems are dequantized, while the last 64 BF16 elems are copied as is. There is also multi-stage piplining with `cp.async` to slightly improve perf.

## Test Plan

### Microbenchmarks

<details>
<summary>Script</summary>

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import statistics

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.deepseek_v4_ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
)

HEAD_DIM = 512
TOKEN_FP8_DIM = 448
TOKEN_BF16_DIM = 64
TOKEN_SCALE_DIM = 8
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2
HEAD_BYTES = TOKEN_DATA_SIZE + TOKEN_SCALE_DIM
BLOCK_SIZE = 64
MAX_MODEL_LEN = 1048576
COMPRESS_RATIO = 4
COMPRESSED_OFFSET = 0
SWA_OFFSET = (MAX_MODEL_LEN + COMPRESS_RATIO - 1) // COMPRESS_RATIO
BYTES_PER_TOKEN = TOKEN_FP8_DIM + TOKEN_FP8_DIM // 64 + TOKEN_BF16_DIM * 2
BYTES_PER_TOKEN += HEAD_DIM * 2 + 4

COMPRESSED_FULL_SEQ_LENS = [
    1,
    8,
    32,
    128,
    512,
    2048,
    8192,
    16384,
    32000,
    MAX_MODEL_LEN // COMPRESS_RATIO,
]
SWA_CASES = [
    (8192, 128),
    (8192, 512),
    (8192, 1024),
    (16384, 128),
    (16384, 512),
    (16384, 1024),
    (MAX_MODEL_LEN, 1024),
]
BATCHED_FULL_CASE = [97, 1024, 8192, 16384]
BATCHED_SWA_CASE = ([97, 1024, 8192, 16384], [33, 128, 512, 1024])


def bench(
    label: str,
    seq_lens_list: list[int],
    gather_lens_list: list[int] | None,
    offset: int,
) -> dict[str, float | str]:
    blocks_per_req = [
        max(1, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) for seq_len in seq_lens_list
    ]
    block_offsets = [0]
    for num_blocks in blocks_per_req[:-1]:
        block_offsets.append(block_offsets[-1] + num_blocks)

    total_blocks = sum(blocks_per_req)
    max_blocks_per_req = max(blocks_per_req)
    physical_blocks = torch.randperm(total_blocks, dtype=torch.int32)

    compressed_kv = torch.randn(sum(seq_lens_list), HEAD_DIM, dtype=torch.bfloat16)
    slot_mapping = torch.cat(
        [
            (
                physical_blocks[block_offset + torch.arange(seq_len) // BLOCK_SIZE]
                * BLOCK_SIZE
                + torch.arange(seq_len) % BLOCK_SIZE
            ).to(torch.int64)
            for seq_len, block_offset in zip(seq_lens_list, block_offsets)
        ],
        dim=0,
    )

    k_cache = torch.empty(total_blocks, BLOCK_SIZE, HEAD_BYTES, dtype=torch.uint8)
    quantize_and_insert_k_cache(
        compressed_kv,
        k_cache.view(total_blocks, -1),
        slot_mapping,
        block_size=BLOCK_SIZE,
    )
    torch.cuda.synchronize()

    block_table = torch.full(
        (len(seq_lens_list), max_blocks_per_req), -1, dtype=torch.int32
    )
    for req_id, (block_offset, num_blocks) in enumerate(
        zip(block_offsets, blocks_per_req)
    ):
        block_table[req_id, :num_blocks] = physical_blocks[
            block_offset : block_offset + num_blocks
        ]

    valid_lens = seq_lens_list if gather_lens_list is None else gather_lens_list
    out_len = offset + max(valid_lens)
    out = torch.empty(
        len(seq_lens_list),
        out_len,
        HEAD_DIM,
        dtype=torch.bfloat16,
    )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    gather_lens = (
        None
        if gather_lens_list is None
        else torch.tensor(gather_lens_list, dtype=torch.int32)
    )

    ref_out = torch.empty_like(out)
    dequantize_and_gather_k_cache_triton(
        ref_out, k_cache, seq_lens, gather_lens, block_table, BLOCK_SIZE, offset
    )
    dequantize_and_gather_k_cache(
        out, k_cache, seq_lens, gather_lens, block_table, BLOCK_SIZE, offset
    )
    torch.cuda.synchronize()
    for req_id, length in enumerate(valid_lens):
        torch.testing.assert_close(
            out[req_id, offset : offset + length],
            ref_out[req_id, offset : offset + length],
            rtol=0,
            atol=0,
        )

    torch.cuda.synchronize()

    cutedsl_ms = statistics.median(bench_gpu_time_with_cupti(
        lambda: dequantize_and_gather_k_cache(
            out, k_cache, seq_lens, gather_lens, block_table, BLOCK_SIZE, offset
        )
    ))
    triton_ms = statistics.median(bench_gpu_time_with_cupti(
        lambda: dequantize_and_gather_k_cache_triton(
            ref_out, k_cache, seq_lens, gather_lens, block_table, BLOCK_SIZE, offset
        )
    ))
    moved_bytes = sum(valid_lens)
    moved_bytes *= BYTES_PER_TOKEN
    cutedsl_gbps = moved_bytes / (cutedsl_ms * 1e-3) * 1e-9
    triton_gbps = moved_bytes / (triton_ms * 1e-3) * 1e-9
    return {
        "case": label,
        "triton_us": triton_ms * 1e3,
        "cutedsl_us": cutedsl_ms * 1e3,
        "triton_GB/s": triton_gbps,
        "cutedsl_GB/s": cutedsl_gbps,
        "speedup": triton_ms / cutedsl_ms,
    }


def main() -> None:
    torch.set_default_device("cuda")

    print(f"\ncompressed full gather (offset={COMPRESSED_OFFSET})")
    full_rows = [
        bench(str(seq_len), [seq_len], None, COMPRESSED_OFFSET)
        for seq_len in COMPRESSED_FULL_SEQ_LENS
    ]
    full_rows.append(bench("batched", BATCHED_FULL_CASE, None, COMPRESSED_OFFSET))
    print(
        pd.DataFrame(full_rows)
        .rename(columns={"case": "k_len"})
        .to_markdown(index=False, floatfmt=".2f")
    )

    print(f"\nuncompressed swa tail (offset={SWA_OFFSET})")
    swa_rows = [
        bench(f"{seq_len}/{gather_len}", [seq_len], [gather_len], SWA_OFFSET)
        for seq_len, gather_len in SWA_CASES
    ]
    swa_rows.append(bench("batched", *BATCHED_SWA_CASE, SWA_OFFSET))
    print(
        pd.DataFrame(swa_rows)
        .rename(columns={"case": "seq/gather"})
        .to_markdown(index=False, floatfmt=".2f")
    )


if __name__ == "__main__":
    main()
```

</details>

On GB200

compressed full gather (offset=0)
| k_len   |   triton_us |   cutedsl_us |   triton_GB/s |   cutedsl_GB/s |   speedup |
|:--------|------------:|-------------:|--------------:|---------------:|----------:|
| 1       |        5.82 |         2.98 |          0.28 |           0.54 |      1.96 |
| 8       |        5.47 |         2.59 |          2.36 |           4.97 |      2.11 |
| 32      |        5.89 |         2.62 |          8.76 |          19.65 |      2.24 |
| 128     |        6.40 |         2.69 |         32.22 |          76.71 |      2.38 |
| 512     |       17.31 |         2.94 |         47.65 |         280.17 |      5.88 |
| 2048    |       61.02 |         3.30 |         54.07 |        1001.01 |     18.51 |
| 8192    |      215.95 |         5.44 |         61.11 |        2425.98 |     39.70 |
| 16384   |      429.33 |         7.39 |         61.48 |        3570.70 |     58.08 |
| 32000   |      834.79 |        12.10 |         61.75 |        4261.90 |     69.01 |
| 262144  |     6750.22 |        71.01 |         62.56 |        5947.41 |     95.06 |
| batched |      429.34 |        11.33 |         96.42 |        3654.47 |     37.90 |

uncompressed swa tail (offset=262144)
| seq/gather   |   triton_us |   cutedsl_us |   triton_GB/s |   cutedsl_GB/s |   speedup |
|:-------------|------------:|-------------:|--------------:|---------------:|----------:|
| 8192/128     |        6.46 |         2.98 |         31.90 |          69.29 |      2.17 |
| 8192/512     |       17.47 |         3.07 |         47.21 |         268.50 |      5.69 |
| 8192/1024    |       32.45 |         3.20 |         50.84 |         515.52 |     10.14 |
| 16384/128    |        6.56 |         3.04 |         31.43 |          67.83 |      2.16 |
| 16384/512    |       17.98 |         3.04 |         45.86 |         271.33 |      5.92 |
| 16384/1024   |       32.26 |         3.17 |         51.14 |         520.73 |     10.18 |
| 1048576/1024 |       31.52 |         3.26 |         52.34 |         505.41 |      9.66 |
| batched      |       32.19 |         3.84 |         84.92 |         711.94 |      8.38 |

### Prefill benchmark

DSv4-Flash, 4x GB200

8k-1, concurrency 256

Branch | TPGS | TTFT
---|---|---
main | 35732 | 14.65s
This PR | 37390 (+4.6%) | 13.68s

128k-1, concurrency 256

Branch | TPGS | TTFT
---|---|---
main | 30502 | 267.68s
This PR | 32513 (+6.6%) | 250.58s

### Accuracy eval

DSv4-Flash. GSM8K: 0.9553

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>



## Comments

### zyongye @ 2026-05-10T23:00:32Z

Do we have unit test for this op?
---
### gau-nernst @ 2026-05-11T01:39:49Z

@zyongye Added unit test (courtesy to Codex)
---
### mergify @ 2026-05-11T04:00:15Z

Hi @gau-nernst, the pre-commit checks have failed. Please run:

```bash 
uv pip install pre-commit>=4.5.1
pre-commit install
pre-commit run --all-files
```

Then, commit the changes and push to your branch.

For future commits, `pre-commit` will run automatically on changed files before each commit.

> [!TIP]
> <details>
> <summary>Is <code>mypy</code> failing?</summary>
> <br/>
> <code>mypy</code> is run differently in CI. If the failure is related to this check, please use the following command to run it locally:
>
> ```bash
> # For mypy (substitute "3.10" with the failing version if needed)
> pre-commit run --hook-stage manual mypy-3.10
> ```
> </details>
---