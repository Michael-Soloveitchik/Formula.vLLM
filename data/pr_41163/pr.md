# PR #41163: [Perf] Optimize `AllPool.forward` by slicing first, 51% faster in the method level benchmark

- field: gpu_kernels  | area: model_executor/layers
- author: yewentao256
- state: MERGED  created: 2026-04-28T19:41:29Z  merged: 2026-04-29T15:11:05Z
- url: https://github.com/vllm-project/vllm/pull/41163
- labels: ready

## Body

## Purpose

Part of https://github.com/vllm-project/vllm/issues/35631

## Test

### Acc

Covered in current CI

### Perf

```bash
import gc
import statistics
import time

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
HIDDEN_SIZE = 4096

FULL_LENGTHS = [128] * 64
# FULL_LENGTHS = [32 + (i * 17) % 224 for i in range(64)]  # varying-length case
SUBGROUP_OFFSET = 16
SUBGROUP_SIZE = 32

WARMUP = 500
ITERS = 4000
REPEATS = 5


def build_metadata(device: torch.device) -> PoolingMetadata:
    tasks = ["classify"] * len(FULL_LENGTHS)
    for i in range(SUBGROUP_OFFSET, SUBGROUP_OFFSET + SUBGROUP_SIZE):
        tasks[i] = "token_classify"

    metadata = PoolingMetadata(
        prompt_lens=torch.tensor(FULL_LENGTHS, dtype=torch.int32),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task=t) for t in tasks],
        pooling_states=[PoolingStates() for _ in tasks],
    )

    query_start_loc_gpu = torch.tensor(
        [0] + list(np.cumsum(FULL_LENGTHS)),
        dtype=torch.int32,
        device=device,
    )
    metadata.build_pooling_cursor(
        np.asarray(FULL_LENGTHS, dtype=np.int32),
        seq_lens_cpu=metadata.prompt_lens,
        device=device,
        query_start_loc_gpu=query_start_loc_gpu,
    )
    return metadata[SUBGROUP_OFFSET:SUBGROUP_OFFSET + SUBGROUP_SIZE]


def measure_us(pooler: AllPool, hidden_states: torch.Tensor, metadata: PoolingMetadata):
    samples = []
    for _ in range(REPEATS):
        for _ in range(WARMUP):
            pooler(hidden_states, metadata)
        torch.cuda.synchronize()

        # Use wall time, not CUDA events: current AllPool.forward may include
        # host-visible sync via .tolist() / .item().
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            pooler(hidden_states, metadata)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gc.enable()

        samples.append((t1 - t0) * 1e6 / ITERS)
    return samples


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device = torch.device(DEVICE)
    hidden_states = torch.randn(
        sum(FULL_LENGTHS),
        HIDDEN_SIZE,
        device=device,
        dtype=DTYPE,
    )

    cfg = VllmConfig()
    cfg.scheduler_config.enable_chunked_prefill = False

    with set_current_vllm_config(cfg):
        pooler = AllPool()
        metadata = build_metadata(device)

        out = pooler(hidden_states, metadata)
        assert len(out) == SUBGROUP_SIZE
        assert sum(x.shape[0] for x in out) == sum(
            FULL_LENGTHS[SUBGROUP_OFFSET:SUBGROUP_OFFSET + SUBGROUP_SIZE]
        )

        samples = measure_us(pooler, hidden_states, metadata)

    print(f"AllPool.forward median: {statistics.median(samples):.2f} us")
    print(f"runs: {[round(x, 2) for x in samples]}")


if __name__ == "__main__":
    main()
```

Now

```bash
AllPool.forward median: 39.65 us
runs: [39.67, 39.65, 39.72, 39.59, 39.47]
```

Main

```bash
AllPool.forward median: 59.89 us
runs: [60.19, 59.4, 60.06, 59.59, 59.89]
```


## Comments

### mergify @ 2026-04-28T19:42:08Z

Hi @yewentao256, the pre-commit checks have failed. Please run:

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