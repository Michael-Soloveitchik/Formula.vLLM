# Method-level benchmark from PR #41163 body (author: yewentao256), verbatim. Run inside the task image: python3 /task/bench.py
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
