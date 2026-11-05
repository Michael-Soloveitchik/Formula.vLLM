"""Oracle for PR #22940 (BlockTable.compute_slot_mapping).

Vendored per task because the merge-tree test module imports names the pinned
wheel renamed. Checks compute_slot_mapping against a plain NumPy reference for
power-of-two and non-power-of-two block sizes, random tables and positions.
Run inside the task image: pytest -q /task/tests/test_block_table_slot_mapping.py
"""
import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable


@pytest.mark.parametrize("block_size", [16, 48])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_slot_mapping_matches_reference(block_size, seed):
    rng = np.random.default_rng(seed)
    max_num_reqs, max_blocks, max_tokens = 8, 12, 512
    bt = BlockTable(block_size=block_size, max_num_reqs=max_num_reqs,
                    max_num_blocks_per_req=max_blocks,
                    max_num_batched_tokens=max_tokens, pin_memory=False,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    for r in range(max_num_reqs):
        n = int(rng.integers(1, max_blocks + 1))
        bt.add_row([int(x) for x in rng.integers(0, 10_000, size=n)], r)

    num_tokens = int(rng.integers(1, max_tokens + 1))
    req_indices = rng.integers(0, max_num_reqs, size=num_tokens).astype(np.int32)
    positions = np.empty(num_tokens, dtype=np.int64)
    for i, r in enumerate(req_indices):
        positions[i] = rng.integers(0, int(bt.num_blocks_per_row[r]) * block_size)

    bt.compute_slot_mapping(req_indices, positions)
    got = np.array(bt.slot_mapping_np[:num_tokens], copy=True)

    table = bt.block_table_np
    ref = table[req_indices, positions // block_size].astype(np.int64) * block_size + positions % block_size
    np.testing.assert_array_equal(got, ref)
