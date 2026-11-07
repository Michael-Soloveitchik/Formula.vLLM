#!/usr/bin/env python3
"""Microbench for vLLM PR #27673: [Core] Early return in SlidingWindowManager.remove_skipped_blocks.

remove_skipped_blocks runs under allocate_slots on every scheduling step of a sliding-window
model; most calls remove nothing. pre-PR: always fetch the request's block list, walk the (empty)
range, build an empty list and call block_pool.free_blocks([]). PR: return early when the
context is shorter than the window (last_useful_block <= 0) or when the block just below the
window is already the null block (everything older was freed on an earlier call).
Author: no-op call 6.3 us -> 1.6 us.

The bench builds a real BlockPool + SlidingWindowManager (block 16, window 4096), gives one
request 2048 real blocks (32768 tokens), lets a first call free the blocks below the window,
then times the two no-op cases the PR short-circuits:
  remove_skipped_blocks           - steady state: blocks below the window already freed
  remove_skipped_blocks_short_ctx - context shorter than the sliding window
us per call, CPU only. Runs inside the task image: python3 /task/bench.py
"""
import time

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec

BLOCK_SIZE = 16
SLIDING_WINDOW = 4096
NUM_GPU_BLOCKS = 8192
REQ_BLOCKS = 2048            # 32768 tokens allocated to the request
WARMUP = 2000
ITERS = 200000


def us_per_call(fn):
    for _ in range(WARMUP):
        fn()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    return (time.perf_counter() - t0) * 1e6 / ITERS


def main():
    spec = SlidingWindowSpec(block_size=BLOCK_SIZE, num_kv_heads=8, head_size=128,
                             dtype=torch.bfloat16, sliding_window=SLIDING_WINDOW)
    pool = BlockPool(num_gpu_blocks=NUM_GPU_BLOCKS, enable_caching=True)
    mgr = SlidingWindowManager(spec, pool, kv_cache_group_id=0)
    req = "req-0"
    mgr.req_to_blocks[req] = pool.get_new_blocks(REQ_BLOCKS)
    computed = REQ_BLOCKS * BLOCK_SIZE          # 32768 computed tokens

    mgr.remove_skipped_blocks(req, computed)    # frees the blocks below the window once
    blocks = mgr.req_to_blocks[req]
    last_useful_block = (computed - SLIDING_WINDOW + 1) // BLOCK_SIZE
    assert blocks[last_useful_block - 1] is pool.null_block
    assert blocks[last_useful_block] is not pool.null_block
    free_before = pool.get_num_free_blocks()

    us_steady = us_per_call(lambda: mgr.remove_skipped_blocks(req, computed))
    us_short = us_per_call(lambda: mgr.remove_skipped_blocks(req, SLIDING_WINDOW // 2))
    assert pool.get_num_free_blocks() == free_before       # both paths removed nothing

    print(f"remove_skipped_blocks {us_steady:.4f}")
    print(f"remove_skipped_blocks_short_ctx {us_short:.4f}")
    print(f"# block={BLOCK_SIZE} window={SLIDING_WINDOW} req_blocks={REQ_BLOCKS} "
          f"computed={computed} iters={ITERS}")


if __name__ == "__main__":
    main()
