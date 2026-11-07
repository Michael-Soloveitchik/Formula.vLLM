#!/usr/bin/env python3
"""Microbench for vLLM PR #23928: [Model loader]: support multi-thread model weight loading.

base: DefaultModelLoader reads safetensors shards one after another with
      `safetensors_weights_iterator(files, use_tqdm_on_load)`.
gold: adds `multi_thread_safetensors_weights_iterator(files, use_tqdm_on_load, max_workers)` -
      a ThreadPoolExecutor loading shards concurrently (enabled by the loader's
      `enable_multithread_load` extra config, default 8 threads). Author: QwQ-32B (62 GB,
      NVMe, TP8) load time 20 s -> 14 s.

The bench writes NUM_FILES synthetic bf16 safetensors shards (SHARD_MB each, ~1 GB total) into
a temp dir, then times a full pass over the arm's iterator (sum of element counts as the sink):
gold = the multi-thread iterator with 8 workers, base tree (no such function) = the single-thread
iterator the loader uses there. Files sit in the page cache after being written, so this measures
the deserialise + tensor materialisation path, not disk bandwidth. Seconds per pass, best of 3.
The temp dir is removed at exit. Needs `safetensors` (a vLLM dependency) to write the shards.
Runs inside the task image: python3 /task/bench.py
"""
import os
import shutil
import tempfile
import time

import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

try:
    from vllm.model_executor.model_loader.weight_utils import (
        multi_thread_safetensors_weights_iterator,
    )
    MODE = "multi_thread"
except ImportError:
    multi_thread_safetensors_weights_iterator = None
    MODE = "single_thread"

NUM_FILES = 8
SHARD_MB = 128
TENSORS_PER_FILE = 16
MAX_WORKERS = 8
REPEATS = 3


def write_shards(root):
    files = []
    numel = SHARD_MB * 1024 * 1024 // 2 // TENSORS_PER_FILE      # bf16 elements per tensor
    for i in range(NUM_FILES):
        tensors = {f"layer{i}.w{j}": torch.randn(numel, dtype=torch.bfloat16)
                   for j in range(TENSORS_PER_FILE)}
        p = os.path.join(root, f"model-{i:05d}-of-{NUM_FILES:05d}.safetensors")
        save_file(tensors, p)
        files.append(p)
    return files


def one_pass(files):
    if multi_thread_safetensors_weights_iterator is not None:
        it = multi_thread_safetensors_weights_iterator(files, False, max_workers=MAX_WORKERS)
    else:
        it = safetensors_weights_iterator(files, False)
    n, total = 0, 0
    for name, t in it:
        n += 1
        total += t.numel()
    return n, total


def main():
    root = tempfile.mkdtemp(prefix="pr23928_")
    try:
        files = write_shards(root)
        n, total = one_pass(files)           # warm (page cache + imports)
        assert n == NUM_FILES * TENSORS_PER_FILE
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            one_pass(files)
            times.append(time.perf_counter() - t0)
        print(f"weights_load_1gb_s {min(times):.4f}")
        print(f"# mode={MODE} files={NUM_FILES} shard_mb={SHARD_MB} tensors={n} elems={total} "
              f"workers={MAX_WORKERS} runs={[round(t, 4) for t in times]}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
