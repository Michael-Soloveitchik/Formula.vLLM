#!/usr/bin/env python3
"""Benchmark PR #6779: "[Core] Use array to speedup padding".

IMPORTANT FINDING (honest): this PR does NOT touch make_tensor_with_pad.
At BOTH shas (base 084a01fd3544 / gold 89a84b0bb7b3) make_tensor_with_pad is
BYTE-IDENTICAL (already the np.full / make_ndarray_with_pad array path). What
the PR actually changes is the *input* that feeds make_tensor_with_pad inside
SamplingTensors.from_sampling_metadata / from_lists:

  base: SequenceData stores tokens as a python list; prompt_token_ids is a
        property returning tuple(...). The sampler builds the padding input via
        `prompt_tokens.append(list(seq_data.prompt_token_ids))`  -> List[List[int]]
  gold: SequenceData stores tokens as array('l'); a new *_array property
        exposes it directly. The sampler does
        `prompt_tokens.append(seq_data.prompt_token_ids_array)`   -> List[array]

So the measured unit here is the FULL padding-input preparation + the two
make_tensor_with_pad calls (prompt + output), exactly as from_lists does them,
which is precisely the code region the PR's 79ms->22ms microbench covers.

APPROACH: hybrid "real tree code" load. Importing vllm.utils / vllm.sequence
wholesale pulls vllm._custom_ops (compiled C ext, absent in this CPU image), so
we cannot `import vllm`. Instead we extract the REAL, UNMODIFIED function bodies
for TORCH_DTYPE_TO_NUMPY_DTYPE / make_ndarray_with_pad / make_tensor_with_pad
from the mounted /src tree by source and exec them (torch+numpy injected). The
arm (base vs gold) is detected from the mounted sequence.py, and the token
storage + prep path is replicated EXACTLY as the diff defines it for that arm.
Both arms call the same (identical) make_tensor_with_pad; the only difference is
list-copy-from-tuple vs direct-array, which is the whole point of the PR.

Workload: input_len=1024, batch_size=2048 (from the PR body). 5 timed repeats,
1 warmup. Prints per-call ms (full prep + 2x pad), per arm.
"""
import ast
import os
import re
import time
from array import array

import numpy as np
import torch

SRC = "/src"
INPUT_LEN = 1024
BATCH_SIZE = 2048
VOCAB_SIZE = 32000
REPEATS = 5


def _extract_source(path, names):
    """Return source text for the named top-level defs/assignments in `path`."""
    with open(path) as f:
        text = f.read()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    chunks = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            seg = "".join(lines[node.lineno - 1:node.end_lineno])
            chunks[name] = seg
    missing = [n for n in names if n not in chunks]
    if missing:
        raise RuntimeError(f"could not extract {missing} from {path}")
    return "\n".join(chunks[n] for n in names)


def load_pad_fn():
    """Load the REAL make_tensor_with_pad (+ helpers) from the mounted tree."""
    utils = os.path.join(SRC, "vllm", "utils.py")
    src = _extract_source(
        utils,
        ["TORCH_DTYPE_TO_NUMPY_DTYPE", "make_ndarray_with_pad",
         "make_tensor_with_pad"],
    )
    import numpy.typing as npt
    ns = {"np": np, "npt": npt, "torch": torch,
          "Optional": __import__("typing").Optional,
          "Union": __import__("typing").Union, "List": list, "T": None}
    exec(compile(src, utils, "exec"), ns)
    return ns["make_tensor_with_pad"]


def detect_arm():
    """gold if SequenceData exposes the new *_array property / array storage."""
    seq = os.path.join(SRC, "vllm", "sequence.py")
    with open(seq) as f:
        text = f.read()
    is_gold = ("output_token_ids_array" in text
               and "array('l'" in text.replace('"', "'"))
    return "gold" if is_gold else "base"


def build_token_store(arm, rng):
    """Create the per-seq token storage exactly as SequenceData would hold it.

    base: python list (the real _prompt_token_ids / _output_token_ids type,
          and prompt_token_ids property returns a tuple).
    gold: array('l') (the real storage; *_array property returns it directly).
    """
    prompt_store = []
    output_store = []
    for _ in range(BATCH_SIZE):
        p = rng.integers(0, VOCAB_SIZE, size=INPUT_LEN, dtype=np.int64).tolist()
        # 1 output token (decode step), matching sampler penalty path.
        o = [int(rng.integers(0, VOCAB_SIZE))]
        if arm == "gold":
            prompt_store.append(array('l', p))
            output_store.append(array('l', o))
        else:
            prompt_store.append(list(p))
            output_store.append(list(o))
    return prompt_store, output_store


def prep_and_pad(arm, prompt_store, output_store, make_tensor_with_pad):
    """Replicate from_sampling_metadata loop + from_lists pad, for this arm."""
    prompt_tokens = []
    output_tokens = []
    for p, o in zip(prompt_store, output_store):
        if arm == "gold":
            # gold: seq_data.prompt_token_ids_array -> the array, no copy
            prompt_tokens.append(p)
            output_tokens.append(o)
        else:
            # base: list(seq_data.prompt_token_ids); prompt_token_ids is a
            # property returning tuple(self._prompt_token_ids), so this is
            # list(tuple(list)) -> a fresh list copy, exactly as base did.
            prompt_tokens.append(list(tuple(p)))
            output_tokens.append(list(tuple(o)))
    # from_lists: two make_tensor_with_pad calls, pad=vocab_size, int64, cpu.
    prompt_t = make_tensor_with_pad(
        prompt_tokens, VOCAB_SIZE, dtype=torch.int64, device="cpu")
    output_t = make_tensor_with_pad(
        output_tokens, VOCAB_SIZE, dtype=torch.int64, device="cpu")
    return prompt_t, output_t


def main():
    arm = detect_arm()
    make_tensor_with_pad = load_pad_fn()
    rng = np.random.default_rng(6779)
    prompt_store, output_store = build_token_store(arm, rng)

    # warmup (untimed)
    prep_and_pad(arm, prompt_store, output_store, make_tensor_with_pad)

    runs_ms = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        prep_and_pad(arm, prompt_store, output_store, make_tensor_with_pad)
        runs_ms.append((time.perf_counter() - t0) * 1000.0)

    for r in runs_ms:
        print(f"make_tensor_with_pad {arm} {r:.4f}")
    mean = sum(runs_ms) / len(runs_ms)
    print(f"# arm={arm} input_len={INPUT_LEN} batch={BATCH_SIZE} "
          f"repeats={REPEATS} mean_ms={mean:.4f} runs_ms={runs_ms} "
          f"NOTE=unit is full padding-input prep + 2x make_tensor_with_pad "
          f"(prompt+output); make_tensor_with_pad itself is identical base==gold")


if __name__ == "__main__":
    main()
