#!/usr/bin/env python3
"""Microbench for vLLM PR #32425: [LoRA] Update LoRA expand kernel heuristic.

The PR changes one line of the default (untuned-GPU) config in vllm/lora/ops/triton_ops/utils.py:
    block_n: 128  ->  max(64, next_power_of_2(128 // num_slices))
so multi-slice layers (QKVParallelLinearWithLoRA, num_slices=3, small N per slice) launch more,
smaller blocks of `_lora_expand_kernel`. Author (profiling, gpt-oss-20b + rank-16 LoRA):
_lora_expand_kernel ~14 us -> ~10 us; e2e output throughput +1%.

The bench drives the arm's `lora_expand` custom op exactly like tests/lora/test_punica_ops.py
(LoRAKernelMeta.make / prepare_tensors / meta_args), with 3 slices x hidden 1024, rank 16, one
LoRA, decode batches of 16 and 256 tokens (plus a 1-slice control that the heuristic leaves at
block_n 128). CUDA-event us per lora_expand call. The default config is only used when no tuned
JSON exists for the GPU (true for an RTX 5070 Ti).
Runs inside the task image: python3 /task/bench.py
"""
import torch

import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
RANK = 16
NUM_LORAS = 1
WARMUP = 50
ITERS = 500
# (metric, num_slices, hidden per slice, num_tokens)
SCENARIOS = [
    ("lora_expand_3slice_h1024_t16", 3, 1024, 16),
    ("lora_expand_3slice_h1024_t256", 3, 1024, 256),
    ("lora_expand_1slice_h4096_t16", 1, 4096, 16),
]


def build(nslices, hidden, tokens, gen):
    inputs = torch.rand(nslices, tokens, RANK, device=DEVICE, dtype=DTYPE, generator=gen)
    lora_b = [torch.rand(NUM_LORAS, hidden, RANK, device=DEVICE, dtype=DTYPE, generator=gen)
              for _ in range(nslices)]
    out = torch.rand(tokens, hidden * nslices, device=DEVICE, dtype=DTYPE, generator=gen)
    token_lora_mapping = torch.zeros(tokens, dtype=torch.long, device=DEVICE)  # all -> lora 0
    meta = LoRAKernelMeta.make(max_loras=NUM_LORAS, max_num_tokens=tokens, device="cuda")
    meta.prepare_tensors(token_lora_mapping)
    args = meta.meta_args(token_nums=tokens)

    def run():
        triton_ops.lora_expand(inputs, lora_b, out, *args, offset_start=0, add_inputs=True)
        return out

    return run, inputs, lora_b


def reference(inputs, lora_b):
    return torch.cat([inputs[s].float() @ lora_b[s][0].float().t() for s in range(len(lora_b))], 1)


def time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / ITERS


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(0)
    for name, nslices, hidden, tokens in SCENARIOS:
        run, inputs, lora_b = build(nslices, hidden, tokens, gen)
        before = run().clone()
        run()
        torch.cuda.synchronize()
        delta = (run() - before).float()           # add_inputs=True: each call adds x @ B^T
        err = (delta / 2 - reference(inputs, lora_b)).abs().max().item()
        assert err < 0.1, err
        us = time_us(run)
        print(f"{name} {us:.3f}")
        print(f"# {name}: slices={nslices} hidden={hidden} tokens={tokens} rank={RANK} "
              f"max_abs_err={err:.4f} iters={ITERS}")


if __name__ == "__main__":
    main()
