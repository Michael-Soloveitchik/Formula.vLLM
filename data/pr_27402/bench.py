#!/usr/bin/env python3
"""Microbench for vLLM PR #27402: [Kernel] LoRA triton kernels support PDL.

The PR enables Programmatic Dependent Launch (griddepcontrol wait / launch_dependents) in the
LoRA shrink / expand / fused-MoE-LoRA Triton kernels so consecutive kernels overlap their
prologue with the previous kernel's tail (Hopper+; sm_120 is supported). Author: "5% to 20%
kernel runtime speedup" on the shrink/expand kernels.

The bench drives the arm's `lora_shrink` + `lora_expand` custom ops the way
tests/lora/test_punica_ops.py does (LoRAKernelMeta.make / prepare_tensors / meta_args) in the
back-to-back pattern of a LoRA linear layer (shrink then expand), hidden 4096, rank 16, one
LoRA, 1 and 3 slices, decode batches of 16 and 256 tokens. CUDA-event us per shrink+expand pair
(PDL only pays off when kernels are queued back to back, so the pair is the unit).
Runs inside the task image: python3 /task/bench.py
"""
import torch

import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HIDDEN = 4096
RANK = 16
NUM_LORAS = 1
SCALING = 1.0
WARMUP = 50
ITERS = 500
# (metric, num_slices, num_tokens)
SCENARIOS = [
    ("lora_shrink_expand_1slice_t16", 1, 16),
    ("lora_shrink_expand_1slice_t256", 1, 256),
    ("lora_shrink_expand_3slice_t16", 3, 16),
    ("lora_shrink_expand_3slice_t256", 3, 256),
]


def build(nslices, tokens, gen):
    x = torch.rand(tokens, HIDDEN, device=DEVICE, dtype=DTYPE, generator=gen)
    lora_a = [torch.rand(NUM_LORAS, RANK, HIDDEN, device=DEVICE, dtype=DTYPE, generator=gen)
              for _ in range(nslices)]
    lora_b = [torch.rand(NUM_LORAS, HIDDEN, RANK, device=DEVICE, dtype=DTYPE, generator=gen)
              for _ in range(nslices)]
    shrunk = torch.zeros(nslices, tokens, RANK, device=DEVICE, dtype=torch.float32)
    out = torch.zeros(tokens, HIDDEN * nslices, device=DEVICE, dtype=DTYPE)
    token_lora_mapping = torch.zeros(tokens, dtype=torch.long, device=DEVICE)
    meta = LoRAKernelMeta.make(max_loras=NUM_LORAS, max_num_tokens=tokens, device="cuda")
    meta.prepare_tensors(token_lora_mapping)
    args = meta.meta_args(token_nums=tokens)

    def run():
        shrunk.zero_()
        triton_ops.lora_shrink(x, lora_a, shrunk, *args, SCALING)
        triton_ops.lora_expand(shrunk, lora_b, out, *args, offset_start=0, add_inputs=False)
        return out

    def ref():
        xs = x.float()
        return torch.cat([(xs @ lora_a[s][0].float().t() * SCALING) @ lora_b[s][0].float().t()
                          for s in range(nslices)], 1)

    return run, ref


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
    for name, nslices, tokens in SCENARIOS:
        run, ref = build(nslices, tokens, gen)
        out = run()
        torch.cuda.synchronize()
        r = ref()
        rel = ((out.float() - r).abs().max() / r.abs().max()).item()
        assert rel < 0.05, rel
        us = time_us(run)
        print(f"{name} {us:.3f}")
        print(f"# {name}: slices={nslices} tokens={tokens} hidden={HIDDEN} rank={RANK} "
              f"max_rel_err={rel:.4f} iters={ITERS}")


if __name__ == "__main__":
    main()
