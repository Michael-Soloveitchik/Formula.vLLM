#!/usr/bin/env python3
"""Microbench for vLLM PR #20839: [Model] Replace Mamba2 RMSNorm Gated with Fused Triton Kernel.

base: Mixer2RMSNormGated.forward_cuda = python silu-gate (fp32) -> cast -> ops.rms_norm (CUDA op)
gold: Mixer2RMSNormGated.forward_cuda = one fused Triton kernel (rms_norm_gated, mamba_ssm port)

The layer is constructed exactly as mamba_mixer2.py does (TP=1, n_groups=1, use_rms_norm=True);
tensor-parallel lookups are stubbed to world_size=1 / rank=0 so no process group is needed.
Shapes are Bamba-9B-v2 (the author's model): d_model 4096, expand 2 -> intermediate 8192, bf16,
prompt lengths 16384 and 32768 as in the PR's benchmark_latency runs (author: whole-model
latency 0.632 s -> 0.581 s at 16k, 1.280 s -> 1.179 s at 32k; this bench isolates the layer).
Timed with CUDA events, mean of ITERS calls of forward_cuda (the CUDA dispatch target).
Runs inside the task image: python3 /task/bench.py
"""
import torch

import vllm.model_executor.layers.mamba.mamba_mixer2 as mm2

# single-process stand-ins for the TP helpers the ctor calls
mm2.get_tensor_model_parallel_world_size = lambda: 1
mm2.get_tensor_model_parallel_rank = lambda: 0

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
HIDDEN = 8192         # Bamba-9B-v2 intermediate size (d_model 4096 x expand 2)
N_GROUPS = 1
EPS = 1e-5
WARMUP = 5
ITERS = 50
SEQ_LENS = [16384, 32768]


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
    torch.manual_seed(0)
    layer = mm2.Mixer2RMSNormGated(HIDDEN, N_GROUPS, use_rms_norm=True, eps=EPS).to(DEVICE)
    layer.weight.data.uniform_(0.5, 1.5)
    assert layer.tp_size == 1 and layer.n_groups == 1
    for seq in SEQ_LENS:
        x = torch.randn(seq, HIDDEN, device=DEVICE, dtype=DTYPE)
        gate = torch.randn(seq, HIDDEN, device=DEVICE, dtype=DTYPE)
        out = layer.forward_cuda(x, gate)
        ref = layer.forward_native(x, gate)
        assert out.shape == x.shape and torch.isfinite(out.float()).all()
        err = (out.float() - ref.float()).abs().max().item()
        us = time_us(lambda: layer.forward_cuda(x, gate))
        print(f"rmsnorm_gated_{seq // 1024}k {us:.3f}")
        print(f"# seq={seq} hidden={HIDDEN} dtype={DTYPE} max_abs_err_vs_native={err:.4f} iters={ITERS}")
        del x, gate, out, ref


if __name__ == "__main__":
    main()
