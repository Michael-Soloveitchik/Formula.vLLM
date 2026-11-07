#!/usr/bin/env python3
"""Microbench for vLLM PR #42311: [Model] [Perf] Use flatten for Qwen3.5's GDN output projection.

base: GatedDeltaNetAttention._output_projection merges heads with
      einops `rearrange(core_attn_out, "... h d -> ... (h d)")` (python-level pattern parsing
      and validation on every forward call).
gold: `core_attn_out.flatten(-2)`.

We call the arm's real `GatedDeltaNetAttention._output_projection` (unbound) with a minimal
stand-in `self` carrying the two layers the method touches: `norm` = the real RMSNormGated
(head_v_dim, eps 1e-6, norm_before_gate=False, as the layer builds it) and `out_proj` = a plain
Linear returning (y, None) like RowParallelLinear does (TP=1, no process group needed).
Shapes are Qwen3.5-0.8B (the author's profiled model): 16 value heads x head dim 128 -> hidden
1024, decode batch of 8 tokens, fp16, eager. Wall-clock us per call, mean of ITERS calls.
The author's numbers are per-call profiler events for the merge op itself (rearrange 23.26 us
vs aten::flatten 2.83 us on CUDA eager; 41.74 vs 1.95 us on CPU), so they are kept as a
reported-only experiment; the measured metric is the whole _output_projection call.
Runs inside the task image: python3 /task/bench.py
"""
import time

import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
NUM_V_HEADS = 16      # Qwen3.5-0.8B linear_num_value_heads
HEAD_V_DIM = 128      # linear_value_head_dim
HIDDEN = 1024         # hidden_size
NUM_TOKENS = 8        # a decode batch
WARMUP = 200
ITERS = 2000


class OutProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(NUM_V_HEADS * HEAD_V_DIM, HIDDEN, bias=False)

    def forward(self, x):
        return self.linear(x), None


class Shell(nn.Module):
    """Just the attributes _output_projection reads."""

    def __init__(self):
        super().__init__()
        self.norm = RMSNormGated(HEAD_V_DIM, eps=1e-6, group_size=None, norm_before_gate=False)
        self.out_proj = OutProj()


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        shell = Shell().to(DEVICE, DTYPE)
        proj = GatedDeltaNetAttention._output_projection
        core = torch.randn(1, NUM_TOKENS, NUM_V_HEADS, HEAD_V_DIM, device=DEVICE, dtype=DTYPE)
        z = torch.randn(1, NUM_TOKENS, NUM_V_HEADS, HEAD_V_DIM, device=DEVICE, dtype=DTYPE)
        output = torch.zeros(NUM_TOKENS, HIDDEN, device=DEVICE, dtype=DTYPE)

        proj(shell, core, z, output, NUM_TOKENS)
        torch.cuda.synchronize()
        assert torch.isfinite(output.float()).all() and output.abs().sum() > 0

        for _ in range(WARMUP):
            proj(shell, core, z, output, NUM_TOKENS)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            proj(shell, core, z, output, NUM_TOKENS)
        torch.cuda.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / ITERS
    print(f"gdn_output_projection {us:.3f}")
    print(f"# tokens={NUM_TOKENS} v_heads={NUM_V_HEADS} head_v_dim={HEAD_V_DIM} hidden={HIDDEN} "
          f"dtype={DTYPE} iters={ITERS} (wall-clock per _output_projection call)")


if __name__ == "__main__":
    main()
