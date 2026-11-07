#!/usr/bin/env python3
"""Microbench for vLLM PR #31722: [PERF] Speed-up of GDN attention decode part (Qwen3-Next).

The PR changes ONE constant in vllm/model_executor/layers/fla/ops/fused_recurrent.py:
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 8)   # base
    BK, BV = next_power_of_2(K), min(next_power_of_2(V), 32)  # gold
i.e. the value-dim tile of `fused_recurrent_gated_delta_rule_fwd_kernel` grows from 8 to 32,
so the launch grid shrinks 4x (NV = V / BV) and each program does 4x more work.

Author's table (H200, "shapes from Qwen3-Next", kernel time in us):
    TP=2  (H=8,  HV=16, K=V=128):  B=32 30.14->20.29 | B=128 93.38->59.17 | B=512 341.63->212.70
    TP=4  (H=4,  HV=8,  K=V=128):  B=128 52.06->33.76
Same shapes here (Qwen3-Next: 16 k-heads, 32 v-heads, head dim 128, sharded by TP), decode
T=1, bf16 q/k/v, fp32 gate/beta/state, use_qk_l2norm_in_kernel=True, inplace_final_state=True —
exactly the call the GDN decode path makes. Timed with CUDA events, mean of ITERS launches.
Runs inside the task image: python3 /task/bench.py  (the arm's tree is the installed vllm).
"""
import torch

from vllm.model_executor.layers.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)

DEVICE = torch.device("cuda:0")
K = V = 128           # linear_key_head_dim / linear_value_head_dim
WARMUP = 20
ITERS = 200

# (metric, batch, H (k heads per rank), HV (v heads per rank))
SCENARIOS = [
    ("fused_recurrent_b32_tp2", 32, 8, 16),
    ("fused_recurrent_b128_tp2", 128, 8, 16),
    ("fused_recurrent_b512_tp2", 512, 8, 16),
    ("fused_recurrent_b128_tp4", 128, 4, 8),
]


def make_inputs(B, H, HV, gen):
    T = 1
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    g = torch.nn.functional.logsigmoid(
        torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32, generator=gen))
    beta = torch.rand(B, T, HV, device=DEVICE, dtype=torch.float32, generator=gen).sigmoid()
    state = torch.randn(B, HV, K, V, device=DEVICE, dtype=torch.float32, generator=gen)
    return q, k, v, g, beta, state


def run(q, k, v, g, beta, state):
    return fused_recurrent_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=K ** -0.5,
        initial_state=state,
        inplace_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )


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
    for name, B, H, HV in SCENARIOS:
        inputs = make_inputs(B, H, HV, gen)
        o, _ = run(*inputs)
        assert o.shape == (B, 1, HV, V) and torch.isfinite(o.float()).all()
        us = time_us(lambda: run(*inputs))
        print(f"{name} {us:.3f}")
        print(f"# {name}: B={B} H={H} HV={HV} K={K} V={V} iters={ITERS}")


if __name__ == "__main__":
    main()
