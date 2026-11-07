#!/usr/bin/env python3
"""Microbench for vLLM PR #32403: [Performance] Improve Triton prefill attention kernel's performance.

base: vllm/v1/attention/ops/triton_prefill_attention.py `_fwd_kernel` masks with float("-inf"),
      keeps extra -inf guard math for sliding window, and uses tl.exp.
gold: FA2-style rewrite (LightLLM): -1e8 mask value, sm_scale pre-multiplied by 1/ln2 so the
      softmax runs on tl.math.exp2, simpler running-max / rescale bookkeeping.

The bench calls the arm's `context_attention_fwd` (same signature in both trees) on a prefill
batch like the author's (batch 8 x input_len 2048; plus 16 x 1024). Model-like shapes:
32 query heads, 8 KV heads (GQA 4), head dim 128, bf16, causal. Timed with CUDA events, mean of
ITERS kernel calls. The author reports only e2e numbers from an offline Triton-backend run
(bs 8 x 2048: 113.12 -> 109.99 ms latency, 33.9996 -> 34.9690 req/s), kept as reported-only.
Runs inside the task image: python3 /task/bench.py
"""
import torch

from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
WARMUP = 10
ITERS = 50
# (metric, batch, seq_len)
SCENARIOS = [
    ("prefill_attn_b8_s2048", 8, 2048),
    ("prefill_attn_b16_s1024", 16, 1024),
]


def build(batch, seq, gen):
    total = batch * seq
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    o = torch.empty_like(q)
    b_start_loc = torch.arange(0, total, seq, device=DEVICE, dtype=torch.int32)
    b_seq_len = torch.full((batch,), seq, device=DEVICE, dtype=torch.int32)

    def run():
        context_attention_fwd(q, k, v, o, b_start_loc, b_seq_len, seq, is_causal=True)
        return o

    return run


def reference(q, k, v, batch, seq):
    qh = q.view(batch, seq, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2).float()
    rep = NUM_Q_HEADS // NUM_KV_HEADS
    kh = k.view(batch, seq, NUM_KV_HEADS, HEAD_DIM).repeat_interleave(rep, 2).transpose(1, 2).float()
    vh = v.view(batch, seq, NUM_KV_HEADS, HEAD_DIM).repeat_interleave(rep, 2).transpose(1, 2).float()
    out = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
    return out.transpose(1, 2).reshape(batch * seq, NUM_Q_HEADS, HEAD_DIM)


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
    # correctness on a small instance (same code path), then the timed shapes
    b, s = 2, 256
    q = torch.randn(b * s, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    k = torch.randn(b * s, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    v = torch.randn(b * s, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE, generator=gen)
    o = torch.empty_like(q)
    context_attention_fwd(q, k, v, o, torch.tensor([0, s], device=DEVICE, dtype=torch.int32),
                          torch.tensor([s, s], device=DEVICE, dtype=torch.int32), s, is_causal=True)
    err = (o.float() - reference(q, k, v, b, s)).abs().max().item()
    assert err < 5e-2, err
    for name, batch, seq in SCENARIOS:
        run = build(batch, seq, gen)
        us = time_us(run)
        print(f"{name} {us:.3f}")
        print(f"# {name}: batch={batch} seq={seq} heads={NUM_Q_HEADS}/{NUM_KV_HEADS} "
              f"head_dim={HEAD_DIM} dtype={DTYPE} small_case_max_err={err:.4f} iters={ITERS}")


if __name__ == "__main__":
    main()
