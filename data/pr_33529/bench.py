#!/usr/bin/env python3
"""Microbench for vLLM PR #33529: Triton MLA perf fixes (sm120, batch 1, long context).

base: TritonMLAImpl.forward_mqa always uses num_kv_splits = 4, so a 1-request decode over an
      80K context launches B*H*4 programs and leaves most SMs idle; the stage-1 kernel also
      re-loads q with the default cache policy and uses a plain python range.
gold: num_kv_splits = min(next_pow2(max_seq_len // 512), 2 * SM count); kernel loads use
      cache_modifier=".ca", tl.range, hoisted base offsets, and the MLA path (is_mla=True)
      skips the redundant V load since V is a prefix of the K/V latent row.

We call the arm's real `TritonMLAImpl.forward_mqa` (unbound) with a minimal `self`
(kv_lora_rank 512, scale, _sm_count), a stand-in MLA metadata (decode.block_table, decode.seq_lens,
max_seq_len) and a layer stub carrying `_k_scale`, i.e. exactly the objects the method reads.
DeepSeek-V2-Lite-like MLA decode: 16 q heads, latent 512 + rope 64 = 576, bf16 KV cache, block
size 64; contexts 8K / 32K / 80K at batch 1 and 8K at batch 8. CUDA-event us per forward_mqa.
Author (Kimi K2.5, DCP8, tok/s): 80K ctx 1 req 34 -> 52, 8 reqs 176 -> 184; 128 tok 54 -> 55.
Runs inside the task image: python3 /task/bench.py
"""
from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_HEADS = 16
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_HEAD_DIM = 128 + ROPE_DIM          # nope 128 + rope 64 -> softmax scale
BLOCK_SIZE = 64
WARMUP = 10
ITERS = 100
# (metric, batch, context)
SCENARIOS = [
    ("mla_decode_b1_kv8192", 1, 8192),
    ("mla_decode_b1_kv32768", 1, 32768),
    ("mla_decode_b1_kv81920", 1, 81920),
    ("mla_decode_b8_kv8192", 8, 8192),
]


def make_self():
    return SimpleNamespace(
        kv_lora_rank=KV_LORA_RANK,
        scale=QK_HEAD_DIM ** -0.5,
        _sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
    )


def build(batch, ctx, gen):
    q = torch.randn(batch, NUM_HEADS, KV_LORA_RANK + ROPE_DIM, device=DEVICE, dtype=DTYPE,
                    generator=gen)
    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = blocks_per_seq * batch + 1
    kv_cache = torch.randn(num_blocks, BLOCK_SIZE, KV_LORA_RANK + ROPE_DIM, device=DEVICE,
                           dtype=DTYPE, generator=gen)
    block_table = torch.arange(batch * blocks_per_seq, device=DEVICE,
                               dtype=torch.int32).view(batch, blocks_per_seq)
    seq_lens = torch.full((batch,), ctx, device=DEVICE, dtype=torch.int32)
    meta = SimpleNamespace(decode=SimpleNamespace(block_table=block_table, seq_lens=seq_lens),
                           max_seq_len=ctx)
    layer = SimpleNamespace(_k_scale=torch.ones((), device=DEVICE, dtype=torch.float32))
    me = make_self()

    def run():
        return TritonMLAImpl.forward_mqa(me, q, kv_cache, meta, layer)

    return run


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
    for name, batch, ctx in SCENARIOS:
        run = build(batch, ctx, gen)
        o, lse = run()
        torch.cuda.synchronize()
        assert o.shape == (batch, NUM_HEADS, KV_LORA_RANK) and torch.isfinite(o.float()).all()
        us = time_us(run)
        print(f"{name} {us:.3f}")
        print(f"# {name}: batch={batch} ctx={ctx} heads={NUM_HEADS} latent={KV_LORA_RANK}+{ROPE_DIM} "
              f"block={BLOCK_SIZE} iters={ITERS}")


if __name__ == "__main__":
    main()
