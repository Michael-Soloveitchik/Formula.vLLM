#!/usr/bin/env python3
"""Microbench for vLLM PR #31984: [Kernel] Optimize Sliding Window Attention in 3D Triton Kernel.

base: kernel_unified_attention_3d (the decode / small-batch split-softmax kernel) walks every KV
      tile of the sequence and masks out-of-window keys, so a sliding-window layer still pays for
      the whole context.
gold: computes [tile_start, tile_end) from the window and only iterates in-window tiles (the
      optimisation the 2D kernel already had).

Setup mirrors the author's run (openai/gpt-oss-20b, batch size 1, TRITON_ATTN backend; its
sliding-window layers use window 128): 64 query heads, 8 KV heads, head dim 64, bf16, block
size 16, one decode query against KV lengths 4096 / 8192 / 16384, plus a batch of 8 at 8192.
The 3D path is forced the way the backend does it (seq_threshold_3D >= num_seqs, max_seqlen_q =
1, split-softmax buffers for 16 segments). Inputs follow tests/kernels/attention/
test_triton_unified_attention.py. Timed with CUDA events, mean of ITERS calls of
`unified_attention` (3D kernel + reduce_segments). The author reports 3-4% e2e latency at
6.4k-12.8k output tokens (no kernel-level number), so `reported` is null.
Runs inside the task image: python3 /task/bench.py
"""
import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE = 64, 8, 64   # gpt-oss-20b
SLIDING_WINDOW = 128
BLOCK_SIZE = 16
NUM_SEGMENTS = 16
SEQ_THRESHOLD_3D = 64
WARMUP = 20
ITERS = 200
# (metric, num_seqs, kv_len)
SCENARIOS = [
    ("sw128_decode_b1_kv4096", 1, 4096),
    ("sw128_decode_b1_kv8192", 1, 8192),
    ("sw128_decode_b1_kv16384", 1, 16384),
    ("sw128_decode_b8_kv8192", 8, 8192),
]


def next_pow2(n):
    return 1 << (n - 1).bit_length()


def build(num_seqs, kv_len, gen):
    q = torch.randn(num_seqs, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE, dtype=DTYPE, generator=gen)
    blocks_per_seq = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = blocks_per_seq * num_seqs + 1
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
                          device=DEVICE, dtype=DTYPE, generator=gen)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_seqs * blocks_per_seq, device=DEVICE,
                               dtype=torch.int32).view(num_seqs, blocks_per_seq)
    cu_q = torch.arange(num_seqs + 1, device=DEVICE, dtype=torch.int32)
    kv_lens = torch.full((num_seqs,), kv_len, device=DEVICE, dtype=torch.int32)
    out = torch.empty_like(q)
    hp = next_pow2(HEAD_SIZE)
    segm_out = torch.empty(SEQ_THRESHOLD_3D, NUM_Q_HEADS, NUM_SEGMENTS, hp,
                           device=DEVICE, dtype=torch.float32)
    segm_max = torch.empty(SEQ_THRESHOLD_3D, NUM_Q_HEADS, NUM_SEGMENTS,
                           device=DEVICE, dtype=torch.float32)
    segm_expsum = torch.empty_like(segm_max)

    def run():
        unified_attention(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=1,
            seqused_k=kv_lens, max_seqlen_k=kv_len,
            softmax_scale=HEAD_SIZE ** -0.5, causal=True,
            window_size=(SLIDING_WINDOW - 1, 0),
            block_table=block_table, softcap=0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=SEQ_THRESHOLD_3D,
            num_par_softmax_segments=NUM_SEGMENTS,
            softmax_segm_output=segm_out,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
        )
        return out

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
    for name, num_seqs, kv_len in SCENARIOS:
        run = build(num_seqs, kv_len, gen)
        out = run()
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all()
        us = time_us(run)
        print(f"{name} {us:.3f}")
        print(f"# {name}: seqs={num_seqs} kv_len={kv_len} window={SLIDING_WINDOW} "
              f"heads={NUM_Q_HEADS}/{NUM_KV_HEADS} head={HEAD_SIZE} block={BLOCK_SIZE} iters={ITERS}")


if __name__ == "__main__":
    main()
