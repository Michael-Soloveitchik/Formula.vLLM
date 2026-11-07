#!/usr/bin/env python3
"""Microbench for vLLM PR #20308: [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel.

base: kernel_unified_attention_2d iterates over ALL KV tiles of the sequence for every query
      block and relies on the causal mask to zero the future tiles.
gold: computes the longest causal prefix any query in the block can see
      (max_seq_prefix_len) and only loops over that many tiles - roughly halves the work of a
      long prefill (author: ~1.75x kernel-side latency win at 16k prompt, Llama-3.1-8B, H100).

The bench calls the arm's `unified_attention` on a pure prefill (query_len == kv_len, one
sequence) with Llama-3.1-8B shapes: 32 query heads, 8 KV heads, head dim 128, block size 16,
bf16, prompt lengths 2000 / 8000 / 16000 (the author's benchmark_latency sweep). Inputs follow
tests/kernels/attention/test_triton_unified_attention.py at the PR's sha. CUDA-event us per
call. The PR's absolute numbers are e2e only (benchmark_latency graph; ShareGPT serving TTFT
3874 -> 3588 ms, 44.37 -> 45.90 req/s), kept reported-only.
Runs inside the task image: python3 /task/bench.py
"""
import torch

from vllm.attention.ops.triton_unified_attention import unified_attention

DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE = 32, 8, 128
BLOCK_SIZE = 16
WARMUP = 5
ITERS = 30
PROMPTS = [2000, 8000, 16000]


def build(seq_len, gen):
    q = torch.randn(seq_len, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE, dtype=DTYPE, generator=gen)
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE + 1
    k_cache = torch.randn(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE,
                          dtype=DTYPE, generator=gen)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.arange(num_blocks - 1, device=DEVICE, dtype=torch.int32).view(1, -1)
    cu_q = torch.tensor([0, seq_len], device=DEVICE, dtype=torch.int32)
    kv_lens = torch.tensor([seq_len], device=DEVICE, dtype=torch.int32)
    out = torch.empty_like(q)

    def run():
        unified_attention(
            q=q, k=k_cache, v=v_cache, out=out,
            cu_seqlens_q=cu_q, max_seqlen_q=seq_len,
            seqused_k=kv_lens, max_seqlen_k=seq_len,
            softmax_scale=HEAD_SIZE ** -0.5, causal=True,
            window_size=(-1, -1), block_table=block_table, softcap=0,
            q_descale=None, k_descale=None, v_descale=None,
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
    for seq_len in PROMPTS:
        run = build(seq_len, gen)
        out = run()
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all()
        us = time_us(run)
        print(f"prefill_attn_s{seq_len} {us:.3f}")
        print(f"# prompt={seq_len} heads={NUM_Q_HEADS}/{NUM_KV_HEADS} head={HEAD_SIZE} "
              f"block={BLOCK_SIZE} dtype={DTYPE} iters={ITERS}")


if __name__ == "__main__":
    main()
