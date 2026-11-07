#!/usr/bin/env python3
"""Microbench for vLLM PR #24573: [EPLB] Reduce EPLB Inference Overhead.

base: FusedMoE.select_experts, with EPLB enabled, picks a replica per routed expert with
      `torch.rand` + masks + records expert load inline (several small kernels, not compilable).
gold: the EPLB tail is `eplb_map_to_physical_and_record` - a torch.compile'd function using a
      (token position % replica count) pseudo-random pick and scatter_add_ for the load view.

We call the arm's real `FusedMoE.select_experts` (static method, same signature in both trees)
on synthetic routing: 256 logical experts, 32 of them replicated once (288 physical), top_k=8,
softmax routing with renormalize, like DeepSeek-V3 with 32 redundant experts. Decode batch of 64
tokens and a prefill chunk of 1024 tokens. Wall-clock us per select_experts call (the EPLB tail
is host-launch bound, which is what the PR attacks), synchronised per timed block.
The PR reports only DP16 DeepSeek-V3 serving numbers (kept reported-only).
Runs inside the task image: python3 /task/bench.py
"""
import time

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE

DEVICE = torch.device("cuda:0")
HIDDEN = 1024
NUM_LOGICAL = 256
NUM_REDUNDANT = 32
TOPK = 8
WARMUP = 50
ITERS = 500
BATCHES = [64, 1024]


def build_eplb_tables():
    num_physical = NUM_LOGICAL + NUM_REDUNDANT
    replica_count = torch.ones(NUM_LOGICAL, dtype=torch.int64)
    replica_count[:NUM_REDUNDANT] = 2
    l2p = torch.full((NUM_LOGICAL, 2), -1, dtype=torch.int64)
    l2p[:, 0] = torch.arange(NUM_LOGICAL)
    l2p[:NUM_REDUNDANT, 1] = NUM_LOGICAL + torch.arange(NUM_REDUNDANT)
    load = torch.zeros(num_physical, dtype=torch.int64)
    return (replica_count.to(DEVICE), l2p.to(DEVICE), load.to(DEVICE), num_physical)


def wall_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / ITERS


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.manual_seed(0)
    replica_count, l2p, load, num_physical = build_eplb_tables()
    for M in BATCHES:
        hidden = torch.randn(M, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        logits = torch.randn(M, NUM_LOGICAL, device=DEVICE, dtype=torch.float32)

        def run():
            return FusedMoE.select_experts(
                hidden_states=hidden,
                router_logits=logits,
                top_k=TOPK,
                use_grouped_topk=False,
                renormalize=True,
                enable_eplb=True,
                expert_map=None,
                expert_load_view=load,
                logical_to_physical_map=l2p,
                logical_replica_count=replica_count,
            )

        w, ids = run()
        torch.cuda.synchronize()
        assert ids.shape == (M, TOPK)
        assert int(ids.max()) < num_physical and int(ids.min()) >= 0
        us = wall_us(run)
        print(f"eplb_select_experts_m{M} {us:.3f}")
        print(f"# M={M} logical={NUM_LOGICAL} redundant={NUM_REDUNDANT} topk={TOPK} iters={ITERS} "
              f"load_sum={int(load.sum())}")


if __name__ == "__main__":
    main()
