#!/usr/bin/env python3
"""Benchmark EPLB rebalance for vLLM PR #29499 (Optimize EPLB with numpy).

All metrics are in **milliseconds** (`<metric> <ms>` lines, parse_workload.py
GENERIC parser; metadata.json declares unit "ms").

Author's setting: Qwen3-Next-80B-A3B, `-dp 8 --enable-expert-parallel
--eplb-config.num_redundant_experts 32` -> 512 logical + 32 redundant = 544
physical experts over 8 EP ranks = 68 local experts per rank, 48 MoE layers.
Profile of the 3rd EPLB call: `rearrange` 1370 ms (main) -> 1077 ms (PR).

Headline  rearrange_e2e (ms): the per-layer CPU index-mapping path the PR
  rewrote, run exactly as `rearrange_expert_weights_inplace` runs it -- the
  arm's REAL `move_to_buffer` + `move_from_buffer` from the tree on PYTHONPATH,
  for every layer and for every EP rank (8 x 48 calls per pass). What a single
  CPU container cannot do is the P2P traffic and the GPU copies, so
  `P2POp` / `batch_isend_irecv` / `get_global_rank` are stubbed to no-ops and
  the expert weights are tiny CPU tensors (num_local x 8). The weight bytes are
  identical across arms and are not what the PR changed; the index logic is.
  Baseline receives `.tolist()` rows and the PR arm numpy rows, as in the real
  callers. One rank's share is rearrange_e2e / NUM_RANKS.

Secondary (same geometry, best-of-INNER):
  get_ep_ranks_with_expert  ISOLATED IMPORT of the real function: baseline loops
      get_ep_ranks_with_expert over every unique expert, PR arm makes one
      get_ep_ranks_with_experts_batch call.
  move_to_buffer / move_from_buffer  INLINE FROM DIFF: the index-mapping body
      of each function (the only arm difference), copied verbatim from
      rebalance_execute.py at base_sha 0ada960a2076 / merge_sha 6170d47d2220,
      for one rank, without the sends / tensor copies.

The arm (baseline vs PR) is auto-detected from the imported module's API.
"""

import statistics
import sys
import time

import numpy as np
import torch

from vllm.distributed.eplb import rebalance_execute as R

# Qwen3-Next-80B-A3B EPLB geometry (the author's benchmark server)
NUM_RANKS = 8
NUM_LOGICAL = 512
NUM_REDUNDANT = 32
NUM_PHYS = NUM_LOGICAL + NUM_REDUNDANT          # 544
NUM_LOCAL = NUM_PHYS // NUM_RANKS               # 68
NUM_LAYERS = 48
HIDDEN = 8                                      # tiny weights: index logic only
INNER = 20
REPEATS = 5
E2E_REPEATS = 3

GOLD = hasattr(R, "get_ep_ranks_with_experts_batch")


def _make_indices(seed, num_layers=1):
    """(num_layers, NUM_PHYS) old / new physical->logical maps: every logical
    expert present at least once, NUM_REDUNDANT replicas, new = rebalanced
    permutation of old (every rank keeps some, sends some, receives some)."""
    rng = np.random.default_rng(seed)
    old = np.empty((num_layers, NUM_PHYS), dtype=np.int64)
    new = np.empty((num_layers, NUM_PHYS), dtype=np.int64)
    for L in range(num_layers):
        base = np.concatenate([np.arange(NUM_LOGICAL),
                               rng.integers(0, NUM_LOGICAL, size=NUM_REDUNDANT)])
        old[L] = rng.permutation(base)
        new[L] = rng.permutation(old[L])
    return old, new


def _best_ms(fn):
    best = float("inf")
    for _ in range(INNER):
        t0 = time.perf_counter()
        fn()
        dt = (time.perf_counter() - t0) * 1e3
        if dt < best:
            best = dt
    return best


# ---------------- get_ep_ranks_with_expert (ISOLATED IMPORT) -------------
def bench_get_ep_ranks(old, new):
    if GOLD:
        experts = np.unique(old)

        def run():
            R.get_ep_ranks_with_experts_batch(experts, NUM_LOCAL, old, new)
    else:
        old_l, new_l = old.tolist(), new.tolist()
        experts = sorted(set(old_l))

        def run():
            for e in experts:
                R.get_ep_ranks_with_expert(e, NUM_LOCAL, old_l, new_l)
    return run


# ---------------- move_to_buffer mapping body (INLINE FROM DIFF) ---------
# BASELINE: pure-python per-row double loop (sends / copies excluded).
def _mtb_base(old_l, new_l, ep_rank):
    n = NUM_LOCAL
    local2global = lambda i: ep_rank * n + i
    is_unchanged = [old_l[local2global(i)] == new_l[local2global(i)]
                    for i in range(n)]
    is_received_locally = is_unchanged[:]
    for src in range(n):
        src_g = local2global(src)
        for dst in range(n):
            dst_g = local2global(dst)
            if is_received_locally[dst]:
                continue
            if old_l[src_g] == -1 or new_l[dst_g] == -1:
                continue
            if old_l[src_g] == new_l[dst_g]:
                is_received_locally[dst] = True
    experts_send_loc = {}
    for src in range(n):
        e = old_l[local2global(src)]
        if e == -1 or e in experts_send_loc:
            continue
        experts_send_loc[e] = src
    experts_recv_loc = {}
    for dst in range(n):
        if is_received_locally[dst]:
            continue
        e = new_l[local2global(dst)]
        if e == -1 or e in experts_recv_loc:
            continue
        experts_recv_loc[e] = dst
    return is_unchanged, is_received_locally, experts_send_loc, experts_recv_loc


# PR: numpy vectorized body copied verbatim from merge_sha 6170d47d2220.
def _mtb_gold(old_indices, new_indices, ep_rank):
    num_local_experts = NUM_LOCAL
    recv_primary_mask = np.zeros((num_local_experts,), dtype=np.bool_)
    send_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    send_src_rows = np.full((num_local_experts,), -1, dtype=np.int32)
    recv_expert_ids = np.full((num_local_experts,), -1, dtype=np.int64)
    recv_dst_rows = np.full((num_local_experts,), -1, dtype=np.int32)

    base = ep_rank * num_local_experts
    local_rows = np.arange(num_local_experts, dtype=np.int32)
    local_global = base + local_rows
    old_local = old_indices[local_global]
    new_local = new_indices[local_global]

    is_unchanged = old_local == new_local
    new_valid = new_local != -1
    can_recv_local = np.isin(new_local, old_local, assume_unique=False)
    is_received_locally = np.logical_or(
        is_unchanged, np.logical_and(new_valid, can_recv_local))

    send_count = 0
    valid_old = old_local != -1
    if np.any(valid_old):
        uniq, first_idx = np.unique(old_local[valid_old], return_index=True)
        filtered = local_rows[valid_old]
        src_rows = filtered[first_idx]
        send_count = int(uniq.shape[0])
        send_expert_ids[:send_count] = uniq
        send_src_rows[:send_count] = src_rows

    recv_count = 0
    need_recv = np.logical_and(~is_received_locally, new_valid)
    if np.any(need_recv):
        desired = new_local[need_recv]
        desired_dsts = local_rows[need_recv]
        uniq_r, uniq_i = np.unique(desired, return_index=True)
        dst_rows = desired_dsts[uniq_i]
        recv_count = int(uniq_r.shape[0])
        recv_expert_ids[:recv_count] = uniq_r
        recv_dst_rows[:recv_count] = dst_rows
        recv_primary_mask[dst_rows] = True
    return (is_unchanged, is_received_locally, send_expert_ids, send_src_rows,
            recv_expert_ids, recv_dst_rows, recv_primary_mask,
            send_count, recv_count)


def bench_move_to_buffer(old, new):
    ep_rank = NUM_RANKS // 2
    if GOLD:
        def run():
            _mtb_gold(old, new, ep_rank)
    else:
        old_l, new_l = old.tolist(), new.tolist()

        def run():
            _mtb_base(old_l, new_l, ep_rank)
    return run


# ---------------- move_from_buffer mapping body (INLINE FROM DIFF) -------
# BASELINE: python loop resolving each dst's source row via experts_recv_loc.
def _mfb_base(is_unchanged, is_recv_local, experts_recv_loc, new_l, ep_rank):
    n = len(is_unchanged)
    local2global = lambda i: ep_rank * n + i
    plan = []  # (dst, src) copy plan -- the computed mapping
    for dst in range(n):
        if is_unchanged[dst]:
            continue
        if is_recv_local[dst]:
            plan.append((dst, dst))
        else:
            e = new_l[local2global(dst)]
            if e == -1:
                continue
            src = experts_recv_loc[e]
            plan.append((dst, src))
    return plan


# PR: numpy duplicate-resolution body copied verbatim from 6170d47d2220.
def _mfb_gold(is_unchanged, is_received_locally, recv_primary_mask,
              recv_count, recv_expert_ids, recv_dst_rows, new_indices, ep_rank):
    num_local_experts = is_unchanged.shape[0]
    copy_mask = np.logical_or(is_received_locally, recv_primary_mask)
    dest_mask = np.logical_and(~is_unchanged, copy_mask)
    _ = np.nonzero(dest_mask)[0].tolist()  # primary copy plan
    if recv_count == 0:
        return
    base = ep_rank * num_local_experts
    local_experts = new_indices[base + np.arange(num_local_experts, dtype=np.int32)]
    duplicate_mask = np.logical_and(
        np.logical_and(~is_unchanged, ~is_received_locally),
        np.logical_and(~recv_primary_mask, local_experts != -1))
    if not bool(duplicate_mask.any()):
        return
    dup_dst_rows = np.nonzero(duplicate_mask)[0]
    dup_experts = local_experts[dup_dst_rows]
    prim_experts = recv_expert_ids[:recv_count]
    prim_dsts = recv_dst_rows[:recv_count]
    order = np.argsort(prim_experts, kind="stable")
    prim_experts_sorted = prim_experts[order]
    prim_dsts_sorted = prim_dsts[order]
    pos = np.searchsorted(prim_experts_sorted, dup_experts)
    valid = np.logical_and(
        pos < prim_experts_sorted.shape[0],
        prim_experts_sorted[np.minimum(pos, prim_experts_sorted.shape[0] - 1)]
        == dup_experts)
    if not bool(valid.any()):
        return
    _ = (dup_dst_rows[valid], prim_dsts_sorted[pos[valid]])  # dup copy plan


def bench_move_from_buffer(old, new):
    ep_rank = NUM_RANKS // 2
    if GOLD:
        (is_unchanged, is_recv_local, _se, _sr, recv_expert_ids,
         recv_dst_rows, recv_primary_mask, _sc, recv_count) = _mtb_gold(
            old, new, ep_rank)

        def run():
            _mfb_gold(is_unchanged, is_recv_local, recv_primary_mask,
                      recv_count, recv_expert_ids, recv_dst_rows, new, ep_rank)
    else:
        old_l, new_l = old.tolist(), new.tolist()
        is_unchanged, is_recv_local, _esl, experts_recv_loc = _mtb_base(
            old_l, new_l, ep_rank)

        def run():
            _mfb_base(is_unchanged, is_recv_local, experts_recv_loc, new_l,
                      ep_rank)
    return run


# ---------------- rearrange_e2e: the REAL per-layer path, P2P stubbed -----
class _FakeReq:
    def wait(self):
        pass


class _FakeGroup:
    """stands in for the EP ProcessGroup: rank() / size() only"""
    def __init__(self, rank, size):
        self._rank, self._size = rank, size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def _stub_p2p():
    # The module binds these names at import time, so patching the module
    # attributes is enough. P2POp objects are built per send/recv exactly as
    # in production (that Python cost is real); only the transport is a no-op.
    R.P2POp = lambda op, tensor, peer: (op, tensor, peer)
    R.batch_isend_irecv = lambda ops: [_FakeReq()]
    R.get_global_rank = lambda group, rank: rank


def make_e2e():
    old_np_all, new_np_all = _make_indices(4242, NUM_LAYERS)
    old_t, new_t = torch.from_numpy(old_np_all), torch.from_numpy(new_np_all)
    weights = [torch.zeros(NUM_LOCAL, HIDDEN)]
    buffers = [torch.empty_like(w) for w in weights]
    groups = [_FakeGroup(r, NUM_RANKS) for r in range(NUM_RANKS)]

    if GOLD:
        old_np = old_t.cpu().numpy()
        new_np = new_t.cpu().numpy()

        def run():
            for layer in range(NUM_LAYERS):
                for g in groups:
                    is_unch, is_loc, meta = R.move_to_buffer(
                        num_local_experts=NUM_LOCAL,
                        old_indices=old_np[layer], new_indices=new_np[layer],
                        expert_weights=weights, expert_weights_buffers=buffers,
                        cuda_stream=None, ep_group=g)
                    R.move_from_buffer(
                        expert_weights=weights, expert_weights_buffers=buffers,
                        is_unchanged=is_unch, is_received_locally=is_loc,
                        recv_metadata=meta, new_indices=new_np[layer],
                        ep_rank=g.rank())
    else:
        old_cpu, new_cpu = old_t.cpu(), new_t.cpu()

        def run():
            for layer in range(NUM_LAYERS):
                for g in groups:
                    is_unch, is_loc, recv_loc = R.move_to_buffer(
                        num_local_experts=NUM_LOCAL,
                        old_indices=old_cpu[layer].tolist(),
                        new_indices=new_cpu[layer].tolist(),
                        expert_weights=weights, expert_weights_buffer=buffers,
                        cuda_stream=None, ep_group=g)
                    R.move_from_buffer(
                        expert_weights=weights, expert_weights_buffer=buffers,
                        is_unchanged=is_unch, is_received_locally=is_loc,
                        experts_recv_loc=recv_loc,
                        new_indices=new_t[layer].tolist(), ep_group=g)
    return run


def main():
    _stub_p2p()
    results = {"get_ep_ranks_with_expert": [],
               "move_to_buffer": [],
               "move_from_buffer": []}
    for rep in range(REPEATS):
        old, new = _make_indices(1000 + rep)
        old, new = old[0], new[0]
        results["get_ep_ranks_with_expert"].append(
            _best_ms(bench_get_ep_ranks(old, new)))
        results["move_to_buffer"].append(
            _best_ms(bench_move_to_buffer(old, new)))
        results["move_from_buffer"].append(
            _best_ms(bench_move_from_buffer(old, new)))

    e2e = make_e2e()
    e2e()                                   # warm-up pass (untimed)
    e2e_ms = []
    for _ in range(E2E_REPEATS):
        t0 = time.perf_counter()
        e2e()
        e2e_ms.append((time.perf_counter() - t0) * 1e3)

    print(f"rearrange_e2e {statistics.median(e2e_ms):.3f}")
    for name, vals in results.items():
        print(f"{name} {statistics.median(vals):.6f}")
    print(f"# arm={'pr' if GOLD else 'baseline'} ranks={NUM_RANKS} local={NUM_LOCAL} "
          f"phys={NUM_PHYS} layers={NUM_LAYERS} e2e_ms={[round(x, 1) for x in e2e_ms]} "
          f"per_rank_ms={statistics.median(e2e_ms) / NUM_RANKS:.1f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
