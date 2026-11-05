#!/usr/bin/env python3
"""Benchmark EPLB primitives for vLLM PR #29499 (Optimize EPLB with numpy).

Times the three primitives the PR rewrote in numpy, against whatever vllm tree
is on PYTHONPATH. The arm (base vs gold) is auto-detected from the imported
module's API.

Workload: 32 ranks x 8 local experts = 256 physical expert slots, heavy
replication (8 distinct logical experts), representative of a MoE EPLB
rebalance. 5 repeats, best-of-INNER inner iterations each.

Approaches (stated per-primitive in the result "note"):

  * get_ep_ranks_with_expert  -> ISOLATED IMPORT. Pure computation (no dist /
    no P2P), so the real function is imported from the tree. base loops
    get_ep_ranks_with_expert over every unique expert; gold makes one
    get_ep_ranks_with_experts_batch call. This is the PR's headline rewrite.

  * move_to_buffer / move_from_buffer -> INLINE FROM DIFF. These functions
    require a live multi-rank process group + GPU P2P that cannot run in a
    single CPU container. We therefore reproduce the EXACT index-mapping body
    the PR changed (the only arm-difference; the P2P sends and tensor .copy_
    calls are byte-identical across arms and excluded). Code copied verbatim
    from rebalance_execute.py at base_sha 0ada960a2076 and merge_sha
    6170d47d2220.
"""

import sys
import time

import numpy as np

from vllm.distributed.eplb import rebalance_execute as R

# Large-scale MoE (e.g. DeepSeek-V3 class): 128 ranks x 16 local slots =
# 2048 physical expert slots, 256 distinct logical experts. This is the
# regime the PR targets -- the base python loops scale with NUM_PHYS per
# expert, so numpy vectorization pays off here.
NUM_RANKS = 128
NUM_LOCAL = 16
NUM_PHYS = NUM_RANKS * NUM_LOCAL        # 2048
NUM_LOGICAL = 256
INNER = 100
REPEATS = 5

GOLD = hasattr(R, "get_ep_ranks_with_experts_batch")


def _make_indices(seed):
    rng = np.random.default_rng(seed)
    old = rng.integers(0, NUM_LOGICAL, size=NUM_PHYS).astype(np.int64)
    new = rng.permutation(old.copy()).astype(np.int64)
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
# BASE: pure-python per-row double loop + per-expert get_ep_ranks_with_expert.
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
    # send-loc map
    experts_send_loc = {}
    for src in range(n):
        e = old_l[local2global(src)]
        if e == -1 or e in experts_send_loc:
            continue
        experts_send_loc[e] = src
    # recv-loc map
    experts_recv_loc = {}
    for dst in range(n):
        if is_received_locally[dst]:
            continue
        e = new_l[local2global(dst)]
        if e == -1 or e in experts_recv_loc:
            continue
        experts_recv_loc[e] = dst
    return is_unchanged, is_received_locally, experts_send_loc, experts_recv_loc


# GOLD: numpy vectorized body copied verbatim from merge_sha 6170d47d2220.
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
# BASE: python loop resolving each dst's source row via experts_recv_loc.
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


# GOLD: numpy duplicate-resolution body copied verbatim from 6170d47d2220.
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


def main():
    results = {"get_ep_ranks_with_expert": [],
               "move_to_buffer": [],
               "move_from_buffer": []}
    for rep in range(REPEATS):
        old, new = _make_indices(1000 + rep)
        results["get_ep_ranks_with_expert"].append(
            _best_ms(bench_get_ep_ranks(old, new)))
        results["move_to_buffer"].append(
            _best_ms(bench_move_to_buffer(old, new)))
        results["move_from_buffer"].append(
            _best_ms(bench_move_from_buffer(old, new)))
    for name, vals in results.items():
        for v in vals:
            print(f"{name} {v:.6f}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
