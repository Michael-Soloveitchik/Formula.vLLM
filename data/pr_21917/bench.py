#!/usr/bin/env python3
"""Microbench for vLLM PR #21917: [Core] Optimize scheduler request removal for single completions.

base: Scheduler.update_from_output drops finished requests with
          self.running = [req for req in self.running if req not in stopped_running_reqs]
      i.e. an O(len(running)) rebuild + set lookup per element, even when ONE request stopped.
gold: `remove_all(self.running, stopped)` (vllm/v1/core/sched/utils.py): for a single item
      `list.remove()` in place (one linear scan, no new list); comprehension only for >1 items.

The bench imports `remove_all` from the arm's tree; the base tree does not have it, so there
the exact pre-PR expression from scheduler.py is used (same inputs, same semantics). Workload:
a running queue of RUNNING_LEN request objects (identity hash like Request), remove one request
at a time (the common "single completion" case) - the item is at the front / middle / back of
the queue in rotation, and the queue is refilled so its length stays constant - plus the >1 case
(4 stopped requests) that the PR leaves on the comprehension path. CPU only.
us per removal (mean over ITERS); the author reports "2.6-3.4x speedup for single request
removal, 64% improvement in request removal operations" without absolute numbers.
Runs inside the task image: python3 /task/bench.py
"""
import time

try:
    from vllm.v1.core.sched.utils import remove_all
    ARM = "gold"
except ImportError:
    ARM = "base"

    def remove_all(lst, items_to_remove):           # pre-PR scheduler.py expression
        return [req for req in lst if req not in items_to_remove]

RUNNING_LEN = 256     # --max-num-seqs
WARMUP = 2000
ITERS = 100000


class Req:
    __slots__ = ("request_id",)

    def __init__(self, i):
        self.request_id = f"req-{i}"


def bench_single():
    running = [Req(i) for i in range(RUNNING_LEN)]
    positions = [0, RUNNING_LEN // 2, RUNNING_LEN - 1]
    counter = RUNNING_LEN

    def step(k):
        nonlocal running, counter
        victim = running[positions[k % 3]]
        running = remove_all(running, {victim})
        running.append(Req(counter))
        counter += 1

    for k in range(WARMUP):
        step(k)
    t0 = time.perf_counter()
    for k in range(ITERS):
        step(k)
    return (time.perf_counter() - t0) * 1e6 / ITERS, len(running)


def bench_multi(n_stop=4):
    running = [Req(i) for i in range(RUNNING_LEN)]
    counter = RUNNING_LEN

    def step(k):
        nonlocal running, counter
        stride = RUNNING_LEN // n_stop
        victims = {running[(k + j * stride) % RUNNING_LEN] for j in range(n_stop)}
        running = remove_all(running, victims)
        running.extend(Req(counter + j) for j in range(n_stop))
        counter += n_stop

    for k in range(WARMUP):
        step(k)
    t0 = time.perf_counter()
    for k in range(ITERS // 4):
        step(k)
    return (time.perf_counter() - t0) * 1e6 / (ITERS // 4), len(running)


def main():
    us1, n1 = bench_single()
    assert n1 == RUNNING_LEN
    print(f"remove_single_request {us1:.4f}")
    us4, n4 = bench_multi()
    assert n4 == RUNNING_LEN
    print(f"remove_four_requests {us4:.4f}")
    print(f"# arm={ARM} running_len={RUNNING_LEN} iters={ITERS}")


if __name__ == "__main__":
    main()
