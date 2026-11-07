#!/usr/bin/env python3
"""Microbench for vLLM PR #39328: [Core] Cache InductorPass.hash_source with functools.cache.

InductorPass.hash_source(*srcs) hashes the source text of functions / classes / instances
(inspect.getsource + sha256) to build pass UUIDs; it is called repeatedly for the same objects
during compilation. The PR routes it through a functools.cache'd module helper keyed on the
resolved types, so after the first miss each lookup is a dict hit.
Author: cache lookup 32.50 ms -> 11.30 ms (10000 lookups), cold compile 34.10 s -> 29.70 s
(Llama-3-70B TP4; not reproducible without the model, kept reported-only).

The bench calls the arm's real `InductorPass.hash_source` 10000 times cycling over five source
objects (vLLM's own pass classes/functions plus a local class instance), after one untimed call
per object. Reports the total ms for the 10000 lookups (the author's scale), median of REPEATS.
CPU only. Runs inside the task image: python3 /task/bench.py
"""
import time

from vllm.compilation.passes import inductor_pass as ip

N_LOOKUPS = 10000
REPEATS = 5


def _local_fn(x):
    return x * 2 + 1


class _LocalPass:
    def __call__(self, graph):
        return graph


def objects():
    objs = [ip.InductorPass, ip.InductorPass.hash_dict, _local_fn, _LocalPass(), ip]
    # a module is not a valid src; use the pass_context function instead if present
    objs[-1] = getattr(ip, "pass_context", _local_fn)
    return objs


def main():
    objs = objects()
    hash_source = ip.InductorPass.hash_source
    digests = [hash_source(o) for o in objs]            # untimed first call per object
    assert len(set(digests)) == len(objs) and all(len(d) == 64 for d in digests)
    totals = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        for i in range(N_LOOKUPS):
            hash_source(objs[i % len(objs)])
        totals.append((time.perf_counter() - t0) * 1e3)
    totals.sort()
    print(f"cache_lookup {totals[len(totals) // 2]:.4f}")
    print(f"# total ms for {N_LOOKUPS} hash_source lookups over {len(objs)} objects; "
          f"runs_ms={[round(t, 2) for t in totals]}")


if __name__ == "__main__":
    main()
