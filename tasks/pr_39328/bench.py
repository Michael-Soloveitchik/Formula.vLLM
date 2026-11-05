#!/usr/bin/env python3
"""Microbenchmark for vLLM PR #39328.

"[Core] Cache InductorPass.hash_source with functools.cache"

The PR wraps the inspect.getsource(...) + sha256 hashing done by
InductorPass.hash_source in functools.cache, so repeated pass-uuid /
hash_source lookups over the same (hashable) source objects become O(1)
after the first miss.

This times the exact hot path BOTH ways in one process:

  base : recompute inspect.getsource(obj) + sha256 on every call
         (the pre-PR hash_source body)
  gold : the post-PR path -> resolve instances to class, delegate to a
         functools.cache-wrapped helper that does getsource+sha256 once.

We emulate "repeated pass-uuid lookups" by calling over a handful of
distinct source objects many times (10000 lookups each repeat), which is
exactly the cache_lookup experiment from the PR. We report per-call ms
to match the PR's reported ms scale, 5 repeats.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import time
import types

# ---------------------------------------------------------------------------
# A handful of distinct source objects to hash (functions + classes +
# instances), mirroring the kind of args hash_source receives for pass uuids.
# ---------------------------------------------------------------------------


def _src_fn_a(x):
    return x + 1


def _src_fn_b(x, y):
    return x * y - 1


class _SrcClassA:
    def run(self):
        return 42


class _SrcClassB:
    def run(self):
        return "hello world"


class _SrcClassC:
    def method(self, a, b):
        return (a, b)


# Mix of function objects, type objects, and instances (resolved to class).
_OBJECTS: list = [
    _src_fn_a,
    _src_fn_b,
    _SrcClassA,
    _SrcClassB(),
    _SrcClassC(),
]


# ---------------------------------------------------------------------------
# base arm: pre-PR hash_source (recomputes getsource + sha256 every call)
# ---------------------------------------------------------------------------
def hash_source_base(*srcs) -> str:
    hasher = hashlib.sha256()
    for src in srcs:
        if isinstance(src, str):
            src_str = src
        elif isinstance(src, (types.FunctionType, type)):
            src_str = inspect.getsource(src)
        else:
            # object instance
            src_str = inspect.getsource(src.__class__)
        hasher.update(src_str.encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# gold arm: post-PR hash_source (functools.cache over resolved types)
# ---------------------------------------------------------------------------
@functools.cache
def _hash_source_cached(*srcs) -> str:
    hasher = hashlib.sha256()
    for src in srcs:
        src_str = src if isinstance(src, str) else inspect.getsource(src)
        hasher.update(src_str.encode("utf-8"))
    return hasher.hexdigest()


def hash_source_gold(*srcs) -> str:
    cache_key = tuple(
        src if isinstance(src, (str, type, types.FunctionType)) else src.__class__
        for src in srcs
    )
    return _hash_source_cached(*cache_key)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------
N_LOOKUPS = 10000  # lookups per repeat (cycled over the objects)
REPEATS = 5


def time_arm(fn, objects, n_lookups) -> float:
    """Return per-call time in ms for n_lookups calls, one object per call."""
    m = len(objects)
    t0 = time.perf_counter()
    for i in range(n_lookups):
        fn(objects[i % m])
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1e3
    return total_ms / n_lookups  # per-call ms


def main() -> None:
    # Correctness: both arms must produce identical hashes for each object.
    for obj in _OBJECTS:
        b = hash_source_base(obj)
        g = hash_source_gold(obj)
        assert b == g, f"hash mismatch for {obj!r}: {b} != {g}"

    for _ in range(REPEATS):
        # Fresh cache each repeat so gold pays one cold miss per object then
        # hits -- this is the realistic "repeated lookup" profile.
        _hash_source_cached.cache_clear()

        base_ms = time_arm(hash_source_base, _OBJECTS, N_LOOKUPS)
        gold_ms = time_arm(hash_source_gold, _OBJECTS, N_LOOKUPS)

        print(f"cache_lookup base {base_ms:.6f}")
        print(f"cache_lookup gold {gold_ms:.6f}")


if __name__ == "__main__":
    main()
