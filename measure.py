#!/usr/bin/env python3
"""Paired A-B-B-A timing + log-speedup + MAD noise gate.

Reusable library (and CLI) ported from FormulaCode DataSmith measurement/
noise.py concepts, stripped to CPU + no-Docker + no-ASV:

- paired A-B-B-A interleaving in one process to cancel drift
- natural-log speedup  s = ln(T_base / T_patch)  (additive, symmetric noise)
- robust SD via MAD:   sd = 1.4826 * median|x - median(x)|
- nop (base-vs-base) baseline to learn host spread s_repo
- pooled sigma:        sigma^2 = (n_b*s_b^2 + n0*s_repo^2)/(n_b+n0)
- threshold:           theta = clip(k*sigma, theta_min, theta_max)
- a gain "counts" only if >= ORACLE_MIN_RUNS paired samples clear +theta

No torch, no GPU, no Docker. A task supplies two zero-arg callables (base_fn,
patch_fn) that do the SAME work via different implementations; we time them.
"""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# ---- constants (FormulaCode defaults, tunable per host) --------------------
ORACLE_MIN_RUNS = 2       # paired samples that must clear +theta to "count"
ORACLE_MAX_BROKEN = 0.5   # fraction of benchmarks allowed to regress
K_SIGMA = 3.0             # theta = clip(k*sigma, ...)
THETA_MIN = 0.01          # ~1% log-speedup floor
THETA_MAX = 0.50          # cap so a huge sigma can't swallow real gains
DEFAULT_PAIRS = 7         # A-B-B-A pairs per comparison
DEFAULT_NOP = 7           # base-vs-base pairs for host spread


def robust_sd(xs: list[float], center: Optional[float] = None) -> float:
    """1.4826 * median|x - center|  (MAD -> normal-consistent SD)."""
    if len(xs) < 2:
        return 0.0
    c = statistics.median(xs) if center is None else center
    mad = statistics.median([abs(x - c) for x in xs])
    return 1.4826 * mad


def _time_once(fn: Callable[[], None], inner: int) -> float:
    """Seconds for `inner` calls of fn (one timed block)."""
    t0 = time.perf_counter()
    for _ in range(inner):
        fn()
    return time.perf_counter() - t0


def paired_logspeedups(base_fn: Callable[[], None],
                       patch_fn: Callable[[], None],
                       pairs: int = DEFAULT_PAIRS,
                       inner: int = 1000,
                       warmup: int = 2) -> list[float]:
    """A-B-B-A interleaving. Returns per-pair ln(T_base / T_patch)."""
    for _ in range(warmup):
        _time_once(base_fn, inner)
        _time_once(patch_fn, inner)
    out: list[float] = []
    for _ in range(pairs):
        a1 = _time_once(base_fn, inner)
        b1 = _time_once(patch_fn, inner)
        b2 = _time_once(patch_fn, inner)
        a2 = _time_once(base_fn, inner)
        tb = (a1 + a2) / 2.0
        tp = (b1 + b2) / 2.0
        if tb > 0 and tp > 0:
            out.append(math.log(tb / tp))
    return out


@dataclass
class Comparison:
    """Result of base-vs-patch with noise gate."""
    name: str
    samples: list[float]          # per-pair log-speedups
    s_repo: float                 # host spread from nop baseline
    median_log: float = field(init=False)
    median_pct: float = field(init=False)
    sigma: float = field(init=False)
    theta: float = field(init=False)
    counts: bool = field(init=False)     # cleared gate as improvement
    broken: bool = field(init=False)     # regressed past -theta

    def __post_init__(self) -> None:
        n_b = len(self.samples)
        self.median_log = statistics.median(self.samples) if self.samples else 0.0
        self.median_pct = (math.exp(self.median_log) - 1.0) * 100.0
        s_b = robust_sd(self.samples, center=0.0)
        n0 = max(n_b, 1)
        sigma2 = (n_b * s_b * s_b + n0 * self.s_repo * self.s_repo) / (n_b + n0)
        self.sigma = math.sqrt(sigma2)
        self.theta = min(max(K_SIGMA * self.sigma, THETA_MIN), THETA_MAX)
        up = sum(1 for s in self.samples if s >= self.theta)
        dn = sum(1 for s in self.samples if s <= -self.theta)
        self.counts = up >= ORACLE_MIN_RUNS
        self.broken = dn >= ORACLE_MIN_RUNS


def compare(name: str,
            base_fn: Callable[[], None],
            patch_fn: Callable[[], None],
            pairs: int = DEFAULT_PAIRS,
            inner: int = 1000,
            nop: int = DEFAULT_NOP) -> Comparison:
    """Measure base-vs-patch; learn host spread from base-vs-base nop runs."""
    nop_samples = paired_logspeedups(base_fn, base_fn, pairs=nop, inner=inner)
    s_repo = robust_sd(nop_samples, center=0.0)
    samples = paired_logspeedups(base_fn, patch_fn, pairs=pairs, inner=inner)
    return Comparison(name=name, samples=samples, s_repo=s_repo)


def advantage(t_base: float, t_agent: float, t_expert: float) -> float:
    """idea-A metric: min(1, (T_base - T_agent)/(T_base - T_expert)).

    Expressed on raw times. Correctness gate (I(tests pass)) applied by caller.
    """
    denom = t_base - t_expert
    if denom <= 0:
        return 0.0
    return min(1.0, (t_base - t_agent) / denom)


if __name__ == "__main__":
    # self-test: patch that is ~2x faster should "count"; identical should not.
    def slow():
        s = 0
        for i in range(200):
            s += i * i
    def fast():
        s = 0
        for i in range(100):
            s += i * i
    c = compare("selftest_2x", slow, fast, pairs=5, inner=200)
    print(f"median_log={c.median_log:.3f} ({c.median_pct:+.1f}%) "
          f"theta={c.theta:.3f} s_repo={c.s_repo:.3f} "
          f"counts={c.counts} broken={c.broken}")
    c2 = compare("selftest_nop", slow, slow, pairs=5, inner=200)
    print(f"median_log={c2.median_log:.3f} ({c2.median_pct:+.1f}%) "
          f"theta={c2.theta:.3f} counts={c2.counts} broken={c2.broken}")
