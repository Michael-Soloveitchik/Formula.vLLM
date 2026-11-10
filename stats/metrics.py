#!/usr/bin/env python3
"""Metric functions over a single PR's headline-derived scores.

Each function takes a `pr` dict and returns a float (or None when the inputs are
absent). `pr` is the `summary.headline_derived` block of data/pr_<n>/results.<hw>.json:

    {
      "unit": "ms", "higher_is_better": false,
      "reported":   {"baseline": .., "pr_solution": .., "ratio": .., "pct": .., "marker": ""},
      "reproduced": {"baseline": .., "pr_solution": .., "ratio": .., "ratio_sd": ..},
      "reliability": {"flag": "ok"|"below_host_floor"|..},
    }

gap   = PR's solution − baseline, relative to its own baseline so units cancel:
        throughput units  (higher is better): (ratio − 1)
        latency   units  (lower  is better): (1 − 1/ratio)
gap ratio     = reproduced gap ÷ reported gap   (1.0 = we reproduce the author's gap exactly)
baseline ratio = author's baseline ÷ our baseline (only when the author gave an absolute baseline)

METRICS maps metric name -> function, so a caller can do:
    from stats import metrics
    metrics.compute(pr, "gap_ratio")        # one metric
    metrics.compute_all(pr)                 # {name: value} for every metric
"""
import math
import re

_UP_RE = re.compile(r"/s$|tok/s|req/s")


def is_higher_better(pr):
    """True when a bigger number is a better result (throughput units)."""
    if pr.get("higher_is_better") is not None:
        return bool(pr["higher_is_better"])
    return bool(_UP_RE.search(pr.get("unit") or ""))


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _gap_from_ratio(ratio, up):
    """gap relative to baseline from a speedup ratio (>0). None if unusable."""
    if not _finite(ratio) or ratio <= 0:
        return None
    return ratio - 1 if up else 1 - 1 / ratio


def reported_gap(pr):
    r = pr.get("reported") or {}
    return _gap_from_ratio(r.get("ratio"), is_higher_better(pr))


def reproduced_gap(pr):
    r = pr.get("reproduced") or {}
    return _gap_from_ratio(r.get("ratio"), is_higher_better(pr))


def gap_ratio(pr):
    """reproduced gap ÷ reported gap. 1.0 = exact reproduction of author's gap."""
    rg, mg = reported_gap(pr), reproduced_gap(pr)
    if rg is None or mg is None or rg == 0:
        return None
    return mg / rg


def distance_from_1(pr):
    """|gap_ratio − 1| — how far the reproduction is from the author's gap. Lower = better."""
    g = gap_ratio(pr)
    return None if g is None else abs(g - 1)


def baseline_ratio(pr):
    """author's reported baseline ÷ our reproduced baseline. Only when the author
    gave an absolute baseline (marker == '' and baseline present)."""
    rr, rp = pr.get("reported") or {}, pr.get("reproduced") or {}
    rb, mb = rr.get("baseline"), rp.get("baseline")
    if rr.get("marker", "") != "" or not _finite(rb) or not _finite(mb) or mb == 0:
        return None
    return rb / mb


def reproduced_ratio(pr):
    return (pr.get("reproduced") or {}).get("ratio")


def reported_ratio(pr):
    return (pr.get("reported") or {}).get("ratio")


def reliability(pr):
    return (pr.get("reliability") or {}).get("flag") or "ok"


METRICS = {
    "gap_ratio": gap_ratio,
    "distance_from_1": distance_from_1,
    "baseline_ratio": baseline_ratio,
    "reported_gap": reported_gap,
    "reproduced_gap": reproduced_gap,
    "reported_ratio": reported_ratio,
    "reproduced_ratio": reproduced_ratio,
}


def compute(pr, name):
    """Single metric by name. Raises KeyError on an unknown metric."""
    return METRICS[name](pr)


def compute_all(pr):
    """{metric name: value} for every metric, skipping None."""
    out = {}
    for name, fn in METRICS.items():
        v = fn(pr)
        if v is not None:
            out[name] = v
    return out
