#!/usr/bin/env python3
"""Reproduce report: reported gain (from the PR) vs gain measured here (baseline vs PR's solution).

    python3 reproduce_report.py 21222 21281 ...
Reads results/<pr>.json only; prints one row per PR. Speedup = baseline / PR-solution per
scenario (lower latency = better; for throughput units PR-solution / baseline). Scenarios
where either arm is missing are skipped.
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))

def speedups(d):
    """baseline/PR-solution ratio per experiment (>1 = PR faster). Throughput units (/s) inverted.
    baseline falls back to the script's PyTorch column (pytorch_ref) when the baseline tree cannot run the bench."""
    out = []
    pyref = (d.get("latency") or {}).get("pytorch_ref") or {}
    exps = d.get("experiments") or [{"name": k, "unit": v.get("unit"), "arms": v} for k, v in ((d.get("latency") or {}).get("scenarios") or {}).items()]
    for e in exps:
        a = e.get("arms") or {}
        b = (a.get("base") or {}).get("mean")
        if b is None and e["name"] in pyref:
            b = pyref[e["name"]].get("mean")
        g = (a.get("gold") or {}).get("mean")
        if not b or not g:
            continue
        thr = "/s" in (e.get("unit") or "")
        out.append(g / b if thr else b / g)
    return out


rows = []
for pr in sys.argv[1:]:
    p = os.path.join(HERE, "results", f"{pr}.json")
    if not os.path.exists(p):
        rows.append((pr, "-", "no result", "-", "-")); continue
    d = json.load(open(p))
    sp = speedups(d)
    tests = "/".join(str((d.get("arms") or {}).get(k, {}).get("tests", "?")) for k in ("base", "gold"))
    if sp:
        meas = f"median {statistics.median(sp):.2f}x  max {max(sp):.2f}x  n={len(sp)}"
    else:
        meas = "no baseline + PR-solution timing"
    rows.append((pr, (d.get("reported_gain") or "-")[:40], meas + f"  tests {tests}", d.get("title", "")[:50]))
print(f"{'PR':>6} | {'reported':40} | measured here (baseline -> PR's solution) | title")
for r in rows:
    pr, rep, meas = r[0], r[1], r[2]
    title = r[3] if len(r) > 3 else ""
    print(f"{pr:>6} | {rep:40} | {meas} | {title}")
