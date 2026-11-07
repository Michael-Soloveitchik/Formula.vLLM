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
    out = []
    for sc in (d.get("latency") or {}).get("scenarios", {}).values():
        b, g = (sc.get("base") or {}).get("mean"), (sc.get("gold") or {}).get("mean")
        if not b or not g:
            continue
        thr = "/s" in (sc.get("unit") or "")
        out.append(g / b if thr else b / g)
    return out

rows = []
for pr in sys.argv[1:]:
    p = os.path.join(HERE, "results", f"{pr}.json")
    if not os.path.exists(p):
        rows.append((pr, "-", "no result", "-", "-")); continue
    d = json.load(open(p))
    sp = speedups(d)
    t = d.get("logs", {})
    tests = f"baseline={'FAIL' if 'FAIL' in str(d.get('arms_tests', '')) else '?'}"
    if sp:
        meas = f"median {statistics.median(sp):.2f}x  max {max(sp):.2f}x  n={len(sp)}"
    else:
        meas = "no baseline + PR-solution timing"
    rows.append((pr, (d.get("reported_gain") or "-")[:40], meas, d.get("title", "")[:50]))
print(f"{'PR':>6} | {'reported':40} | measured here (baseline -> PR's solution) | title")
for r in rows:
    pr, rep, meas = r[0], r[1], r[2]
    title = r[3] if len(r) > 3 else ""
    print(f"{pr:>6} | {rep:40} | {meas} | {title}")
