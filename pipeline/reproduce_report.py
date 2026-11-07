#!/usr/bin/env python3
"""Reproduce report: reported gain (from the PR) vs gain measured here (baseline vs PR's solution).

    python3 pipeline/reproduce_report.py 21222 21281 ...
Reads data/pr_<n>/results.<hw-slug>.json (this machine's slug, else the first slug available — named in
the row); prints one row per PR. Speedup = baseline / PR-solution per scenario (lower latency = better;
for throughput units PR-solution / baseline). Scenarios where either arm is missing are skipped.
"""
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402


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


def main():
    rows = []
    for pr in sys.argv[1:]:
        p = layout.results_json_any(int(pr))
        if not p:
            rows.append((pr, "-", "no result", ""))
            continue
        d = json.load(open(p))
        sp = speedups(d)
        tests = "/".join(str((d.get("arms") or {}).get(k, {}).get("tests", "?")) for k in ("base", "gold"))
        meas = f"median {statistics.median(sp):.2f}x  max {max(sp):.2f}x  n={len(sp)}" if sp else "no baseline + PR-solution timing"
        v = (d.get("summary") or {}).get("verdict", "")
        hw = d.get("hw_slug") or os.path.basename(p).split(".")[1]
        rows.append((pr, (d.get("reported_gain") or "-")[:40], f"{meas}  tests {tests}  {v}  [{hw}]", (d.get("title") or "")[:50]))
    print(f"{'PR':>6} | {'reported':40} | measured here (baseline -> PR's solution) [hardware] | title")
    for pr, rep, meas, title in rows:
        print(f"{pr:>6} | {rep:40} | {meas} | {title}")


if __name__ == "__main__":
    main()
