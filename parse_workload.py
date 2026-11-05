#!/usr/bin/env python3
"""Fold per-arm workload logs into one latency dict (ms per scenario, mean ± sd).

Usage:
    parse_workload.py <work_dir> <arm1> [arm2 ...]

Reads <work_dir>/wl_<arm>_<i>.log for every repeat i, parses the SUMMARY TABLE
printed by benchmarks/benchmark_topk_topp.py (one row per scenario:
`<scenario> <batch> <vocab> <ops>% <triton_ms> <pytorch_ms> <speedup>x ...`),
and prints JSON:

    {"unit": "ms",
     "scenarios": {"topk_whole_b2048_v32768": {"<arm>": {"mean":..,"sd":..,"n":..}, ...}},
     "pytorch_ref": {"topk_whole_b2048_v32768": {"mean":..,"sd":..,"n":..}}}

`pytorch_ref` is the script's own sort-based PyTorch column (same process, same
logits) taken from the gold arm's runs; it is the pre-PR number for PRs whose
base tree cannot run the script at all (e.g. 33538 introduces the kernel).
"""
import glob
import json
import math
import os
import re
import sys

ROW = re.compile(
    r"^(\w+?)\s+(\d+)\s+(\d+)\s+(\d+)%\s+([\d.]+|inf)\s+([\d.]+|inf)\s+([\d.]+|inf)x"
)


def stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return {"mean": round(m, 4), "sd": round(sd, 4), "n": len(xs)}


def parse_log(path):
    tri, pyt = {}, {}
    for ln in open(path, errors="replace"):
        m = ROW.match(ln.strip())
        if not m:
            continue
        name = f"{m.group(1)}_b{m.group(2)}_v{m.group(3)}"
        tri[name] = float(m.group(5))
        pyt[name] = float(m.group(6))
    return tri, pyt


def main():
    work, arms = sys.argv[1], sys.argv[2:]
    per_arm = {}      # arm -> scenario -> [ms]
    pyt_ref = {}      # scenario -> [ms] (gold runs only)
    for arm in arms:
        logs = sorted(glob.glob(os.path.join(work, f"wl_{arm}_*.log")))
        per_arm[arm] = {}
        for lg in logs:
            tri, pyt = parse_log(lg)
            for k, v in tri.items():
                per_arm[arm].setdefault(k, []).append(v)
            if arm == "gold":
                for k, v in pyt.items():
                    pyt_ref.setdefault(k, []).append(v)
    scen = sorted({k for a in per_arm.values() for k in a})
    out = {"unit": "ms", "scenarios": {}, "pytorch_ref": {}}
    for k in scen:
        out["scenarios"][k] = {}
        for arm in arms:
            s = stats(per_arm[arm].get(k, []))
            if s:
                out["scenarios"][k][arm] = s
    for k, xs in pyt_ref.items():
        s = stats(xs)
        if s:
            out["pytorch_ref"][k] = s
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
