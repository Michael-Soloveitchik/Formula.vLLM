#!/usr/bin/env python3
"""Fold per-arm workload logs into one latency dict (per scenario, mean ± sd).

Usage:
    python3 pipeline/parse_workload.py <work_dir> <arm1> [arm2 ...]   (work_dir = data/pr_<n>/work)

Reads <work_dir>/wl_<arm>_<i>.log for every repeat i and extracts every metric
line it recognises (one "scenario" per metric):

  * benchmark_topk_topp.py SUMMARY TABLE rows
        <scenario> <batch> <vocab> <ops>% <triton_ms> <pytorch_ms> <speedup>x ...
    -> "<scenario>_b<batch>_v<vocab>" (ms, Triton column); the PyTorch column of
       the PR's-solution arm is kept as `pytorch_ref` (= pre-PR number when the baseline tree
       cannot run the script at all, e.g. 33538 introduces it)
  * "AllPool.forward median: 39.65 us"            -> allpool_forward_median (us)
  * benchmark_throughput.py
        "Throughput: 123.4 requests/s, 5678.9 total tokens/s, 910.1 output tokens/s"
    -> throughput_req_s (req/s), throughput_total_tok_s, throughput_output_tok_s (tok/s)

Prints JSON:
    {"scenarios": {"<name>": {"unit": "ms", "<arm>": {"mean":..,"sd":..,"n":..}, ...}},
     "pytorch_ref": {"<name>": {"mean":..,"sd":..,"n":..}}}
Units ending in "/s" are higher-is-better (web/index.html handles direction).
"""
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402

TOPK_ROW = re.compile(
    r"^(\w+?)\s+(\d+)\s+(\d+)\s+(\d+)%\s+([\d.]+|inf)\s+([\d.]+|inf)\s+([\d.]+|inf)x"
)
ALLPOOL = re.compile(r"AllPool\.forward median:\s*([\d.]+)\s*us")
# benchmarks/kv_cache/benchmark_block_pool.py (tabulate grid): | iters | total | alloc | get avg | get max | free avg | free max |
BLOCKPOOL = re.compile(r"^\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|")
# vendored data/pr_<n>/bench.py: "<metric> [<label>] <number>"  (unit from task.json experiments)
GENERIC = re.compile(r"^([A-Za-z_]\w*)(?:\s+[A-Za-z_][\w.'-]*)?\s+(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\s*$")
THROUGHPUT = re.compile(
    r"Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s,\s*([\d.]+)\s*output tokens/s"
)


def stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return {"mean": round(m, 4), "sd": round(sd, 4), "n": len(xs)}


def _units_for(path):
    """units declared in data/pr_<n>/task.json experiments (for GENERIC lines); pr taken from the work dir path"""
    m = re.search(r"pr_(\d+)", path)
    try:
        meta = layout.load_task(int(m.group(1)))
        return {x["name"]: x.get("unit") or "unit" for x in meta.get("experiments", [])}
    except Exception:
        return {}


def parse_log(path):
    """-> ({scenario: (value, unit)}, {scenario: pytorch_ms})"""
    vals, pyt = {}, {}
    units = _units_for(path)
    for ln in open(path, errors="replace"):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        m = TOPK_ROW.match(s)
        if m:
            name = f"{m.group(1)}_b{m.group(2)}_v{m.group(3)}"
            vals[name] = (float(m.group(5)), "ms")
            pyt[name] = float(m.group(6))
            continue
        m = BLOCKPOOL.match(s)
        if m:
            vals[f"get_new_blocks_a{m.group(1)}"] = (float(m.group(2)), "ms")
            vals[f"free_blocks_a{m.group(1)}"] = (float(m.group(3)), "ms")
            continue
        m = GENERIC.match(s)
        if m and (m.group(1) in units or not units):
            vals[m.group(1)] = (float(m.group(2)), units.get(m.group(1), "unit"))
            continue
        m = ALLPOOL.search(s)
        if m:
            vals["allpool_forward_median"] = (float(m.group(1)), "us")
            continue
        m = THROUGHPUT.search(s)
        if m:
            vals["throughput_req_s"] = (float(m.group(1)), "req/s")
            vals["throughput_total_tok_s"] = (float(m.group(2)), "tok/s")
            vals["throughput_output_tok_s"] = (float(m.group(3)), "tok/s")
    return vals, pyt


def main():
    work, arms = sys.argv[1], sys.argv[2:]
    per_arm = {}      # arm -> scenario -> [values]
    units = {}        # scenario -> unit
    pyt_ref = {}      # scenario -> [ms] (PR-solution runs only)
    for arm in arms:
        per_arm[arm] = {}
        # exact arm match: wl_<arm>_<n>.log (a bare glob would also match
        # another arm whose slug starts with this one, e.g. claude vs claude___PR_s_solution)
        pat = re.compile(rf"^wl_{re.escape(arm)}_(\d+)\.log$")
        logs = sorted(os.path.join(work, f) for f in os.listdir(work) if pat.match(f))
        for lg in logs:
            vals, pyt = parse_log(lg)
            for k, (v, u) in vals.items():
                per_arm[arm].setdefault(k, []).append(v)
                units[k] = u
            if arm == "gold":
                for k, v in pyt.items():
                    pyt_ref.setdefault(k, []).append(v)
    out = {"scenarios": {}, "pytorch_ref": {}}
    for k in sorted(units):
        out["scenarios"][k] = {"unit": units[k]}
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
