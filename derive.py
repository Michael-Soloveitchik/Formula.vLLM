#!/usr/bin/env python3
"""Compute, offline, everything the dashboard used to compute in the browser and
dump it into results/<pr>.json so the web page only displays.

    python3 derive.py --all              # backfill every results/<pr>.json (via aggregate.py --write)
    python3 derive.py 37225 41163        # specific PRs
record.py calls derive_pr() on every write, so new runs carry it automatically.

Per experiment -> experiments[i]["derived"]:
  unit, higher_is_better
  reported:   baseline, pr_solution (absolute, in the task unit), how, ratio, pct
              - author gave both numbers            -> used as is                 (how = "absolute")
              - author gave only a % or a delta     -> applied to OUR reproduced   (how = "our baseline ± %" / "± diff")
                baseline, so both columns are comparable in the same unit
  reproduced: baseline, pr_solution (our means), sd, ratio, pct
  reproduced_vs_reported: ratio_of_ratios (= reproduced ratio / reported ratio),
              verdict: reproduced (>=0.8) | partial (0.3-0.8) | not_reproduced (<0.3) | regression (reproduced ratio < 1) | unknown
Per PR -> "summary": headline experiment + median over experiments.
ratio = baseline / PR's solution for time units, PR's solution / baseline for throughput units (>1 = PR faster).
"""
import json, os, re, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def higher_better(unit):
    return bool(re.search(r"/s$|\bit/s\b|tok/s|req/s", unit or ""))


def ratio(val, bl, up):
    if not val or not bl:
        return None
    return (val / bl) if up else (bl / val)


def resolve_reported(rep, our_base, up):
    """-> (baseline, pr_solution, how) in the task unit, or (None, None, None)"""
    if not rep:
        return None, None, None
    bl = rep.get("base")
    if rep.get("patch") is not None:
        return (float(bl) if bl is not None else None), float(rep["patch"]), "absolute"
    ab = float(bl) if bl is not None else our_base
    if ab is None:
        return None, None, None
    if rep.get("delta") is not None:
        d = abs(float(rep["delta"]))
        return ab, (ab + d if up else ab - d), ("author baseline ± diff" if bl is not None else "our baseline ± diff")
    if rep.get("pct") is not None:
        f = float(rep["pct"]); f = f / 100 if abs(f) > 1 else f
        return ab, (ab * (1 + f) if up else ab * (1 - f)), ("author baseline ± %" if bl is not None else "our baseline ± %")
    return None, None, None


def derive_experiment(e, pyref=None):
    unit = e.get("unit") or ""
    up = higher_better(unit)
    a = e.get("arms") or {}
    b = (a.get("base") or {}).get("mean")
    if b is None and pyref and e.get("name") in pyref:
        b = pyref[e["name"]].get("mean")
    g = (a.get("gold") or {}).get("mean")
    rb, rg, how = resolve_reported(e.get("reported"), b, up)
    rr = ratio(rg, rb, up)
    mr = ratio(g, b, up)
    d = {
        "unit": unit, "higher_is_better": up,
        "reported": {"baseline": rb, "pr_solution": rg, "how": how, "ratio": rr,
                     "pct": None if rr is None else round((rr - 1) * 100, 2)},
        "reproduced": {"baseline": b, "pr_solution": g,
                       "baseline_sd": (a.get("base") or {}).get("sd"), "pr_solution_sd": (a.get("gold") or {}).get("sd"),
                       "ratio": mr, "pct": None if mr is None else round((mr - 1) * 100, 2)},
    }
    if mr is None:
        v, rr2 = "unknown", None
    elif rr is None or rr <= 0:
        v, rr2 = ("regression" if mr < 1 else "no_reported_number"), None
    else:
        rr2 = mr / rr
        v = "regression" if mr < 1 else ("reproduced" if rr2 >= 0.8 else "partial" if rr2 >= 0.3 else "not_reproduced")
    d["reproduced_vs_reported"] = {"ratio_of_ratios": None if rr2 is None else round(rr2, 3), "verdict": v}
    for k in ("reported", "reproduced"):
        for kk, vv in d[k].items():
            if isinstance(vv, float):
                d[k][kk] = round(vv, 6)
    return d


def derive_pr(out, headline=None):
    """mutates a results/<pr>.json dict: experiments[i].derived + summary"""
    exps = out.get("experiments") or []
    pyref = (out.get("latency") or {}).get("pytorch_ref") or {}
    for e in exps:
        e["derived"] = derive_experiment(e, pyref)
    ratios = [e["derived"]["reproduced"]["ratio"] for e in exps if e["derived"]["reproduced"]["ratio"]]
    rep = [e["derived"]["reported"]["ratio"] for e in exps if e["derived"]["reported"]["ratio"]]
    head = next((e for e in exps if e["name"] == headline), exps[0] if exps else None)
    out["summary"] = {
        "experiments": len(exps),
        "measured": len(ratios),
        "headline": head["name"] if head else None,
        "headline_derived": head["derived"] if head else None,
        "reproduced_ratio_median": round(statistics.median(ratios), 4) if ratios else None,
        "reproduced_ratio_max": round(max(ratios), 4) if ratios else None,
        "reported_ratio_median": round(statistics.median(rep), 4) if rep else None,
        "verdict": head["derived"]["reproduced_vs_reported"]["verdict"] if head else "unknown",
        "tests": {k: (out.get("arms") or {}).get(k, {}).get("tests") for k in ("base", "gold")},
    }
    return out


def main():
    prs = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if "--all" in sys.argv:
        prs = sorted(int(f[:-5]) for f in os.listdir(os.path.join(HERE, "results")) if re.fullmatch(r"\d+\.json", f))
    for pr in prs:
        p = os.path.join(HERE, "results", f"{pr}.json")
        d = json.load(open(p))
        try:
            head = json.load(open(os.path.join(HERE, "tasks", f"pr_{pr}", "metadata.json"))).get("workload", {}).get("headline")
        except Exception:
            head = None
        derive_pr(d, head)
        tmp = os.path.join(HERE, ".work", f"derive_{pr}.json")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        json.dump(d, open(tmp, "w"), indent=1)
        subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py"), "--write", str(pr), tmp], check=True, capture_output=True)
        s = d["summary"]
        print(f"{pr}: {s['measured']}/{s['experiments']} measured, reproduced x{s['reproduced_ratio_median']} vs reported x{s['reported_ratio_median']} -> {s['verdict']}")


if __name__ == "__main__":
    main()
