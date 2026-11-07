#!/usr/bin/env python3
"""Pick the next PRs to reproduce. Prints PR numbers, one per line (for run.sh --reproduce --next N).

    python3 next_prs.py --next 10              # not yet measured, in workloads.json order
    python3 next_prs.py --retry-failed         # have a result but 0 measured experiments
    python3 next_prs.py --status               # table: every workload PR -> measured | failed | unmet | pending

A PR is "measured" when results/<pr>.json has summary.measured > 0 (derive.py / record.py write it).
"unmet" = .work/unmet_<pr>.txt exists (requirements.py said no on this machine); it is retried only with --retry-unmet.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))


def state(pr):
    p = os.path.join(HERE, "results", f"{pr}.json")
    unmet = os.path.join(HERE, ".work", f"unmet_{pr}.txt")
    if os.path.exists(unmet):
        return "unmet", open(unmet).read().strip().splitlines()[:1]
    if not os.path.exists(p):
        return "pending", []
    d = json.load(open(p))
    s = d.get("summary") or {}
    if d.get("ts", "") < "2026-11-07":          # hand-entered scaffold data, never run here
        return "pending", ["old scaffold entry, not a run"]
    if s.get("measured", 0) > 0:
        return "measured", [f"x{s.get('reproduced_ratio_median')} {s.get('verdict')}"]
    t = s.get("tests") or {}
    if s.get("experiments", 0) == 0 and t.get("gold") == "PASS" and t.get("base") == "FAIL":
        return "measured", ["correctness-only PR: tests FAIL -> PASS, no timing by design"]
    if "ts" in d:
        return "failed", [f"ran, 0 measured ({s.get('tests')})"]
    return "pending", []


def main():
    a = sys.argv[1:]
    W = J("workloads.json")
    prs = [int(k) for k in W]
    if "--status" in a:
        for pr in prs:
            st, why = state(pr)
            print(f"{pr:>6}  {st:9} {' '.join(why)}")
        return
    want = []
    if "--retry-failed" in a:
        want = [pr for pr in prs if state(pr)[0] == "failed"]
    elif "--retry-unmet" in a:
        want = [pr for pr in prs if state(pr)[0] == "unmet"]
    else:
        n = int(a[a.index("--next") + 1]) if "--next" in a else 10
        want = [pr for pr in prs if state(pr)[0] == "pending"][:n]
    print("\n".join(str(p) for p in want))


main()
