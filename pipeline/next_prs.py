#!/usr/bin/env python3
"""Pick the next PRs to reproduce on THIS machine. Prints PR numbers, one per line (for run.sh --reproduce --next N).

    python3 pipeline/next_prs.py --next 10              # not yet measured here, in PR order
    python3 pipeline/next_prs.py --retry-failed         # have a result here but 0 measured experiments
    python3 pipeline/next_prs.py --status               # table: every workload PR -> measured | failed | unmet | pending

The set = every data/pr_<n>/task.json with a workload cmd. A PR is "measured" when
data/pr_<n>/results.<hw-slug>.json (this machine's slug) has summary.measured > 0 (derive.py / record.py
write it). "unmet" = data/pr_<n>/work/unmet.txt exists (requirements.py said no on this machine); it is
retried only with --retry-unmet. Results from other hardware slugs are listed but do not count.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402


def state(pr):
    p = layout.results_json(pr)
    unmet = layout.work_file(pr, "unmet.txt")
    others = [h for h in layout.results_files(pr) if h != layout.hw_slug()]
    extra = [f"(also measured on {','.join(others)})"] if others else []
    if os.path.exists(unmet):
        return "unmet", open(unmet).read().strip().splitlines()[:1] + extra
    if not os.path.exists(p):
        return "pending", extra
    d = json.load(open(p))
    s = d.get("summary") or {}
    if s.get("measured", 0) > 0:
        return "measured", [f"x{s.get('reproduced_ratio_median')} {s.get('verdict')}"] + extra
    t = s.get("tests") or {}
    if s.get("experiments", 0) == 0 and t.get("gold") == "PASS" and t.get("base") == "FAIL":
        return "measured", ["correctness-only PR: tests FAIL -> PASS, no timing by design"] + extra
    if "ts" in d:
        return "failed", [f"ran, 0 measured ({s.get('tests')})"] + extra
    return "pending", extra


def main():
    a = sys.argv[1:]
    prs = layout.workload_prs()
    if "--status" in a:
        print(f"hardware slug: {layout.hw_slug()}")
        for pr in prs:
            st, why = state(pr)
            print(f"{pr:>6}  {st:9} {' '.join(why)}")
        return
    if "--retry-failed" in a:
        want = [pr for pr in prs if state(pr)[0] == "failed"]
    elif "--retry-unmet" in a:
        want = [pr for pr in prs if state(pr)[0] == "unmet"]
    else:
        n = int(a[a.index("--next") + 1]) if "--next" in a else 10
        want = [pr for pr in prs if state(pr)[0] == "pending"][:n]
    print("\n".join(str(p) for p in want))


if __name__ == "__main__":
    main()
