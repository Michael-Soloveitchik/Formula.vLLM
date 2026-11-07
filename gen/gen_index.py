#!/usr/bin/env python3
"""Generate index.json (the 202-PR catalog the web and the scripts load) from data/pr_*/task.json.

    python3 gen/gen_index.py

One entry per data/pr_<n>/: the catalog facts (title, date, field, area, hardware, reported gain,
shas) plus, when the PR has a workload, its workload summary, experiments (reported numbers +
scope notes, so the dashboard can show reported-only rows before a run), requirements and the
hardware slugs that have a generated workload dir / a results file. No timestamps: a regeneration
with unchanged task.json files is a no-op diff. task.json is the owner of every fact here.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402

CATALOG = ("title", "date", "field", "area", "hardware", "reported_gain", "gain_kind", "gain_evidence",
           "base_sha", "merge_sha", "source_path")


def entry(pr):
    t = layout.load_task(pr)
    e = {"pr": pr}
    for k in CATALOG:
        e[k] = t.get(k)
    e["url"] = t.get("url") or f"https://github.com/vllm-project/vllm/pull/{pr}"
    wl = t.get("workload") or {}
    if t.get("experiments"):          # reported numbers + scope notes: shown by the web before any run
        e["experiments"] = t["experiments"]
        e["experiments_note"] = t.get("experiments_note")
    if wl.get("cmd"):
        e["workload"] = {k: wl.get(k) for k in ("kind", "script", "cmd", "unit", "needs", "headline")}
        e["wheel"] = t.get("wheel")
        e["requirements"] = t.get("requirements")
        e["hardware_overrides"] = sorted(t.get("hardware_overrides") or {})
        e["workload_hws"] = layout.workload_hws(pr)
    e["results_hws"] = sorted(layout.results_files(pr))
    return e


def build():
    return [entry(pr) for pr in layout.prs()]


def write(out=None):
    out = out or layout.index_json()
    data = build()
    txt = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    try:
        if open(out).read() == txt:
            return data, False
    except Exception:
        pass
    tmp = out + ".tmp"
    open(tmp, "w").write(txt)
    os.replace(tmp, out)
    return data, True


def main():
    data, changed = write()
    n_wl = sum(1 for e in data if "workload" in e)
    n_task = sum(1 for e in data if e.get("wheel"))
    print(f"{'wrote' if changed else 'unchanged'} {layout.rel(layout.index_json())}: {len(data)} prs, "
          f"{n_wl} with a workload, {n_task} wheel-mode tasks")


if __name__ == "__main__":
    main()
