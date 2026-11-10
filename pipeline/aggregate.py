#!/usr/bin/env python3
"""Fold every data/pr_<n>/results.<hw-slug>.json into results/all.json + results/manifest.json
(the two aggregates the dashboard reads) — atomically and under a lock.

    python3 pipeline/aggregate.py                            # rebuild results/all.json + manifest.json
    python3 pipeline/aggregate.py --write <pr> <json-file> [--hw <slug>]
        # atomically install data/pr_<n>/results.<slug>.json (slug = this machine's unless --hw), then rebuild

manifest.json: {"current": "<this machine's slug>", "prs": [..], "slugs": {"<pr>": ["rtx5070ti", ..]},
                "files": {"<pr>": {"<slug>": "data/pr_<n>/results.<slug>.json"}}}
all.json: the list of every results file (each entry carries "hw_slug").

Concurrency rules (several run.sh / run_one.sh may run at once):
  * every JSON is written to a temp file in the same dir and os.replace()d, so
    a reader (serve.py -> web/index.html) never sees a half-written file;
  * results/.lock (fcntl.flock) serialises the read-all + write step,
    so two finishing PRs cannot interleave and drop each other's entry;
  * each entry is re-read from disk under the lock, never from memory.
"""
import fcntl
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def rebuild_locked():
    agg, slugs, files, ai = [], {}, {}, {}
    for pr, hw, p in layout.all_results_files():
        try:
            d = json.load(open(p))
        except ValueError:
            print(f"skip unreadable {p}", file=sys.stderr)
            continue
        d.setdefault("hw_slug", hw)
        agg.append(d)
        slugs.setdefault(str(pr), []).append(hw)
        files.setdefault(str(pr), {})[hw] = layout.rel(p)
    # re-solved: CPU re-measures of solver arms (results.ai.<arm>.json), per PR -> {arm: measured_scenarios}
    for pr in layout.prs():
        for arm, ap in layout.results_ai_files(pr).items():
            try:
                scs = (json.load(open(ap)).get("scenarios") or {})
            except ValueError:
                continue
            n = sum(1 for e in scs.values() if e.get("ratio_arm_over_gold") is not None)
            ai.setdefault(str(pr), {})[arm] = {"scenarios": n, "file": layout.rel(ap)}
    atomic_write_json(layout.all_json(), agg)
    atomic_write_json(layout.manifest_json(), {"current": layout.hw_slug(), "prs": sorted(int(k) for k in slugs),
                                               "slugs": slugs, "files": files, "ai": ai})
    return len(agg), len(slugs)


def main():
    os.makedirs(layout.AGG, exist_ok=True)
    args = sys.argv[1:]
    hw = args[args.index("--hw") + 1] if "--hw" in args else None
    with open(layout.agg_lock(), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if len(args) >= 3 and args[0] == "--write":
            pr, src = int(args[1]), args[2]
            obj = json.load(open(src))
            obj["hw_slug"] = hw or layout.hw_slug()
            dst = layout.results_json(pr, hw)
            atomic_write_json(dst, obj)
            print(f"installed {layout.rel(dst)}")
        n, nprs = rebuild_locked()
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"wrote {layout.rel(layout.all_json())}: {n} result files, {nprs} prs")


if __name__ == "__main__":
    main()
