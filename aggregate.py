#!/usr/bin/env python3
"""Fold results/<pr>.json into results/all.json — atomically and under a lock.

    python3 aggregate.py                 # rebuild results/all.json
    python3 aggregate.py --write <pr> <json-file>   # atomically install results/<pr>.json, then rebuild

Concurrency rules (several run.sh / run_one.sh may run at once):
  * every JSON is written to a temp file in the same dir and os.replace()d, so
    a reader (serve.py -> web/index.html) never sees a half-written file;
  * results/.lock (fcntl.flock) serialises the read-all + write-all.json step,
    so two finishing PRs cannot interleave and drop each other's entry;
  * each entry is re-read from disk under the lock, never from memory, so the
    aggregate always reflects what is actually in results/.
"""
import fcntl
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LOCK = os.path.join(RES, ".lock")


def atomic_write_json(path, obj):
    d = os.path.dirname(path)
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
    agg = []
    for f in sorted(glob.glob(os.path.join(RES, "*.json"))):
        if os.path.basename(f) in ("all.json", "manifest.json") or os.path.basename(f).startswith(".tmp-"):
            continue
        try:
            agg.append(json.load(open(f)))
        except ValueError:
            print(f"skip unreadable {f}", file=sys.stderr)
    atomic_write_json(os.path.join(RES, "all.json"), agg)
    # tiny index for the web: it fetches results/<pr>.json per PR from this list
    atomic_write_json(os.path.join(RES, "manifest.json"),
                      {"prs": sorted(int(x["pr"]) for x in agg if "pr" in x)})
    return len(agg)


def main():
    os.makedirs(RES, exist_ok=True)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if len(sys.argv) >= 4 and sys.argv[1] == "--write":
            pr, src = sys.argv[2], sys.argv[3]
            obj = json.load(open(src))
            atomic_write_json(os.path.join(RES, f"{pr}.json"), obj)
            print(f"installed results/{pr}.json")
        n = rebuild_locked()
        fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"wrote results/all.json: {n} prs")


if __name__ == "__main__":
    main()
