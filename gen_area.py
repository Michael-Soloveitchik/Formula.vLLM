#!/usr/bin/env python3
"""Build data/area.json: pr -> {field, area} for the 202 benchmark PRs only.

web/index.html needs field/area to construct each PR's data/prs/<field>/<area>/<pr>.md
path. The full crawl data/index.jsonl is ~10 MB; this subset is ~20 KB, so the
dashboard loads instantly even over a slow link (Tailscale Funnel). Re-run after
changing index.json or data/index.jsonl.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    want = {x["pr"] for x in json.load(open(os.path.join(HERE, "index.json")))}
    area = {}
    for line in open(os.path.join(HERE, "data", "index.jsonl")):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("pr") in want:
            area[str(r["pr"])] = {"field": r.get("field"), "area": r.get("area")}
    out = os.path.join(HERE, "data", "area.json")
    json.dump(area, open(out, "w"), separators=(",", ":"))
    print(f"wrote {out}: {len(area)} prs")


if __name__ == "__main__":
    main()
