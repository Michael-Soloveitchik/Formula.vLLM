#!/usr/bin/env python3
"""Generate benchmark/index.json (202 PRs) from the pr202 manifest.

Each entry: pr, title, date, field, hardware, reported gain, base_sha
(= merge_sha^, the real pre-PR tree) and merge_sha. base_sha resolved from a
local vllm git clone; falls back to null if the repo/commit is absent.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "..", "benchmark_ideas", "pr202", "manifest.json")
VLLM_GIT = os.environ.get("VLLM_GIT",
                          os.path.join(HERE, "..", "..", "vllm"))
OUT = os.path.join(HERE, "index.json")


def rev(ref):
    r = subprocess.run(["git", "-C", VLLM_GIT, "rev-parse", "--verify", "-q", ref],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    man = json.load(open(MANIFEST))
    have_git = os.path.isdir(os.path.join(VLLM_GIT, ".git"))
    out = []
    for e in man:
        av = e.get("availability", {})
        merge = av.get("merge_sha")
        base = rev(f"{merge}^") if (have_git and merge) else None
        pg = e.get("perf_gain") or {}
        out.append({
            "pr": e["pr"],
            "title": e.get("title"),
            "date": e.get("date"),
            "field": e.get("field"),
            "hardware": e.get("hardware") or [],
            "reported_gain": pg.get("top_gain"),
            "gain_kind": pg.get("gain_kind"),
            "gain_evidence": pg.get("gain_evidence"),
            "base_sha": base,
            "merge_sha": merge,
            "source_path": av.get("path"),
        })
    json.dump(out, open(OUT, "w"), indent=1)
    n_base = sum(1 for x in out if x["base_sha"])
    print(f"wrote {OUT}: {len(out)} prs, {n_base} with resolved base_sha "
          f"(git={'yes' if have_git else 'no'})")


if __name__ == "__main__":
    main()
