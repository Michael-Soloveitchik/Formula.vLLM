#!/usr/bin/env python3
"""Write results/<pr>.json from a task's .work dir (and refresh all.json/manifest).

    python3 record.py <pr>

Called by run_one.sh right after the base/gold measurement (so the dashboard
shows measured numbers immediately) and again after every solver arm. Reads:
  .work/pr_<pr>/arms.json        {"base":{"tests":..},"gold":{..},"<harness>":{"tests":..,"slug":..}}
  .work/pr_<pr>/wl_<slug>_<i>.log  workload logs (parse_workload.py)
  tasks/pr_<pr>/metadata.json    experiments[], workload, wheel
  index.json                     title/field/reported gain
Writes through aggregate.py --write (lock + atomic), never directly.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    pr = int(sys.argv[1])
    work = os.path.join(HERE, ".work", f"pr_{pr}")
    task = os.path.join(HERE, "tasks", f"pr_{pr}")
    idx = {x["pr"]: x for x in json.load(open(os.path.join(HERE, "index.json")))}[pr]
    try:
        meta = json.load(open(os.path.join(task, "metadata.json")))
    except Exception:
        meta = {}
    arms = json.load(open(os.path.join(work, "arms.json")))
    slugs = [v.get("slug", k) for k, v in arms.items()]
    slug2name = {v.get("slug", k): k for k, v in arms.items()}

    latency, experiments = None, None
    wl = (meta.get("workload") or {}).get("cmd")
    if wl:
        out = subprocess.run([sys.executable, os.path.join(HERE, "parse_workload.py"), work] + slugs,
                             capture_output=True, text=True, check=True).stdout
        latency = json.loads(out)
        for s, d in latency["scenarios"].items():
            latency["scenarios"][s] = {slug2name.get(k, k): v for k, v in d.items()}
        listed = meta.get("experiments") or []
        names = [e["name"] for e in listed] + [s for s in latency["scenarios"] if s not in {e["name"] for e in listed}]
        rep_by = {e["name"]: e.get("reported") for e in listed}
        pyref = latency.get("pytorch_ref") or {}
        experiments = []
        for n in names:
            a = dict(latency["scenarios"].get(n, {}))
            unit = a.pop("unit", None) or (meta.get("workload") or {}).get("unit") or "ms"
            note = None
            if "base" not in a and n in pyref:
                a["base"] = pyref[n]
                note = (meta.get("workload") or {}).get("base_note") or "base = script's PyTorch column"
            e = {"name": n, "unit": unit, "workload": wl, "reported": rep_by.get(n), "arms": a}
            if note:
                e["note"] = note
            experiments.append(e)
    elif os.path.exists(os.path.join(task, "results_measured.json")):
        latency = json.load(open(os.path.join(task, "results_measured.json")))

    # per-arm log tails so a failure is diagnosable from the JSON alone
    logs = {}
    for k, v in arms.items():
        slug = v.get("slug", k)
        ent = {}
        for kind, fn in (("tests", f"test_{slug}.log"), ("solve", f"solve_{slug}.log")):
            p = os.path.join(work, fn)
            if os.path.exists(p):
                txt = open(p, errors="replace").read()
                ent[kind] = {"bytes": len(txt), "tail": txt[-1500:]}
        n = len([f for f in os.listdir(work) if f.startswith(f"wl_{slug}_") and f.endswith(".log")])
        if n:
            ent["workload_runs"] = n
        p = os.path.join(work, f"solve_{slug}.patch")
        if os.path.exists(p):
            ent["patch_bytes"] = os.path.getsize(p)
        logs[k] = ent

    bres, gres = arms.get("base", {}).get("tests"), arms.get("gold", {}).get("tests")
    out = {
        "schema": "experiments_v2" if experiments is not None else None,
        "experiments": experiments,
        "pr": pr,
        "pr_url": f"https://github.com/vllm-project/vllm/pull/{pr}",
        "title": idx.get("title"), "field": idx.get("field"),
        "reported_gain": idx.get("reported_gain"), "gain_kind": idx.get("gain_kind"),
        "base_sha": idx.get("base_sha"), "merge_sha": idx.get("merge_sha"),
        "image": {"mode": "wheel" if meta.get("wheel") else "source", "wheel": meta.get("wheel"),
                  "workload_cmd": wl, "repeats": int(os.environ.get("REPEATS", "5"))},
        "arms": arms,
        "improved": gres == "PASS" and bres in ("FAIL", "PASS", "SKIP") and gres == "PASS",
        "latency": latency,
        "logs": logs,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = os.path.join(work, "result.json")
    json.dump(out, open(tmp, "w"), indent=1)
    subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py"), "--write", str(pr), tmp], check=True)
    print(f"recorded results/{pr}.json arms={list(arms)} experiments={len(experiments or [])}")


if __name__ == "__main__":
    main()
