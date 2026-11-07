#!/usr/bin/env python3
"""Write the `requirements` block of data/pr_<n>/task.json for every PR that has a workload,
then regenerate index.json (which carries a copy for the web / requirements.py).

    python3 gen/gen_requirements.py            # update task.json files in place (+ index.json)
    python3 gen/gen_requirements.py --print    # show the table, do not write

Derived from task.json (workload cmd, wheel) and clusters.json (models, sizes, gating), plus the
per-PR overrides below (things only visible by reading the benchmark: tp-size, fp8, server needed...).

Schema (task.json["requirements"]):
  hardware: gpu (bool), gpu_count, vram_gb (per GPU), compute_cap_min / _max,
            gpu_hint (what the author used), ram_gb, disk_gb
  software: pipeline (wheel | needs-wheel-task), docker, nvidia_container_runtime, git, python3, hf_cli,
            vllm_wheel (pypi version, wheel-mode), models [...], datasets [...], env [...],
            needs_server (bench talks to a running vllm server)
pipeline/requirements.py checks this block against the machine.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import gen_index, layout  # noqa: E402

WHEEL_IMG_GB = 30     # vllm-task:<pr> wheel images are 13-25 GB on disk

# per-PR facts not derivable from the files (from the PR / bench script)
OVERRIDE = {
    9972:  {"compute_cap_min": 8.9, "gpu_hint": "FP8 model (Ada/Hopper+)", "vram_gb": 16},
    12583: {"gpu_count": 2, "vram_gb": 48, "gpu_hint": "Mixtral-8x7B TP=2", "env": ["ALLOW_MIXTRAL"], "hf_token": True},
    23504: {"compute_cap_min": 8.9, "gpu_hint": "H200 (--tp-size 8 only shapes the problem; one GPU runs it)", "vram_gb": 24},
    16436: {"needs_server": True, "gpu_hint": "serving bench: needs a vllm server + client", "vram_gb": 6},
    6442:  {"gpu_hint": "A100 in the PR; v0 Sampler", "vram_gb": 8},
    28280: {"needs_server": True, "compute_cap_min": 8.9, "vram_gb": 16, "gpu_hint": "serving (sharegpt) on Qwen2-0.5B / Llama-3-8B-FP8"},
    37225: {"compute_cap_min": 7.5, "gpu_hint": "Triton kernel"},
    33538: {"compute_cap_min": 7.5, "gpu_hint": "Triton kernel"},
    41163: {"compute_cap_min": 7.0, "gpu_hint": "AllPool.forward on cuda:0"},
    22940: {"vram_gb": 4},
}


def gb(s):
    t = 0.0
    for n, u in re.findall(r"([\d.]+)\s*(GB|MB)", s or ""):
        t += float(n) / (1024 if u == "MB" else 1)
    return t


def derive(pr, task, clusters):
    """every vLLM workload runs through the GPU wheel image (the bench itself may have no kernels)"""
    wheel = task.get("wheel")
    hw = {"gpu": True, "gpu_count": 1, "vram_gb": 8, "compute_cap_min": 7.0, "compute_cap_max": None,
          "gpu_hint": None, "ram_gb": 8, "disk_gb": WHEEL_IMG_GB}
    sw = {"docker": True, "nvidia_container_runtime": True, "git": True, "python3": True,
          "hf_cli": False, "hf_token": False, "vllm_wheel": wheel,
          "models": [], "datasets": [], "env": [], "needs_server": False,
          # wheel = task.json has "wheel" (run_one.sh builds the per-PR wheel image) |
          # needs-wheel-task = no wheel yet: python3 gen/gen_tasks.py <pr> fills wheel/source_files/test_files
          "pipeline": "wheel" if wheel else "needs-wheel-task"}
    # models come from the clusters this PR belongs to, but only the ones the workload cmd names
    # (a PR can sit in a cpu + a gpu cluster)
    cmd = (task.get("workload") or {}).get("cmd", "") or ""
    model_clusters = [k for k, v in clusters.items() if pr in v["prs"] and v.get("model")]
    for ck in model_clusters:
        c = clusters[ck]
        if c["model"] in cmd or len(model_clusters) == 1:
            sw["models"].append(c["model"])
            sw["hf_cli"] = True
            hw["disk_gb"] += gb(c["size"])
            if c.get("gated"):
                sw["env"].append(c["gated"])
            if c.get("dataset"):
                sw["datasets"].append(c["dataset"] + ".json")
    for k, v in OVERRIDE.get(pr, {}).items():
        tgt = hw if k in hw else sw
        tgt[k] = v if not isinstance(v, list) else sorted(set(tgt[k] + v))
    return {"hardware": hw, "software": sw}


def main():
    C = json.load(open(layout.clusters_json()))["clusters"]
    rows, changed = [], 0
    for pr in layout.workload_prs():
        t = layout.load_task(pr)
        r = derive(pr, t, C)
        h, s = r["hardware"], r["software"]
        rows.append((pr, f"x{h['gpu_count']} {h['vram_gb']}GB cc>={h['compute_cap_min']}", f"disk {h['disk_gb']:.0f}GB",
                     f"wheel {s['vllm_wheel']}" if s["vllm_wheel"] else "no wheel", s["pipeline"],
                     ",".join(s["models"]) or "-", ",".join(s["env"]) or "", "server" if s["needs_server"] else "", h["gpu_hint"] or ""))
        if "--print" in sys.argv:
            continue
        if t.get("requirements") != r:
            t["requirements"] = r
            changed += layout.save_task(pr, t)
    for r in sorted(rows):
        print(" | ".join(str(x) for x in r))
    if "--print" in sys.argv:
        return
    _, idx_changed = gen_index.write()
    print(f"requirements for {len(rows)} PRs: {changed} task.json updated; index.json {'rewritten' if idx_changed else 'unchanged'}")


if __name__ == "__main__":
    main()
