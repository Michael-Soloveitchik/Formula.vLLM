#!/usr/bin/env python3
"""Write a `requirements` block into index.json for every PR that has a workload.

    python3 gen_requirements.py            # update index.json in place
    python3 gen_requirements.py --print    # show the table, do not write

Derived from workloads.json (cmd, needs), clusters.json (models, sizes, gating)
and tasks/pr_<n>/metadata.json (wheel mode), plus the per-PR overrides below
(things only visible by reading the benchmark: tp-size, fp8, server needed...).

Schema (index.json[pr]["requirements"]):
  hardware: gpu (bool), gpu_count, vram_gb (per GPU), compute_cap_min / _max,
            gpu_hint (what the author used), ram_gb, disk_gb
  software: pipeline (wheel | needs-wheel-task), docker, nvidia_container_runtime, git, python3, hf_cli,
            vllm_wheel (pypi version, wheel-mode) | cuda_build (source-mode
            Dockerfile.gpu tag), models [...], datasets [...], env [...],
            needs_server (bench talks to a running vllm server)
requirements.py checks this block against the machine.
"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))

# source-mode GPU images are built by Dockerfile.gpu (CUDA 12.1 -> sm_90 max)
CU121_CC_MAX = 9.0
WHEEL_IMG_GB = 30     # vllm-task:<pr> wheel images are 13-25 GB on disk
SRC_IMG_GB = 40       # source build image + build cache
CPU_IMG_GB = 10

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


def derive(pr, w, clusters):
    pr = int(pr)
    try:
        meta = json.load(open(os.path.join(HERE, "tasks", f"pr_{pr}", "metadata.json")))
    except Exception:
        meta = {}
    gpu = True   # every vLLM workload runs through the GPU image; the "cpu" tag in workloads.json only says the bench itself has no kernels
    wheel = meta.get("wheel")
    hw = {"gpu": gpu, "gpu_count": 1 if gpu else 0, "vram_gb": 8 if gpu else 0,
          "compute_cap_min": 7.0 if gpu else None, "compute_cap_max": None,
          "gpu_hint": None, "ram_gb": 8, "disk_gb": CPU_IMG_GB}
    sw = {"docker": True, "nvidia_container_runtime": gpu, "git": True, "python3": True,
          "hf_cli": False, "hf_token": False, "vllm_wheel": wheel, "cuda_build": None,
          "models": [], "datasets": [], "env": [], "needs_server": False,
          # runner path: wheel (tasks/pr_N/metadata.json has "wheel": pypi version -> run_one.sh
          # wheel image) | needs-wheel-task (no vendored task yet: metadata.json with wheel,
          # test_files, source_files, workload.cmd must be written first; gen_workloads.py
          # already picks the first PyPI release after merge_sha)
          "pipeline": "wheel" if wheel else "needs-wheel-task"}
    if gpu:
        hw["disk_gb"] = WHEEL_IMG_GB
    # models come from the clusters this PR belongs to, but only the ones the
    # workload cmd actually names (a PR can sit in a cpu + a gpu cluster)
    cmd = w.get("cmd", "")
    for ck, c in clusters.items():
        if pr not in c["prs"] or not c.get("model"):
            continue
        if not gpu:
            continue   # a cpu workload never loads a model, whatever cluster it also sits in
        if c["model"] in cmd or len([k for k, v in clusters.items() if pr in v["prs"] and v.get("model")]) == 1:
            sw["models"].append(c["model"])
            sw["hf_cli"] = True
            hw["disk_gb"] += gb(c["size"])
            if c.get("gated"):
                sw["env"].append(c["gated"])
            if c.get("dataset"):
                sw["datasets"].append(c["dataset"] + ".json")
    o = OVERRIDE.get(pr, {})
    for k, v in o.items():
        (hw if k in hw else sw)[k] = v if not isinstance(v, list) else sorted(set((hw if k in hw else sw)[k] + v))
    if gpu and hw["compute_cap_max"] and hw["compute_cap_min"] and hw["compute_cap_min"] > hw["compute_cap_max"]:
        hw["compute_cap_max"] = None
    return {"hardware": hw, "software": sw}


def main():
    W, C = J("workloads.json"), J("clusters.json")["clusters"]
    idx = J("index.json")
    by = {x["pr"]: x for x in idx}
    rows = []
    for p, w in W.items():
        r = derive(p, w, C)
        by[int(p)]["requirements"] = r
        h, s = r["hardware"], r["software"]
        rows.append((int(p), "gpu" if h["gpu"] else "cpu", f"x{h['gpu_count']} {h['vram_gb']}GB cc>={h['compute_cap_min']}" + (f" cc<={h['compute_cap_max']}" if h["compute_cap_max"] else "") if h["gpu"] else "-",
                     f"disk {h['disk_gb']:.0f}GB", s["vllm_wheel"] and f"wheel {s['vllm_wheel']}" or (s["cuda_build"] or "cpu image"),
                     s["pipeline"], ",".join(s["models"]) or "-", ",".join(s["env"]) or "", "server" if s["needs_server"] else "", h["gpu_hint"] or ""))
    for r in sorted(rows):
        print(" | ".join(str(x) for x in r))
    if "--print" in sys.argv:
        return
    tmp = os.path.join(HERE, "index.json.tmp")
    with open(tmp, "w") as f:
        json.dump(idx, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, os.path.join(HERE, "index.json"))
    print(f"wrote requirements for {len(rows)} PRs into index.json")


main()
