#!/usr/bin/env python3
"""Per-PR requirements check: can THIS machine reproduce PR <n> (baseline vs PR's solution)?

    python3 pipeline/requirements.py 22940 37225          # checklist per PR, exit 1 if any unmet
    python3 pipeline/requirements.py --quiet 22940        # print only unmet lines (run.sh --reproduce)
    python3 pipeline/requirements.py --install 22940      # also try to install missing software

Each PR carries its own `requirements` block in data/pr_<n>/task.json (written by
gen/gen_requirements.py from the PR's workload, cluster and wheel):
  hardware: gpu, gpu_count, vram_gb, compute_cap_min/max, ram_gb, disk_gb
  software: docker, nvidia_container_runtime, git, python3, hf_cli, hf_token,
            vllm_wheel, models, datasets, env, needs_server
Hardware that is missing -> the PR is skipped with the reason. Software that is
missing -> install hint; with --install the no-sudo ones (pip packages) are installed on
the spot, the rest still skip with the apt command.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402

GB = 2**30


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return 1, str(e)


def machine():
    m = {"docker": sh("docker info >/dev/null 2>&1")[0] == 0,
         "git": shutil.which("git") is not None, "python3": shutil.which("python3") is not None,
         "hf_cli": shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None,
         "hf_token": bool(os.environ.get("HF_TOKEN")) or os.path.exists(os.path.expanduser("~/.cache/huggingface/token")),
         "nvidia_container_runtime": "nvidia" in sh("docker info 2>/dev/null | grep -i Runtimes")[1],
         "github": sh("git ls-remote --exit-code https://github.com/vllm-project/vllm.git HEAD >/dev/null 2>&1", 30)[0] == 0,
         "gpus": [], "disk_free_gb": shutil.disk_usage(ROOT).free / GB, "hw_slug": layout.hw_slug()}
    rc, out = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits")
    if rc == 0 and out:
        for ln in out.splitlines():
            n, mem, cc = [x.strip() for x in ln.split(",")]
            m["gpus"].append({"name": n, "vram_gb": float(mem) / 1024, "cc": float(cc)})
    try:
        m["ram_gb"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except Exception:
        m["ram_gb"] = 0
    return m


# software item -> (how to install, can we do it without sudo)
INSTALL = {
    "docker": ("sudo apt-get install -y docker.io && sudo systemctl start docker", False),
    "nvidia_container_runtime": ("sudo apt-get install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker", False),
    "git": ("sudo apt-get install -y git", False),
    "python3": ("sudo apt-get install -y python3", False),
    "hf_cli": ("python3 -m pip install -q huggingface_hub[cli]", True),
    "hf_token": ("hf auth login   (or export HF_TOKEN=...)", False),
}


def check(pr, M, install=False):
    """-> list of (ok, kind, requirement, detail); kind in hardware|software"""
    R = []
    add = lambda ok, kind, req, det="": R.append((ok, kind, req, det))
    t = layout.resolve_task(pr)
    if not t:
        add(False, "software", f"data/pr_{pr}/task.json", "unknown PR")
        return R
    req = t.get("requirements")
    if not req:
        add(False, "software", "requirements block in task.json", "no workload for this PR (gen/gen_requirements.py covers the PRs with a workload cmd)")
        return R
    hw, sw = req["hardware"], req["software"]

    # ---- hardware ----
    g = M["gpus"]
    if hw["gpu"]:
        add(len(g) >= hw["gpu_count"], "hardware", f"NVIDIA GPU x{hw['gpu_count']}", f"found {len(g)}: " + ", ".join(x["name"] for x in g) if g else "nvidia-smi found none")
        if g:
            g0 = g[0]
            add(g0["vram_gb"] >= hw["vram_gb"], "hardware", f"VRAM >= {hw['vram_gb']} GB", f"{g0['vram_gb']:.0f} GB")
            if hw.get("compute_cap_min"):
                add(g0["cc"] >= hw["compute_cap_min"], "hardware", f"compute capability >= {hw['compute_cap_min']}", f"sm_{int(g0['cc']*10)}" + (f" ({hw['gpu_hint']})" if hw.get("gpu_hint") else ""))
            if hw.get("compute_cap_max"):
                add(g0["cc"] <= hw["compute_cap_max"], "hardware", f"compute capability <= {hw['compute_cap_max']}", f"sm_{int(g0['cc']*10)}")
    add(M["ram_gb"] >= hw["ram_gb"], "hardware", f"RAM >= {hw['ram_gb']} GB", f"{M['ram_gb']:.0f} GB")
    add(M["disk_free_gb"] >= hw["disk_gb"], "hardware", f"free disk >= {hw['disk_gb']:.0f} GB", f"{M['disk_free_gb']:.0f} GB free")

    # ---- software ----
    pipe = sw.get("pipeline", "wheel" if sw.get("vllm_wheel") else "needs-wheel-task")
    PIPES = {"wheel": (True, f"run_one.sh wheel-mode image (data/pr_{pr}/workload/<hw>/Dockerfile)"),
             "needs-wheel-task": (False, f"no wheel in data/pr_{pr}/task.json: python3 gen/gen_tasks.py {pr} (wheel, source_files, test_files), then review workload.cmd")}
    ok, det = PIPES.get(pipe, (False, f"unknown pipeline '{pipe}'"))
    add(ok, "software", f"runner pipeline: {pipe}", det)
    for k in ("docker", "git", "python3", "nvidia_container_runtime", "hf_cli", "hf_token"):
        if not sw.get(k):
            continue
        ok = M[k]
        how, nosudo = INSTALL[k]
        if not ok and install and nosudo:
            rc, out = sh(how, 300)
            ok = rc == 0
            how = f"installed via `{how}`" if ok else f"install failed: {out[-120:]}"
        add(ok, "software", k, "" if ok else f"install: {how}")
    if sw.get("vllm_wheel"):
        add(True, "software", f"vllm=={sw['vllm_wheel']} wheel image (rendered from templates/Dockerfile.gpu-wheel)")
    add(M["github"], "software", "github.com reachable (the image shallow-fetches the PR's two commits)", "" if M["github"] else "github.com unreachable")
    for mdl in sw.get("models", []):
        cached = os.path.isdir(os.path.join(ROOT, ".weights", "hf", "hub", "models--" + mdl.replace("/", "--")))
        add(True, "software", f"model {mdl}", "cached" if cached else "downloads on first run")
    for ds in sw.get("datasets", []):
        add(True, "software", f"dataset {ds}", "downloads on first run")
    for v in sw.get("env", []):
        add(bool(os.environ.get(v)), "software", f"env {v}=1 (opt-in gate)", "" if os.environ.get(v) else f"export {v}=1 to allow")
    wl = (t.get("workload") or {}).get("cmd") or ""
    if "/task/bench.py" in wl or f"pr_{pr}/bench.py" in wl:
        have = os.path.exists(layout.bench_py(pr))
        add(have, "software", f"vendored bench script data/pr_{pr}/bench.py", "" if have else "missing: the workload cmd points at it but it was never written")
    wd = layout.workload_dir_for_run(pr)
    have_df = os.path.exists(os.path.join(wd, "Dockerfile"))
    add(have_df, "software", f"generated workload dir {layout.rel(wd)}", "" if have_df else f"run: python3 gen/gen_workloads.py {pr}")
    if sw.get("needs_server"):
        add(False, "software", "serving benchmark needs a running vllm server", "run_one.sh runs one command per arm; no server orchestration yet")
    return R


def main():
    args = sys.argv[1:]
    quiet, install = "--quiet" in args, "--install" in args
    prs = [int(a) for a in args if a.isdigit()]
    M = machine()
    bad = 0
    for pr in prs:
        R = check(pr, M, install)
        unmet = [r for r in R if not r[0]]
        bad += bool(unmet)
        if not quiet:
            why = "" if not unmet else " (" + ", ".join(sorted({r[1] for r in unmet})) + " missing)"
            print(f"PR {pr} [{M['hw_slug']}]: {'CAN reproduce' if not unmet else 'CANNOT reproduce here' + why}")
        for ok, kind, req, det in (unmet if quiet else R):
            print(f"  [{'ok' if ok else 'NO'}] {kind:8} {req}" + (f"  -- {det}" if det else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
