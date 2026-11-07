#!/usr/bin/env python3
"""Machine snapshot for a benchmark run (who else is on the box, what the GPU is doing).

    python3 machine.py snapshot          # JSON to stdout
    python3 machine.py snapshot -o f.json
    python3 machine.py slug              # hardware slug: rtx5070ti | h100-80gb | cpu ...

Every field is best-effort: any probe that fails yields null, the script itself
never exits non-zero. Intended to be stored next to data/pr_<n>/results.<hw>.json so an
absolute-latency gap vs the PR author can be attributed (hardware / contention /
clocks) after the fact. See BASELINE_GAP.md.

The hardware slug names the machine class a result / workload belongs to
(data/pr_<n>/results.<slug>.json, data/pr_<n>/workload/<slug>/, task.json:hardware_overrides).
HW_SLUG=<slug> in the environment overrides the probe (layout.py honours it too).
"""
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time


_SLUG_DROP = {"nvidia", "geforce", "tesla", "quadro", "amd", "radeon", "instinct", "graphics", "gpu", ""}


def slug_from_name(name):
    """'NVIDIA GeForce RTX 5070 Ti' -> rtx5070ti, 'NVIDIA H100 80GB HBM3' -> h100-80gb,
    'NVIDIA A100-SXM4-80GB' -> a100sxm4-80gb, '' -> cpu. Lowercase, vendor words and HBM
    generation dropped, model tokens squeezed, the memory size kept as a '-<n>gb' suffix."""
    if not name or not name.strip():
        return "cpu"
    toks = [t for t in re.split(r"[\s\-_/]+", name.lower()) if t not in _SLUG_DROP and not t.startswith("hbm")]
    mem = [t for t in toks if re.fullmatch(r"\d+gb", t)]
    rest = [t for t in toks if t not in mem]
    s = "".join(rest) + ("-" + mem[0] if mem else "")
    s = re.sub(r"[^a-z0-9-]", "", s).strip("-")
    return s or "gpu"


def slug():
    """Hardware slug of THIS machine (first GPU per nvidia-smi; 'cpu' when none)."""
    env = os.environ.get("HW_SLUG")
    if env:
        return re.sub(r"[^a-z0-9-]", "", env.lower()) or "cpu"
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    name = out.strip().splitlines()[0] if out and out.strip() else ""
    return slug_from_name(name)


def _run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else None
    except Exception:
        return None


def _num(s):
    try:
        return float(s.strip().split()[0])
    except Exception:
        return None


def _smi_query(fields):
    out = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    vals = [v.strip() for v in out.strip().splitlines()[0].split(",")]
    return dict(zip(fields, vals)) if len(vals) == len(fields) else None


def gpu():
    g = {"name": None, "driver": None, "cuda": None, "sm_mhz": None, "sm_max_mhz": None,
         "mem_mhz": None, "mem_max_mhz": None, "vram_total_mib": None, "vram_used_mib": None,
         "util_pct": None, "power_w": None, "power_limit_w": None, "temp_c": None, "pstate": None,
         "throttle_reasons": None, "processes": None}
    try:
        q = _smi_query(["name", "driver_version", "clocks.sm", "clocks.max.sm", "clocks.mem",
                        "clocks.max.mem", "memory.total", "memory.used", "utilization.gpu",
                        "power.draw", "power.limit", "temperature.gpu", "pstate",
                        "clocks_event_reasons.active"])
        if q is None:  # older nvidia-smi spells the last field differently
            q = _smi_query(["name", "driver_version", "clocks.sm", "clocks.max.sm", "clocks.mem",
                            "clocks.max.mem", "memory.total", "memory.used", "utilization.gpu",
                            "power.draw", "power.limit", "temperature.gpu", "pstate",
                            "clocks_throttle_reasons.active"])
        if q:
            vals = list(q.values())
            g.update(name=vals[0], driver=vals[1], sm_mhz=_num(vals[2]), sm_max_mhz=_num(vals[3]),
                     mem_mhz=_num(vals[4]), mem_max_mhz=_num(vals[5]), vram_total_mib=_num(vals[6]),
                     vram_used_mib=_num(vals[7]), util_pct=_num(vals[8]), power_w=_num(vals[9]),
                     power_limit_w=_num(vals[10]), temp_c=_num(vals[11]), pstate=vals[12],
                     throttle_reasons=vals[13])
    except Exception:
        pass
    try:
        out = _run(["nvidia-smi"])
        if out:
            for ln in out.splitlines():
                if "CUDA Version" in ln:
                    g["cuda"] = ln.split("CUDA Version:")[1].split("|")[0].strip()
                    break
    except Exception:
        pass
    try:
        out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"])
        procs = []
        if out is not None:
            for ln in out.strip().splitlines():
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                pid = int(parts[0])
                name = parts[1]
                if name in ("[Not Found]", "", "N/A"):  # WSL2 hides names; read /proc instead
                    try:
                        name = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
                    except Exception:
                        pass
                procs.append({"pid": pid, "name": name, "mem_mib": _num(parts[2])})
            g["processes"] = procs
    except Exception:
        pass
    return g


def cpu():
    c = {"model": None, "nproc": None, "load1": None, "load5": None, "load15": None,
         "top_cpu_procs": None}
    try:
        c["nproc"] = os.cpu_count()
    except Exception:
        pass
    try:
        c["load1"], c["load5"], c["load15"] = os.getloadavg()
    except Exception:
        pass
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                c["model"] = ln.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    try:
        out = _run(["ps", "-eo", "pid,pcpu,comm", "--sort=-pcpu"])
        if out:
            rows = []
            for ln in out.strip().splitlines()[1:6]:
                parts = ln.split(None, 2)
                if len(parts) == 3 and _num(parts[1]) is not None and _num(parts[1]) > 5:
                    rows.append({"pid": int(parts[0]), "pcpu": _num(parts[1]), "comm": parts[2]})
            c["top_cpu_procs"] = rows
    except Exception:
        pass
    return c


def ram():
    r = {"total_gb": None, "free_gb": None, "available_gb": None}
    try:
        m = {}
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            m[k] = _num(v)
        kb = 1024.0 * 1024.0
        r["total_gb"] = round(m["MemTotal"] / kb, 1)
        r["free_gb"] = round(m["MemFree"] / kb, 1)
        r["available_gb"] = round(m["MemAvailable"] / kb, 1)
    except Exception:
        pass
    return r


def docker():
    d = {"version": None, "engine": None}
    try:
        eng = os.environ.get("CONTAINER_ENGINE", "docker")
        d["engine"] = eng
        if shutil.which(eng):
            out = _run([eng, "--version"])
            d["version"] = out.strip() if out else None
    except Exception:
        pass
    return d


def host():
    h = {"hostname": None, "kernel": None, "wsl2": None, "os": None}
    try:
        h["hostname"] = platform.node()
        h["kernel"] = platform.release()
        h["wsl2"] = "microsoft" in platform.release().lower() or "wsl" in platform.release().lower()
    except Exception:
        pass
    try:
        for ln in open("/etc/os-release"):
            if ln.startswith("PRETTY_NAME="):
                h["os"] = ln.split("=", 1)[1].strip().strip('"')
                break
    except Exception:
        pass
    return h


def snapshot():
    s = {"ts": None, "host": None, "gpu": None, "cpu": None, "ram": None, "docker": None}
    try:
        s["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception:
        pass
    for k, fn in (("host", host), ("gpu", gpu), ("cpu", cpu), ("ram", ram), ("docker", docker)):
        try:
            s[k] = fn()
        except Exception:
            s[k] = None
    return s


def main(argv):
    try:
        cmd = argv[1] if len(argv) > 1 else "snapshot"
        if cmd == "slug":
            print(slug())
            return 0
        if cmd != "snapshot":
            print(__doc__)
            return 0
        snap = snapshot()
        txt = json.dumps(snap, indent=1)
        if "-o" in argv:
            try:
                with open(argv[argv.index("-o") + 1], "w") as f:
                    f.write(txt + "\n")
            except Exception:
                pass
        print(txt)
    except Exception as e:  # never fail
        print(json.dumps({"ts": None, "error": str(e)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
