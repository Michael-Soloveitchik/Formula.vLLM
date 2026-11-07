#!/usr/bin/env python3
"""Vendor the wheel-mode facts into data/pr_<n>/task.json for PRs that have a workload but no
`wheel` yet (run_one.sh refuses to run those): wheel = first PyPI release after merge_sha,
source_files / test_files from pr.diff, plus per-PR notes below.

    python3 gen/gen_tasks.py 21222 21281 ...      # fill missing fields (skips PRs that already have a wheel)
    python3 gen/gen_tasks.py --force 21222        # recompute wheel / source_files / test_files

Sources: task.json (date, workload), data/pr_<n>/pr.diff, .work/vllm_releases.json (PyPI wheel
upload dates). run.sh --reproduce --next calls this before gen_workloads.py.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gen import layout  # noqa: E402

# wheels whose torch pin has no sm_120 kernels but a cu128 build of the same torch exists
TORCH_FIX = {"0.10.0": "torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
             "0.10.1": "torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128"}
# per-PR notes / extras (read from the PR, not derivable)
NOTES = {
    6779:  {"extra_pip": "", "test_note": "perf-only PR, no test in the diff; v0 sampler, wheel 0.5.4 pins torch 2.4 cu121 (no sm_120 kernels) - the bench execs the source by file and never touches CUDA, so it still measures."},
    13365: {"extra_pip": "numba", "test_note": "perf-only PR (numba JIT n-gram matcher); wheel 0.7.3 pins torch 2.5.1 cu124 (no sm_120) - bench is CPU-only numba, unaffected."},
    21281: {"test_note": "perf-only follow-up of #21222 (same block_pool microbench); no test in the diff."},
    27673: {"test_note": "perf-only early return; no test in the diff; bench times remove_skipped_blocks on the no-op path."},
    27793: {"test_note": "correctness-only PR: the oracle test IS the workload (deterministic fake clock), no latency number."},
    29499: {"test_note": "EPLB numpy rewrite; tests in the diff cover algo + execute."},
    32975: {"extra_pip": "", "test_note": "perf-only detokenizer change; bench builds the detokenizer against the real gpt2 tokenizer (downloads into the shared HF cache)."},
    33593: {"test_note": "perf-only (list slice -> islice); no test in the diff."},
    39328: {"test_note": "perf-only functools.cache over inspect.getsource; no test in the diff."},
}


def changed_files(pr):
    p = layout.raw_diff(pr)
    if not os.path.exists(p):
        return [], []
    files = [m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", open(p, errors="replace").read(), re.M)]
    return ([x for x in files if x.startswith("vllm/") and x.endswith(".py")],
            [x for x in files if x.startswith("tests/") and x.endswith(".py")])


def pick_wheel(merge_date, rel):
    c = sorted((d, v) for v, d in rel.items() if d > merge_date)
    return c[0] if c else (None, None)


def vendor(pr, rel, force=False):
    t = layout.load_task(pr)
    if not t:
        return f"{pr}: no data/pr_{pr}/task.json"
    if not layout.has_workload(t):
        return f"{pr}: no workload.cmd in task.json, skip"
    if t.get("wheel") and not force:
        return f"{pr}: wheel {t['wheel']} already set, skip (--force to recompute)"
    src, tests = changed_files(pr)
    date, wheel = pick_wheel(t.get("date") or "", rel)
    t["source_files"], t["test_files"] = src, tests
    t["wheel"] = wheel
    t["wheel_note"] = f"first PyPI release after merge ({date} > {t.get('date')}); picked by gen/gen_tasks.py from PyPI upload dates."
    wl = t.setdefault("workload", {})
    cmd = wl.get("cmd") or ""
    if cmd.startswith(f"python3 tasks/pr_{pr}/bench.py") or cmd.startswith(f"python3 data/pr_{pr}/bench.py"):
        wl["cmd"] = "python3 /task/bench.py"       # /task = data/pr_<n> mounted in the image
    wl.setdefault("needs", "gpu")
    exps = t.get("experiments") or []
    wl.setdefault("unit", exps[0].get("unit") if exps else None)
    wl.setdefault("headline", exps[0]["name"] if exps else None)
    t.setdefault("experiments_note", "reported = numbers from the PR body (author's machine)")
    n = NOTES.get(pr, {})
    if n.get("test_note"):
        t["test_note"] = n["test_note"]
    if wheel in TORCH_FIX:
        t["torch_fix"] = TORCH_FIX[wheel]
    if n.get("extra_pip"):
        t["extra_pip"] = n["extra_pip"]
    layout.save_task(pr, t)
    return f"{pr}: wheel={wheel} src={len(src)} tests={len(tests)} cmd='{wl['cmd'][:60]}'" + (" torch_fix" if "torch_fix" in t else "")


def main():
    force = "--force" in sys.argv
    prs = [int(a) for a in sys.argv[1:] if a.isdigit()]
    try:
        rel = json.load(open(layout.releases_json()))
    except Exception:
        sys.exit(f"missing {layout.rel(layout.releases_json())} (PyPI upload dates {{version: date}}); cannot pick a wheel")
    for pr in prs:
        print(vendor(pr, rel, force))


if __name__ == "__main__":
    main()
