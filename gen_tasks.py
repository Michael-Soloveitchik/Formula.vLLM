#!/usr/bin/env python3
"""Write wheel-mode tasks/pr_<n>/metadata.json for pure-python PRs that have a
workload in workloads.json but no vendored task yet.

    python3 gen_tasks.py 21222 21281 ...      # write (skips PRs that already have metadata.json)
    python3 gen_tasks.py --force 21222        # overwrite

Sources: index.json (shas, title, field, date), data/prs/**/<pr>.diff (changed
source + test files), workloads.json (cmd, experiments, reported numbers),
.work/vllm_releases.json (PyPI wheel upload dates -> first release after merge).
run_one.sh then runs it like tasks/pr_37225: wheel image, .py-tree swap per arm.
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda p: json.load(open(os.path.join(HERE, p)))

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
    f = (glob.glob(os.path.join(HERE, "data", "prs", "**", f"{pr}.diff"), recursive=True) or [None])[0]
    if not f:
        return [], []
    files = [m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", open(f, errors="replace").read(), re.M)]
    src = [x for x in files if x.startswith("vllm/") and x.endswith(".py")]
    tests = [x for x in files if x.startswith("tests/") and x.endswith(".py")]
    return src, tests


def pick_wheel(merge_date, rel):
    c = sorted((d, v) for v, d in rel.items() if d > merge_date)
    return c[0] if c else (None, None)


def build(pr, idx, wl, rel):
    e = idx[pr]
    src, tests = changed_files(pr)
    date, wheel = pick_wheel(e["date"], rel)
    cmd = wl["cmd"]
    if cmd.startswith(f"python3 tasks/pr_{pr}/bench.py"):
        cmd = "python3 /task/bench.py"            # /task = tasks/pr_<n> mounted in the image
    exps = []
    for x in wl.get("experiments", []):
        exps.append({"name": x["name"], "unit": x.get("unit"), "reported": x.get("reported")})
    n = NOTES.get(pr, {})
    m = {
        "pr": pr, "title": e["title"], "field": e["field"], "date": e["date"],
        "hardware": e.get("hardware") or ["agnostic"],
        "url": e.get("pr_url") or f"https://github.com/vllm-project/vllm/pull/{pr}",
        "base_sha": e["base_sha"], "merge_sha": e["merge_sha"],
        "source_files": src, "test_files": tests,
        "wheel": wheel,
        "wheel_note": f"first PyPI release after merge ({date} > {e['date']}); picked by gen_tasks.py from PyPI upload dates.",
        "workload": {"kind": wl.get("workload_kind", "repo_benchmark"), "script": wl.get("script"),
                     "cmd": cmd, "unit": (exps[0]["unit"] if exps else None), "needs": "gpu",
                     "headline": exps[0]["name"] if exps else None,
                     "note": wl.get("notes", "")},
        "perf_gain": {"top_gain": e.get("reported_gain"), "gain_kind": e.get("gain_kind"), "gain_evidence": e.get("gain_evidence", "")[:400]},
        "needs": "gpu", "experiments": exps,
        "experiments_note": "reported = numbers from the PR body (author's machine)",
        "test_note": n.get("test_note", ""),
        "generated_by": "gen_tasks.py",
    }
    if wheel in TORCH_FIX:
        m["torch_fix"] = TORCH_FIX[wheel]
    if n.get("extra_pip"):
        m["extra_pip"] = n["extra_pip"]
    return m


def main():
    force = "--force" in sys.argv
    prs = [int(a) for a in sys.argv[1:] if a.isdigit()]
    idx = {x["pr"]: x for x in J("index.json")}
    W = J("workloads.json")
    rel = J(".work/vllm_releases.json")
    for pr in prs:
        d = os.path.join(HERE, "tasks", f"pr_{pr}")
        p = os.path.join(d, "metadata.json")
        if os.path.exists(p) and not force:
            print(f"{pr}: metadata.json exists, skip (--force to overwrite)"); continue
        if str(pr) not in W:
            print(f"{pr}: not in workloads.json, skip"); continue
        m = build(pr, idx, W[str(pr)], rel)
        os.makedirs(d, exist_ok=True)
        json.dump(m, open(p, "w"), indent=1, ensure_ascii=False); open(p, "a").write("\n")
        print(f"{pr}: wheel={m['wheel']} src={len(m['source_files'])} tests={len(m['test_files'])} cmd='{m['workload']['cmd'][:60]}'" + (" torch_fix" if "torch_fix" in m else ""))


main()
