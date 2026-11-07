#!/usr/bin/env python3
"""Lint: the words "base" / "gold" must not appear in human-facing text.

Say "baseline" and "PR's solution" instead. Scope: user-facing surfaces (README, MIGRATION.md, runner messages, report,
dashboard, the generated data/pr_<n>/problem.md and WORKLOAD.md, notes.md); solver/benchmark code (bench.py), the raw PR
material (pr.md, pr.diff, solution.md — the author's own words), work/ + logs/ scratch, templates/, prompts/ and tests/
are internal and skipped.
Allowed (internal identifiers only):
JSON/dict keys ("base", 'gold', .base, .gold, arms.base), base_sha, base_*/gold_*
names, /base and /gold container paths, {base,gold} path sets, SHA/ARG names.

    python3 tools/check_terms.py          # exit 1 + list of offenders
"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
SKIP_DIRS = {".git", ".work", "results", ".weights", "node_modules", "__pycache__", "work", "logs", "tests", "templates", }
EXTS = (".py", ".sh", ".md", ".yaml", ".yml", ".html", ".txt")
SKIP_FILES = {"check_terms.py", "numeric_solution.md", "index.json", "index.jsonl",
              "pr.md", "pr.diff", "solution.md", "bench.py"}   # numeric_solution.md defines the rule itself; pr.md/solution.md = the author's words

ALLOWED = [
    r"""["'`]?(?:arms\.)?\b(?:base|gold)\b["'`]\s*[:\]\)]""",   # "base": / ['gold'] / `arms.base`
    r"""["'`](?:base|gold)["'`]""",                                  # quoted internal key
    r"""\b(?:arms|a|e\.arms|res\.arms|L)\.(?:base|gold)\b""",       # attribute access
    r"""\{base,\s*gold\}""", r"""/(?:base|gold)\b""", r"""\b(?:base|gold)_\w+""", r"""\w+_(?:base|gold)\b""",
    r"""\$\{?(?:BASE|GOLD)\w*""", r"""\b(?:BASE|GOLD)_\w+""", r"""\bGOLD\b|\bBASE\b""",  # shell/ARG vars
    r"""rep\.base\b|\.base_us|\.base\b""", r"""\bbase\s*=\s*\w""",       # code assignment / reported.base
    r"""\b(?:base|gold)\s*[,)\]]\s*(?:=|in\b)""",
    # internal arm keys used as code tokens (shell args, loop vars, yaml/JS keys, docker tags)
    r"""\bin base gold\b""", r"""= base \]""", r"""\b(?:run_arm|run_workload) (?:base|gold)\b""", r'ARMS="base gold"',
    r"""^\s*(?:base|gold): \{""", r"""-base-ubuntu""", r"""\{base[:,]""", r"""\{(?:delta|pct),\s*base\?\}""", r"""authorBase=base""",
    r'":\s*base,',
]
ALLOWED.append(r"""workload (?:base|gold) run""")
ALLOWED.append(r"""\bbase class\b|\*Base\b|per merged PR""")   # plain English / data-spine wording
ALLOWED.append(r"""run_workloads_abba base .* gold |P8, SM 180""")
ALLOWED.append(r"""\^SKIP\|base=\|failed rc""")   # grep pattern in queue.sh
ALLOWED.append(r"""[\w.]+-base\b""")             # model ids (e5-base, multilingual-e5-base)
ALLOWED = [re.compile(p) for p in ALLOWED]
QUOTE = re.compile(r"^\s*>")   # markdown blockquote = the PR author's own words (problem.md, WORKLOAD.md): not our prose
WORD = re.compile(r"\b(?:base|gold|Base|Gold)\b|[Ee]xpert(?:'s)? (?:solution|patch)|merged PR\b")


def scan(path):
    out = []
    for n, ln in enumerate(open(path, errors="ignore"), 1):
        if QUOTE.match(ln):
            continue
        t = ln
        for a in ALLOWED:
            t = a.sub(" ", t)
        m = WORD.search(t)
        if m:
            out.append((n, ln.rstrip()[:140]))
    return out


bad = 0
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for f in files:
        if f in SKIP_FILES or not f.endswith(EXTS):
            continue
        p = os.path.join(root, f)
        for n, ln in scan(p):
            bad += 1
            print(f"{os.path.relpath(p, HERE)}:{n}: {ln}")
print(f"{bad} offending line(s)" if bad else "OK: no base/gold in human-facing text")
sys.exit(1 if bad else 0)
