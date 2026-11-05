"""Tests for conf.py (YAML-subset parser + CLI) and the JSON data files."""
import json
import os
import subprocess
import sys

import conf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli(*args):
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "conf.py"), *args],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.rstrip("\n")


# ---- parser internals ------------------------------------------------------

def test_parse_scalar_types():
    assert conf._parse_scalar("true") is True
    assert conf._parse_scalar("false") is False
    assert conf._parse_scalar('"quoted"') == "quoted"
    assert conf._parse_scalar("5") == "5"
    assert conf._parse_scalar("[a, b, c]") == ["a", "b", "c"]
    assert conf._parse_scalar("[]") == []
    assert conf._parse_scalar("{k: v}") == {"k": "v"}


def test_load_nested():
    cfg = conf.load()
    assert cfg["run"]["default"] == "interesting"
    assert cfg["measure"]["repeats"] == "5"
    assert cfg["paths"]["vllm_git"] == "../../vllm"
    assert cfg["run"]["sets"]["cpu"] == ["21222", "21627"]


def test_merge_solver_block():
    solvers = conf._merge_solver_block(conf.load())
    enabled = [s for s in solvers if str(s.get("enabled")).lower() == "true"]
    assert len(enabled) == 1
    s = enabled[0]
    assert s["harness"] == "claude"
    assert s["model"] == "Fable-5.1"
    assert s["cmd"] == "claude --model=Fable-5.1"


# ---- CLI -------------------------------------------------------------------

def test_cli_solvers():
    assert _cli("solvers") == "claude\tFable-5.1\tclaude --model=Fable-5.1"


def test_cli_set():
    assert _cli("set", "interesting") == "21222 21627 6865 7117 8868"
    assert _cli("set", "all") == "all"
    assert _cli("set", "999") == "999"          # bare digit passthrough
    assert _cli("set", "bogus") == "bogus"      # unknown name passthrough


def test_cli_get():
    assert _cli("get", "run.default") == "interesting"
    assert _cli("get", "measure.repeats") == "5"
    assert _cli("get", "paths.vllm_git") == "../../vllm"


# ---- index.json ------------------------------------------------------------

def test_index_json():
    data = json.load(open(os.path.join(ROOT, "index.json")))
    assert isinstance(data, list)
    assert len(data) == 202
    prs = []
    for e in data:
        assert isinstance(e["pr"], int) and e["pr"] > 0
        for k in ("base_sha", "merge_sha", "title", "field"):
            assert e.get(k), f"pr {e['pr']} missing {k}"
        assert e["base_sha"] != e["merge_sha"]
        prs.append(e["pr"])
    assert len(prs) == len(set(prs)), "pr numbers not unique"
