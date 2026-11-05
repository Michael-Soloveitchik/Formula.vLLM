#!/usr/bin/env python3
"""Minimal config reader for config.yaml — no PyYAML needed on the target box.

Supports exactly the subset config.yaml uses: nested maps, inline {a: b} maps,
inline [a, b] lists, and a `solvers:` list of `- key: val` blocks. Prints shell-
friendly lines so run.sh can stay plain bash.

Usage:
  conf.py solvers            # enabled solvers: "<harness>\t<model>\t<cmd>" per line
  conf.py set <name>         # resolve run.sets.<name> (or 'all'/pr-id passthrough)
  conf.py get a.b.c          # scalar lookup by dotted path
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(HERE, "config.yaml")


def _strip(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        v = v[1:-1]
    return v


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def load():
    """Parse into nested dict. `solvers` becomes a list of dicts."""
    with open(CONF) as f:
        raw = [ln.rstrip("\n") for ln in f]
    lines = []
    for ln in raw:
        s = ln.split(" #", 1)[0] if " #" in ln else ln
        if s.strip() and not s.lstrip().startswith("#"):
            lines.append(s)

    root: dict = {}
    # stack of (indent, container)
    stack = [(-1, root)]
    i = 0
    while i < len(lines):
        ln = lines[i]
        ind = _indent(ln)
        body = ln.strip()
        while stack and ind <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if body.startswith("- "):
            # list item; parent must be a list
            item = _parse_item(body[2:])
            if not isinstance(parent, list):
                i += 1
                continue
            parent.append(item)
            if isinstance(item, dict):
                stack.append((ind, item))
            i += 1
            continue

        if ":" not in body:
            i += 1
            continue
        key, _, val = body.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            # container: peek next line to decide list vs map
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.strip().startswith("- ") and _indent(nxt) > ind:
                child: object = []
            else:
                child = {}
            if isinstance(parent, dict):
                parent[key] = child
            stack.append((ind, child))
        else:
            if isinstance(parent, dict):
                parent[key] = _parse_scalar(val)
        i += 1
    return root


def _parse_item(s: str):
    if ":" in s:
        d: dict = {}
        key, _, val = s.partition(":")
        d[key.strip()] = _parse_scalar(val.strip()) if val.strip() else ""
        return d
    return _parse_scalar(s)


def _parse_scalar(v: str):
    v = v.strip()
    if v.startswith("{") and v.endswith("}"):
        out = {}
        for part in v[1:-1].split(","):
            if ":" in part:
                k, _, x = part.partition(":")
                out[k.strip()] = _parse_scalar(x)
        return out
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_strip(x) for x in inner.split(",")] if inner else []
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    return _strip(v)


def _merge_solver_block(cfg):
    """`- harness: x` then indented `model:`... lines parse as separate dicts;
    stitch consecutive dicts into one solver until the next `harness` key."""
    out = []
    cur = None
    for item in cfg.get("solvers", []) or []:
        if not isinstance(item, dict):
            continue
        if "harness" in item:
            if cur:
                out.append(cur)
            cur = dict(item)
        elif cur is not None:
            cur.update(item)
    if cur:
        out.append(cur)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: conf.py solvers|set <name>|get <path>")
    cfg = load()
    cmd = sys.argv[1]

    if cmd == "solvers":
        for s in _merge_solver_block(cfg):
            if str(s.get("enabled")).lower() != "true":
                continue
            guided = "1" if str(s.get("guided")).lower() == "true" else "0"
            print(f"{s.get('harness','')}\t{s.get('model','')}\t{s.get('cmd','')}\t{guided}")
        return

    if cmd == "set":
        name = sys.argv[2]
        if name == "all":
            print("all")
            return
        if name.isdigit():
            print(name)
            return
        sets = (cfg.get("run", {}) or {}).get("sets", {}) or {}
        val = sets.get(name)
        if isinstance(val, list):
            print(" ".join(str(x) for x in val))
        elif val:
            print(val)
        else:
            print(name)  # passthrough (bare pr id)
        return

    if cmd == "get":
        node = cfg
        for key in sys.argv[2].split("."):
            if isinstance(node, dict):
                node = node.get(key)
            else:
                node = None
                break
        print("" if node is None else node)
        return

    sys.exit(f"unknown: {cmd}")


if __name__ == "__main__":
    main()
