# Issue + Solution prompt — structured PR enrichment

Source-of-truth for `Dataset/rebuild_pr_md.py`. Rerun after editing:
`cd Dataset && FORCE=1 python3 rebuild_pr_md.py [field ...]`.

## Intent

Turn each MERGED vLLM PR into a structured dataset entry with a richly
elaborated **Issue** (7 fields) and a **Solution** (fix-in-words + author's
own words). One LLM call per PR. The pattern below IS the prompt — the model is
asked to fill exactly these fields.

## Inputs the model sees

- `title` — the PR title.
- `body` — author body **plus the first 3 PR comments** (comments often carry
  the benchmark numbers / reviewer-surfaced motivation; include them so the
  Issue fields can cite concrete scenarios).
- `diff` — the merged diff (authoritative: what actually changed). The Issue is
  derived from body AND diff (the diff reveals what was really wrong);
  `fix_in_words` is derived from the diff; `author_solution` from the body only.

## The pattern (the 9 fields the model must return)

`issue` — the problem, elaborated on 7 axes:
- **problem** — 2-4 sentences: the core inefficiency/bug, clear + technical.
- **example** — 1-3 sentences: a concrete scenario where it bites (model /
  shape / config numbers if the PR gives any).
- **use_case** — 1-2 sentences: the real serving/inference situation it arises in.
- **resources** — 1-2 sentences: the resource stressed (GPU compute, HBM/memory,
  PCIe/NVLink bandwidth, CPU, disk, kernel-launch/compile time, …).
- **type_of_users** — 1-2 sentences: who hits this (long-context, MoE, FP8,
  multi-GPU TP/PP, speculative decode, disaggregated P/D, specific hardware, …).
- **kind_of_inputs** — 1-2 sentences: the input characteristics that trigger it
  (sequence length, batch size, num requests, dtype, sliding window, prefix
  overlap, …).
- **mechanism** — 2-3 sentences: the specific code path / function / data
  structure that has the issue and WHY it is slow or wrong.

`fix_in_words` — 3-5 sentences: what the diff changes (before vs after) and the
effect achieved. Concrete mechanism, derived from the diff.

`author_solution` — 2-4 sentences: the author's OWN stated approach, condensed
faithfully from the description. If none given, exactly:
`(author gave no solution description)`.

## Outputs

Per PR, written to BOTH `Dataset/v1/<field>/prs/<area>/<n>.{md,json}` and
mirrored to `Knowledge/data/prs_v1/<field>/<area>/<n>.{md,json}`:
- **`.md`** — `# PR #<n>: <title>` + metadata bullets → `# Issue` (7 `##`
  subsections) → `# Solution` (`## Fix in words`, `## Author's solution`) →
  `## Ground-truth diff` (files changed + fenced diff).
- **`.json`** — the original PR json with an added `"issue_solution"` key holding
  the 9 structured fields, so downstream consumers get the pattern without
  re-parsing markdown.
- Semantic cache: `Knowledge/data/gt_solutions/<field>/<n>.sem.json`.

## Rules & idempotency

- Rebuild a PR only if its `.md` lacks `# Issue` OR env `FORCE=1`.
- The `.sem.json` cache is authoritative: if present it is reused and NO LLM call
  is made (free re-runs).
- Comments 1-3 are appended to the body before the call — the Issue fields may
  rely on them.
- Return STRICT JSON only. Model: `claude-opus-4-8` via LiteLLM proxy.
