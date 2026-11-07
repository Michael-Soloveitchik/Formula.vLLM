# Agenda prompt — performance vs enabling split

Source-of-truth for `pipeline/agenda.py`. Rerun after editing:
`python3 pipeline/agenda.py`.

## Intent
Split the contributing PRs into two agendas the reader cares about:
- **performance** — the author reports a quantified speed / latency / throughput
  / memory gain.
- **enabling** — real value, but NOT a speedup: correctness / "it now works" /
  doesn't-OOM / model loading / metrics / NUMA / refactor / feature-support.

## The split line (set by a body-read audit — do not loosen casually)
The line is a **measured benchmark number** (membership in
`data/views/perf_gains.jsonl`, see [prompt_perf.md](prompt_perf.md)). A subagent
read PR bodies on both sides:
- No-number side audited **12/12** truly non-performance → the enabling bucket
  is clean. (An **accuracy / MTEB quality** win is NOT performance → stays
  enabling.)
- A bare prose `faster/optimize` keyword was only **~33%** precise (dir names
  like `.../optimization/`, incidental "throughput" in a bugfix, "re-enable
  cudagraph", benchmark tooling).
→ Therefore **prose alone never promotes a PR to performance.** It is kept only
as an advisory `perf_intent` flag on enabling PRs.

## Inputs
- `data/index.jsonl` (contributing PRs) + `data/views/perf_gains.jsonl` (the
  measured set) + `data/pr_<n>/pr.md` (was `data/prs/<field>/<area>/<pr>.md`) (author text).

## Outputs (research / stats files)
- `data/views/agenda.jsonl` — one record per contributing PR:
  `{pr, field, area, agenda, signal, subcat, perf_intent, title}`.
  `agenda` ∈ {performance, enabling}; `signal` = `measured` | `no_number`.
- `data/views/agenda.md` — totals, per-field performance/enabling table, and
  the enabling subcategory breakdown.
- Also surfaced in `Knowledge/report.md` (see [prompt_report.md](prompt_report.md)).

## Rules & edge cases
- Read ONLY the AUTHOR text: the slice of each `.md` between `## Body` and
  `## Comments` (plus the `# PR #N: <title>` line). Never classify on bot /
  reviewer comments.
- `performance` iff `pr ∈ perf_gains`; else `enabling`.
- `perf_intent` (advisory) uses a TIGHTENED perf-claim regex. Key fix: stems
  must take `\w*`, not a trailing `\b` — `optimi\w+`, `latenc\w*`,
  `accelerat\w+`, `throughput\w*` — else "optimizations"/"latency" silently fail
  (a word char follows the stem, so there is no word boundary). A `[perf]` /
  `[performance]` title tag also sets intent. Boilerplate (review-process /
  DCO / fastcheck / slack) is excluded.
- Enabling `subcat` = FIRST keyword hit, in this ORDER (order is the tie-break):
  model_loading, correctness, oom_memory, metrics, numa_topology, refactor,
  enable_support, else `other`.
- `agenda.py` is a VIEW: it does **not** modify `index.jsonl`.

## Learnings
- v1: defined performance as measured OR prose-claim → prose over-captured.
- v2: subagent audit (12 per side) → prose ~33% precise, no-number side clean
  (accuracy ≠ performance). Demoted prose to `perf_intent` flag; split line =
  measured number. Result: performance=376, enabling=5599 (885 w/ unmeasured
  intent).
