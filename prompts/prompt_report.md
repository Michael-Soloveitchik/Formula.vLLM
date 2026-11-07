# Report prompt — report.md + stats.md generation

Source-of-truth for the reporting stage. `data/stats.md` is written by
`extract.py`; `Knowledge/report.md` by `report.py`. Rerun:
`python3 pipeline/extract.py && python3 pipeline/perf_scan.py && python3 pipeline/agenda.py && python3 pipeline/report.py`.

## Artifact map (names of all report / research / stats files)
| file | written by | contents |
|---|---|---|
| `data/index.jsonl` | extract.py | one record per merged PR (field, area, tags, excl_reason, files) — the spine every other stage reads |
| `data/stats.md` | extract.py | clustering stats: totals, exclusion-reason table, field × code-area matrix |
| `data/pr_<n>/pr.md` (was `data/prs/<field>/<area>/<pr>.md`) (+ `.json`) | downloader | the downloaded CONVERSATION (not code) |
| `data/prs/_excluded/<reason>/` | migrate.py | PRs parked after a recluster (survive re-runs) |
| `data/views/perf_gains.{jsonl,md}` | perf_scan.py | measured performance gains — see [prompt_perf.md](prompt_perf.md) |
| `data/views/agenda.{jsonl,md}` | agenda.py | performance vs enabling split — see [prompt_agenda.md](prompt_agenda.md) |
| `Knowledge/report.md` | report.py | top-level human report (progress + agenda + worked examples) |

## report.md sections
1. **Pipeline stages** checklist.
2. **Progress by field** — downloaded / classified per field, top areas, totals.
3. **Performance vs enabling agenda** — from `agenda.jsonl`: performance vs
   enabling totals, per-field table, enabling subcat breakdown.
4. **Worked examples** (every PR id is a clickable GitHub link to verify):
   - **3 quantitative** — 2 top %/× + 1 top absolute-unit from `perf_gains.jsonl`;
     each shows `improvement: <top_gain>`, a one-line `summary` (first clean
     sentence of the author `## Purpose`/Body), and `evidence` (the snippet).
   - **3 non-quantitative** — contributing, NOT in the measured set, a perf word
     in the author Purpose/Body scanned BEFORE `## Comments`, github-actions
     BOILER text skipped, titled-perf PRs preferred.
   - **3 filtered-out** — reason `manual_body_audit` or `arch_non_contrib`.

## Edge cases (that got us to the final version)
- **Counting downloaded conversations**: count only files matching `^\d+\.md$`.
  A PR whose dominant file is a top-level `README.md` produces an area FOLDER
  named `readme.md`; `rglob("*.md")` would otherwise count the folder and
  inflate the total (observed 5977 vs 5975).
- **Worked-example text hygiene**: strip HTML, collapse whitespace, and skip
  github-actions boilerplate ("improve the efficiency of the review process",
  fastcheck, slack, DCO) so examples quote the AUTHOR, not the bot.
- **Display names**: `decoding_tpot`→"decoding(TPOT)", `prefill_ttft`→
  "prefill(ttft)" in every table.
- **Determinism**: all stages are idempotent and read the committed artifacts,
  so the report is reproducible from a clean checkout by rerunning the chain.
