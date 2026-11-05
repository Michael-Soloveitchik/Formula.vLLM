# Performance-scan prompt — quantified-gain extraction

Source-of-truth for `pipeline/perf_scan.py`. Rerun after editing:
`python3 pipeline/perf_scan.py`.

## Intent
From each downloaded contributing PR conversation, extract the **single best
QUANTIFIED performance gain** the author reports. This set is the trustworthy
"performance" bucket; a PR with no parseable number is left to the agenda split
(see [prompt_agenda.md](prompt_agenda.md)).

## Inputs
- `data/index.jsonl` — contributing PRs (field set).
- `data/prs/<field>/<area>/<pr>.md` — the conversation (title + body + comments).

## Outputs (research / stats files)
- `data/views/perf_gains.jsonl` — one record per PR with a measured gain:
  `{pr, field, area, kind, title, top_gain, evidence}`. `kind` = `pct`
  (%/× speedup) | `unit` (absolute before→after or Before/After table).
- `data/views/perf_gains.md` — same, grouped by field, sorted by magnitude.

## Extraction rules (edge cases that got us to the final version)
### %/× speedups (`kind=pct`)
- Sentence-split the whole text; a gain token is `N%`, `Nx`, `N×` **not
  followed by an alnum** (so `SM80`, `8xH100`, `cuda12.1` are not gains).
- A gain must sit in a sentence with a PERF context word (improv/throughput/
  latenc/speed-up/faster/ttft/tpot/e2e/tok/s/reduc/optimi/boost/lower/higher/
  less memory/saving).
- **Mask hardware BEFORE parsing**, in this order: GPU-COUNT notation first
  (`8xB200`, `4×GB200`, `2 x H100`, `8xMI300`) then bare HW tokens (`MI355X`,
  `H100`, `SM90`, `cuda 12.1`). Order matters: mask counts first or the leading
  `8x`/`4×` survives and reads as a speedup.
- `NEG` rejects: tqdm/progress bars, percentile labels, accuracy / success-rate
  / gsm8k / pass@ / correctness-rate / acceptance / hit-rate, flops / tfops /
  bandwidth / mem-bw, `slower` / regress / ↑, hypotheticals ("if we", "would
  be"), spec-decode metric lists.
- **Round-% rule**: bare round values (10/25/50/75/90/99/100%) are normally
  noise and dropped — UNLESS a WIN word (gain/faster/speedup/improv/boost/
  reduc/saving/cut/lower latency/higher throughput/less memory) is in the same
  sentence ("~10% performance gain", "25% faster"). This recovers real round-%
  claims without re-admitting percentile noise.

### Absolute-unit before→after (`kind=unit`)
- Patterns: `A unit -> B unit`, `from A to B`, and the **relaxed arrow**
  `A -> B unit` (unit on the 2nd operand only, inherited by the 1st, e.g.
  `7.87 -> 8.37 req/s`).
- Normalize spaced throughput units first: `req / sec`→`req/s`, `tok / s`→
  `tok/s`.
- Compute improvement %: **lower-is-better** for time (ms/µs/s) and memory
  (GiB/MiB/GB/MB), **higher-is-better** for throughput (tok/s, req/s, qps,
  otps). Keep only `1% ≤ imp ≤ 100000%`. Memory deltas also require a PERF word.

### Before/After markdown tables
- Detect a pipe table; find a row whose first cell matches before/baseline/main/
  without/old and one matching after/with/this-pr/new/optimiz. Per column, infer
  direction from the header (throughput/tok-s/qps→higher; ttft/tpot/latency/ms→
  lower; mem/gib/vram→lower) and compute %.

### Known limits / manual overrides
- `PERF_FP` — body-read confirmed false positives (GPU-count/PTX misread, or
  measured *parity* not a gain) are skipped.
- Multi-block `key: value` benchmark dumps (two separate code blocks of
  `tok/s:` lines, no inline arrow or labelled table, e.g. #32770) are NOT
  parsed — accepted miss (parsing them generically invites false positives).

## Learnings
- v1: title+body %/× only → 337.
- v2: add absolute-unit arrows + Before/After tables → 355.
- v3: HW + GPU-count masking; `PERF_FP` skip set (killed "8×B200" read as 8×).
- v4: round-%-with-WIN rescue + relaxed arrow + spaced-unit normalize → 376.
