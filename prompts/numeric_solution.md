# Numeric solution prompt — Formula.vLLM benchmark numbers (v2)

Defines the NUMERIC contract shared by two artifacts:

1. `data/pr_<n>/results.<hw-slug>.json` — the per-PR, per-hardware result the dashboard live-reads (via `results/manifest.json`).
2. `Knowledge/benchmark/web/index.html` — the dashboard that renders those numbers into
   one table per experiment (`expTable`).

Everything below is BINDING on both. When you add or edit a PR result, the numbers,
units, and row labels MUST follow this file. The renderer is the executable form of these
rules; this file is the spec it implements.

For the benchmark *directory layout*, per-PR `workload/Dockerfile` design, weight-download
scripts, and how `run.sh` drives the isolated per-task images, see the sibling spec
`benchmark.md`. This file governs the NUMBERS; `benchmark.md` governs the HARNESS that
produces them.

---

## 1. The vocabulary — two axes: WHO and WHICH-CODE

Every number in the table is defined by two independent axes:

- **WHO produced it**
  - **measured** (blue) = WE ran it, in docker, on the actual checked-out tree.
  - **reported** (orange) = the PR AUTHOR published it in the PR body/comments.
- **WHICH code it is**
  - **baseline** = the code BEFORE the PR (`base_sha`, the old/unoptimized version).
  - **PR's author solution** = the code AFTER the PR (`merge_sha`, the optimized version).

Cross the two axes → exactly four row labels, GROUPED BY WHICH-CODE (baselines together,
solutions together); within each pair the REPORTED row comes first, MEASURED second:

```
1. baseline (reported by PR's author)            <- orange · author's OLD-code number, or — if absent
2. baseline (measured by us)                     <- blue   · our run of base_sha · 1.00x REFERENCE
3. PR's author solution (reported by PR's author)<- orange · author's NEW-code number, or — if absent
4. PR's author solution (measured by us)         <- blue   · our run of merge_sha
5.. each solver harness arm                      <- claude / codex / +solution, or "pending"
```

The two BASELINE rows sit together (rows 1-2), the two SOLUTION rows together (rows 3-4);
in each pair the author's REPORTED claim is on top, OUR MEASURED run directly below it so
they read side by side. `baseline (measured by us)` is the 1.00× reference for every
`×`/`Δ` column — keyed off `arms.base`, not off render order.

Bracket suffix is CONSISTENT per color: every orange row ends `(reported by PR's author)`,
every blue row ends `(measured by us)`. Never vary the wording row-to-row.

Colors (CSS in `index.html`): measured rows carry a BLUE left-border + `.tag-m` blue tag;
reported rows carry an ORANGE left-border + `.tag-r` orange tag. Use the bracket tags
inside the row label, never plain text.

FORBIDDEN in any rendered/user-facing string: the words **`base`**, **`gold`**,
**`patch`**. Those are INTERNAL JSON keys only (`arms.base`, `arms.gold`, `reported.patch`).
Likewise never label one of OUR runs "gold" — both our runs are
"measured", distinguished only by baseline vs PR's author solution.

`baseline (measured by us)` is the 1.00× reference for EVERY `×` and `Δ` column.

JSON → row mapping (render order = this top-to-bottom):

| JSON source | Row label | Color |
|---|---|---|
| `reported.base` (or — if null) | `baseline (reported by PR's author)` | orange |
| `arms.base.mean` | `baseline (measured by us)` · 1.00× ref | blue |
| `reported.patch` / reconstructed (or — if none) | `PR's author solution (reported by PR's author)` | orange |
| `arms.gold.mean` | `PR's author solution (measured by us)` | blue |
| `arms.<harness>.mean` | harness key verbatim | — |

---

## 2. Units — every number in its own unit, no cross-scale mixing

- Each experiment has ONE `unit` string (`ms/op`, `us/tok`, `s`, `it/s`, `s/call`, `ms`).
  EVERY value in that table is in THAT unit. Never mix two units in one table.
- Report HONESTLY in the PR's own unit. Never convert a ratio into a fake absolute, never
  paste a speedup ratio into an absolute cell (the 32975 unit-bug: `1.628` the *ratio*
  pasted where `0.03936 us/tok` the *value* belongs).
- Direction of "better":
  - throughput units (`/s`, `it/s`, `tok/s`, `req/s`) → HIGHER better.
  - time units (`ms`, `s`, `us/tok`, `s/call`, `ms/op`) → LOWER better.
  `higherBetter(unit)` encodes this; keep units parseable.
- There are now TWO baselines and therefore TWO metric column-pairs:
  - `Δ vs reported` / `× vs reported` — anchored on the AUTHOR's reported baseline (orange).
  - `Δ vs measured` / `× vs measured` — anchored on OUR measured baseline (blue, 1.00× ref).
  Direction-aware (`>1` = faster, `+` = better) in both pairs.
- DIAGONAL FILL — each row fills ONLY its own color's pair; the off-color pair renders `—`:
  - ORANGE reported rows (`gcellsRep`) fill the `vs reported` pair; `vs measured` = `—`.
  - BLUE measured rows (`gcellsMeas`) fill the `vs measured` pair; `vs reported` = `—`.
  - So `baseline (reported)` is `1.00×` in the reported pair (measured pair blank), and
    `baseline (measured)` is `1.00×` in the measured pair (reported pair blank). A row is
    NEVER compared against the opposite color's baseline — that cross-comparison was the
    confusing artifact we removed. Reading DOWN a colored pair = that author's own story.

---

## 3. The three legal shapes of `reported` (what the author published)

### 3a. Absolute pair — author gave BOTH old and new numbers
```json
"reported": { "base": 0.15, "patch": 0.008 }
```
- `baseline (reported by PR's author)` shows `0.15`.
- `PR's author solution (reported)` shows `0.008`, gain referenced to the author's own
  `0.15` (NOT our baseline).
- PRs: 13365, 21222, 21281, 32975, 33593, 39328, 6779(micro).

### 3b. Ratio / percent only — a speedup, NO absolute baseline
```json
"reported": { "pct": 0.225 }
```
- `baseline (reported by PR's author)` renders `—` + source note "author gave a ratio/% —
  using our baseline as reference".
- `PR's author solution (reported)` is reconstructed from OUR baseline: `our_base*(1±pct)`.
- PR: 29499.

### 3c. Delta only — an improvement magnitude in the unit, no baseline
```json
"reported": { "delta": 4.4 }
```
- Same as 3b: author-baseline `—`; solution reconstructed `our_base ± delta`.

### 3d. No numbers at all
```json
"reported": { "base": null, "patch": null }
```
- Both orange rows `—`, source "no number in PR". Still a legal 202 entry if a qualitative
  claim exists (capture it in `reported_gain`).

### 3e. No experiments at all (correctness-only / feature PR)
```json
"experiments": [], "note": "...why there is no timing experiment..."
```
- PR 27793 is an OBSERVABILITY feature; its "0.1%" is an OVERHEAD ceiling, not a gain.
  Set `reported_gain` to `"N/A (...)"`, keep experiments empty, explain in `note`.

---

## 4. Scale / comparability honesty — the most important rule

Many PRs benchmark a DIFFERENT thing than our microbench runs (E2E serving vs isolated
fn, cumulative-over-a-run vs per-call, GPU-transfer-bound vs CPU-only, one harness vs
another). When the author's number and ours are not 1:1-comparable you MUST flag it:

```json
"reported": {
  "base": 54.3, "patch": 1.9,
  "comparable": false,
  "scope": "<what the author's number actually measured>",
  "scale_short": "<ONE short line: the gap + 'TO MAKE 1:1: <fix>'>",
  "scale_note": "<full detail; shown small under the short line>"
}
```

- `comparable: false` ⇒ direction agrees but absolute magnitude / ratio is NOT 1:1.
- `scope` ⇒ one line stating exactly what the author's number covered.
- `scale_short` ⇒ the headline the reader sees FIRST: a FEW very short clear sentences —
  what diverges + an explicit `TO MAKE 1:1: <concrete workload/test change>`. Keep it tight.
- `scale_note` ⇒ full detail, rendered small/secondary under the short line.
- ALWAYS PREFER 1:1. The goal is to make our workload and tests reproduce the author's
  ENCLOSING SCOPE so the ratios converge (e.g. 6779: widen from the pad sub-op to the full
  `from_sampling_metadata` build). Only mark `comparable:false` when 1:1 is impossible here
  (GPU-only path, reverted code, E2E serving run we cannot stand up on an arm Mac). When you
  do mark it, `scale_short` MUST name the exact change that WOULD make it 1:1.
- Keep every NOT-1:1 explanation SHORT: a few clear sentences, not a paragraph. Detail lives
  in `scale_note`; the banner headline is `scale_short`.
- RENDER REQUIREMENT: `index.html` draws a banner UNDER every experiment table whenever any
  of `comparable:false` / `workload_invalid` / `scope` / `scale_short` / `scale_note` is set
  (orange box tagged `NOT 1:1`, or RED `WORKLOAD INVALID`). The headline `scale_short` and
  the detail MUST be visible — a divergence that lives only in JSON is one nobody sees.
- `workload_invalid: true` ⇒ stronger than `comparable:false`: OUR measured ratio is
  MEANINGLESS because the bench never exercises the changed branch (21281). State the
  exact setup fix needed.

Taxonomy (all seen in the 9 CPU PRs):

| Case | What differs | Example |
|---|---|---|
| UNIT-BUG | ratio pasted as an absolute value | 32975 (`1.628` in a `us/tok` cell) |
| SCALE-MISMATCH | E2E vs isolated; cumulative vs per-call | 13365, 6779 |
| HARNESS-MISMATCH | two different author harnesses conflated | 21222 (engine-script 19x vs benchmark_block_pool ~2x) |
| DIFFERENT-OP | author timed a different primitive than the call site | 33593 (materialization vs short-circuit membership) |
| WRONG-ATTRIBUTION | PR doesn't touch the fn we named; real change is elsewhere | 6779 (array.array in SequenceData, NOT make_tensor_with_pad) |
| SIGN-FLIP | our arm slower where author's is faster | 33593 absent, 29499 move_*buffer |
| GPU-BOUND | real win is GPU-transfer/sync elision, not CPU-reproducible | 29499 move_to/from_buffer |
| WORKLOAD-INVALID | bench never hits the changed branch | 21281 (block_hash=None early-out) |
| TWO-CLAIMS | author published two different metrics; pick the headline | 29499 (27% rearrange E2E vs 20-25% per primitive) |
| NO-NUMBERS / REVERTED | author gave none, or benchmarked reverted code | 27793; 32975 min_tokens loop |

---

## 5. Anomaly detection — run this on EVERY new PR before trusting the row

Compute these and INVESTIGATE (open the original PR) whenever any trips:

1. **Shape gaps:** no reported baseline, or no ratio, or no delta, or no numbers — any
   missing piece of the author's claim.
2. **Baseline divergence:** `|our_baseline − reported_baseline| / reported_baseline > 10–15%`.
   Our pre-merge run should be CLOSE to the author's old-code number if the scope matches.
   A big gap means different scope/harness/inputs — dig.
3. **Solution divergence:** `|our_solution − reported_solution| / reported_solution > 10–15%`
   — same investigation.
4. **Ratio gap:** our `×` and the author's `×` differ by more than ~1.5×.
5. **Sign flip:** our `×ratio < 1` (after-merge SLOWER). Legal only if explained
   (GPU-bound artifact, honest tradeoff) — never leave unflagged.
6. **Flat ratio (~1.0x) where author claims a big win** → suspect WORKLOAD-INVALID.

When ANY trips and it is not already explained: open the ORIGINAL PR on GitHub
(`gh pr view <n> --repo vllm-project/vllm --json title,body,comments` +
`gh pr diff <n> ...`), read the body AND all comments (benchmark tables often live in
comments), confirm what the diff ACTUALLY changes, and record `scope` + `scale_note`
(+ `workload_invalid` if the bench is wrong). Add the PR to the suspected list (§7).

Reference anomaly scan (computed over the 9 CPU PRs): EVERY experiment tripped at least
one threshold because microbench-scale ≠ PR-scale — which is EXPECTED and fine AS LONG AS
each carries `comparable:false` + an honest `scale_note`. The point of the thresholds is
not to force 1:1 agreement; it is to force an EXPLANATION for every divergence.

---

## 6. DO / DON'T

**DO**
- Use the four fixed labels, reported rows FIRST then measured rows, with consistent
  `(reported by PR's author)` orange / `(measured by us)` blue bracket suffixes.
- Keep every value in the experiment's single declared `unit`.
- Record the author's claim in its true shape (`base`+`patch`, or `pct`, or `delta`).
- Render `—` for any author number the PR did not publish, + a note on the fallback.
- Set `comparable:false` + `scope` + `scale_note` whenever scopes differ.
- Set `workload_invalid:true` + the exact setup fix when our bench misses the changed branch.
- Verify what the diff ACTUALLY changes before naming the experiment after a function.
- When the author published two metrics, state which is the headline and keep both.

**DON'T**
- DON'T write `base`, `gold`, `patch` in any rendered string: say "baseline" and "PR's solution".
- DON'T call one of OUR runs "measured" and the other "gold" — both are "measured".
- DON'T paste a ratio/percent into an absolute value cell.
- DON'T mix units within one experiment table.
- DON'T invent an author baseline when the PR gave only a ratio/delta — render `—`, use ours.
- DON'T claim 1:1 comparability across scales/harnesses — flag it.
- DON'T present a reverted/retracted benchmark as the merged change's result.
- DON'T trust a ~1.0x ratio as "no gain" without checking the bench exercises the branch.
- DON'T report an overhead ceiling (feature PR) as a speedup gain.

---

## 7. Suspected list (9 CPU PRs) — PR | issues | what was done

| PR | Issues found | What was done to fix |
|---|---|---|
| 6779 | Experiment named `make_tensor_with_pad` but PR does NOT touch it (byte-identical); real change = `array.array('l')` token storage in `SequenceData` feeding `from_sampling_metadata`. Our 20x vs author 3.6x. E2E arm blank. | Set `comparable:false`; `scope` = full from_sampling_metadata build (NVTX, len=1024 batch=2048); `scale_note` = author 3.6x is whole-stage, our 20x isolates the array-accelerated sub-op; filled e2e_qwen05b base=24.76 patch=29.44 it/s (~18.9%, author+reviewer agreed headline). |
| 29499 | JSON claimed "no absolute baseline" but a recoverable one exists (rearrange 1370→1077ms). Conflated two claims: 27% (E2E rearrange) vs 20-25% (per primitive). Sub-1x on move_*buffer under-explained. | `reported_gain` now states both claims; `scale_note` corrected — numpy moved only index/mask bookkeeping off GPU tensors, tensor `.copy_` transfers UNCHANGED, so CPU pays bookkeeping overhead with no GPU-sync elision → sub-1x contradicts no published number. |
| 27793 | Labeled "0.1%" gain, but it's an OBSERVABILITY feature (KV residency histograms); 0.1% is an overhead CEILING, not a speedup. | `reported_gain` → `"N/A (...)"`; `note` rewritten to explain feature + flag-gated zero-cost-when-disabled; kept experiments empty (correctness-only correct). |
| 21281 | Measured 1.007x (flat) vs author 4.3x. Bench pulls free blocks with `block_hash=None`, so BOTH arms hit the early-out and the 4→2 dict-lookup body NEVER runs. | Set `workload_invalid:true`; `scale_note` gives exact fix: enable_caching, non-None block_hash, pre-populate `cached_block_hash_to_block` so membership HITS, loop `_maybe_evict_cached_block` on cached blocks; expect ~2x isolated (4.3x is a trace upper bound on top of #21222). |
| 21222 | Our 1.7-1.8x vs author 19x/3.7x; free base 0.101ms looked 3x author's 33us. | Explained HARNESS-MISMATCH: 19x is engine "benchmark scripts" (GPU aggregate, screenshots), benchmark_block_pool is only "cut in half" ~2x — which OUR bench matches. 0.101ms is the 500-block-size row, not comparable to the engine's single 33us aggregate. `comparable:false` on both. |
| 13365 | Our 71x vs author ~29x; base 0.26s vs 54.3s. | Already SCALE-MISMATCH: our cumulative over 5000 synthetic calls vs author's full ShareGPT 5K-request E2E run; `comparable:false`, "~200x more cumulative work, exact call count not derivable". Direction genuine. |
| 32975 | UNIT-BUG (ratio 1.628 pasted as absolute). Author's benchmark times the min_tokens loop that was REVERTED (only num_output_tokens merged). | Fixed to 0.03936→0.02418 us/tok; `scale_note` documents reverted-code caveat + njhill's "common case 1 token/update is MORE work". `comparable:false`. |
| 33593 | present: our 2.3x vs author 3400x (author timed `list[:N]` materialization, a DIFFERENT op than short-circuit membership). absent: our 0.78x (SLOWER) vs author 1.18x (consumer differs: in-scan vs sum). | Both `comparable:false`; present scale_note = materialization vs short-circuit; absent scale_note = honest tradeoff (islice iterator-protocol per-element slower when token absent; wins only when present, which dominates). |
| 39328 | Our 354x/596x vs author 2.88x; base 0.35ms vs 32.5ms. | `comparable:false`; author's 32.5→11.3ms scope is per-uuid() computation (inferred, not quoted) diluted by other non-cached work; ours is a pure `functools.cache` HIT vs recompute. cold_compile correctly left unrun (not reproducible on CPU). |

Process to extend to future PRs: run §5 anomaly scan → for each trip, dig the original PR
per §4 → encode `scope`/`scale_note`/`comparable`/`workload_invalid` → add a row here.

---

## 8. Worked what-ifs

- **"Author said 2.88×, no absolute numbers."** → `reported:{pct:1.88}` (2.88× = +188%).
  `baseline (reported by PR's author)` = `—`; `PR's author solution (reported)` =
  our_base / 2.88 (time-like). Add `comparable:false` if the 2.88× covers a wider scope.
- **"Author said 0.15ms → 0.008ms."** → `reported:{base:0.15,patch:0.008}`; both orange
  rows show numbers; solution gain referenced to `0.15`.
- **"Author benchmarked code that was later reverted."** → keep the number, set
  `comparable:false` + `scale_note` that the benchmarked code is not what merged (32975).
- **"Our after-merge run is SLOWER than before-merge."** → legal, honest row. If it's a CPU
  artifact of isolating a GPU-bound change, say so in `scale_note` (29499 move_*buffer).
- **"Our ratio is ~1.0x but author claims a big win."** → suspect WORKLOAD-INVALID; verify
  the changed branch actually runs; if not, `workload_invalid:true` + the setup fix (21281).
- **"Author gave two different numbers (E2E vs per-primitive)."** → record both, name the
  headline in `reported_gain`, don't conflate them into one pct (29499).
- **"PR is a feature / observability change with an overhead %, not a speedup."** →
  `experiments:[]`, `reported_gain:"N/A (...)"`, explain in `note` (27793).
- **"Author published nothing numeric."** → `reported:{base:null,patch:null}`; both orange
  rows `—`; PR still earns its slot via a qualitative claim in `reported_gain`.
