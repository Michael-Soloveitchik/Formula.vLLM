# Prompt — agendas.md + agendas_research.md build

Source-of-truth for the **agenda layer**: turning the clustered PR index into (a)
`agendas.md` — 87 agendas across 9 clusters on one comparable schema, with
measured gains — and (b) `agendas_research.md` — lifecycle, cross-cluster /
cross-area closeness, recurring patterns, and worked examples.

This is downstream of `prompt_clustering.md` (field assignment → `data/index.jsonl`)
and `prompt_subclustering.md` (area assignment). Do NOT re-run clustering here; the
agenda layer consumes its output.

**Hard constraint: do NOT use the web. Everything needed is already on disk.**

---

## 0. Vocabulary this prompt uses

- **cluster / field** — one of the 9 efficiency fields. Each PR is assigned to
  **exactly one** field (single-label). Clusters + PR counts (frontier state):
  `gpu_kernels`(1816) `quantization`(925) `parallelism`(829) `attention`(800)
  `scheduling`(727) `decoding_tpot`(583) `memory`(164) `kv_cache`(87)
  `prefill_ttft`(64).
- **area** — the dominant changed-file subtree of a PR (`csrc/quantization`,
  `model_executor/layers`, `attention/backends`, `distributed/eplb`, …). A PR has
  one area. Areas live *inside* a cluster.
- **agenda** — a recurring direction/lever of improvement behind a group of PRs.
  Target: ~87 agendas total (verify with `grep -c "^### Agenda:" agendas.md`).
- **evidence PR** — a PR cited in an agenda's `- **evidence:**` line (the PR chain
  that proves the agenda exists). ~402 unique across all agendas; only **5** are
  reused across agendas (this is the empirical fact behind "cross-cluster =
  CLOSE agendas, not shared PRs").

---

## 1. Inputs (preconditions — must already exist)

| path | role |
|---|---|
| `data/index.jsonl` | one JSON per PR: `{pr,title,date,sha,field,area,n_files,tags,excl_reason,files}`. `field` is single-label; `null` = excluded. |
| `data/digests/<cluster>.md` | per-field title+date digest fed to the extraction agents (one per cluster, 9 files). |
| `data/stats.md`, `report.md` | human-readable rollups used by agents for context. |
| `data/prs/<field>/<area-slug>/<pr>.md` | downloaded PR body (title + `## Body` + `## Comments`). Needed by the gain scanners. |
| `data/views/perf_gains.jsonl` | **strict** gain scan (`perf_scan.py`), 376 PRs. Reliable PR-number→gain map; used to seed evidence-PR gains in assembly. |
| `data/views/quant_gains.jsonl` | **broad** gain scan (`quant_scan.py`), 443 PRs. Used to attribute gains to ALL quantitative PRs, not just evidence PRs. |

If `quant_gains.jsonl` is stale, regenerate it in Step 2 before assembling.

---

## 2. Output files (what this build creates)

| path | created by | durable? |
|---|---|---|
| `/tmp/agendas_new_<cluster>.md` (9 files) | the 9 extraction agents (Step 3) | temp — regenerated each run |
| `pipeline/quant_scan.py` | committed | yes |
| `pipeline/assemble_agendas.py` | committed (was `/tmp/assemble_agendas.py`) | yes |
| `pipeline/attribute_gains.py` | committed | yes |
| `data/views/quant_gains.jsonl` | `quant_scan.py` | yes |
| **`agendas.md`** | `assemble_agendas.py` then `attribute_gains.py` | **primary deliverable** |
| **`agendas_research.md`** | hand-authored from `agendas.md` + the two gain views | **primary deliverable** |
| `/tmp/agendas_pre_m3.md`, `/tmp/agendas_pre_gains.md` | manual backups before destructive rewrites | temp |

---

## 3. The shared comparable schema (THIS is what makes clusters comparable)

Every agenda in every cluster is written on the SAME schema, in this order. If an
agent emits jargon-only titles or skips a field, the cross-cluster comparison
breaks — reject and re-run that cluster.

```
### Agenda: <plain-English title — a reader with a bachelor's can parse it>
- **lever:** <one of the closed lever vocab>
- **trigger:** <one of the closed trigger vocab>
- **wins:** <one or more of the closed wins vocab>
- **continuity / span:** continuous | scattered | single — <date range> · <N PRs>
- **gains:** <filled by assemble/attribute — leave the literal placeholder line>
- **pattern:** WHEN <condition> → DO <move> → GAIN <metric>.   (reusable recipe)
- **background:** plain-English: why all these PRs are one agenda.
- **plain:** simplest one-line explanation.
- **example:** one concrete fix — "#NNNN — <what it changed>".
- **evidence:** #a #b #c …  (the PR chain)
```

### Closed vocabulary (do not invent terms)

**lever** (the engineering move) — distribution across the 87 agendas in parens:
- `specialize`(24) — write/tune/port a dedicated kernel or path for a specific
  model/hardware/format
- `reduce-precision`(11) — fewer bits (FP8/FP4/INT) to move less data
- `overlap`(11) — run independent work concurrently to hide latency
- `offload`(11) — move data to a cheaper/larger tier (CPU/disk/remote)
- `skip-work`(8) — do less total work (sparsity, speculative accept, eviction,
  early exit)
- `reuse`(7) — avoid recompute by caching/sharing
- `fuse`(6) — merge ops/steps into one kernel to cut round-trips/launches
- `batch`(4) — pack more work into one step
- `reorder`(3) — change the policy/order of what runs when
- `shard`(2) — split work/weights across devices

**trigger** (what keeps new PRs landing — the reuse driver) — counts in parens:
- `overhead-found`(36) — one-shot per design; converges once the overhead is gone
- `new-model`(20) — **perpetual**, re-fires every model release
- `new-backend`(10) — **perpetual**, every new kernel library
- `scale-up`(8) — **perpetual**, as clusters/context grow
- `new-precision`(7) — **perpetual**, every new number format
- `new-hardware`(5) — **perpetual**, every GPU generation
- `correctness`(1) — one-shot

**wins** (metric improved): `throughput` · `TPOT` (decode/per-token) · `TTFT`
(first-token/prefill) · `memory` · `accuracy-neutral`.

**continuity / span**: `continuous` (same lever pushed harder over time) ·
`scattered` (same idea reapplied in new places) · `single` (1-2 PRs).

The two distribution tables above must each sum to 87. They are the answer to
"are these agendas reusable in the future?" — 42 agendas (new-model + new-backend
+ scale-up + new-precision + new-hardware) re-fire on every external release and
never finish; the 36 `overhead-found` agendas are the only ones that can converge.

---

## 4. Step-by-step reproduction

### Step 0 — set the frontier
Latest repo activity date = newest PR date in `index.jsonl` (currently
`2026-05-18`). "Last date" of an agenda = newest evidence PR in it. These drive
the lifecycle buckets in Step 6.

### Step 1 — backups
Copy the current `agendas.md` to `/tmp/agendas_pre_m3.md` (before any schema
rewrite) and `/tmp/agendas_pre_gains.md` (before the gains rewrite). Both rewrites
are destructive full-file writes; keep the restore point.

### Step 2 — broad quantitative gain scan  →  `data/views/quant_gains.jsonl`
Run `python3 pipeline/quant_scan.py`. It scans **every downloaded PR body** (looser
than `perf_scan.py`) and emits `{pr,field,area,title,gains:[...],max_pct,max_x}`.
Critical filters (reproduce EXACTLY — these are the edge cases that cost the most
iterations):
- `HWCOUNT` masks GPU-count notation (`8xB200`, `4×GB200`) BEFORE number parsing,
  else "8x" reads as an 8× speedup.
- `HW` masks bare hardware/format tokens (`MI300X`, `H100`, `fp8`, `w8a8`, `e4m3`,
  `sm90`, `gfx942`) else `MI300X` → "300X".
- `GAIN` keeps only ratio/percent/memory units: `% x × GiB GB`. Absolute `tok/s`,
  `req/s`, `ms` are RAW benchmark output (not a delta) — excluded to kill noise.
- `PERF` word must appear in the same sentence (improv/throughput/latenc/speedup/
  faster/…); `NEG` drops sentences about coverage, gpu util, concurrency, gpu
  blocks, kv cache SIZE, block size, warm-up, profiling, dummy, max_num, memory
  util, "slower", regress, gsm8k, pass@, histograms (`█▏▎…`).
- `is_noise`: drop `x ≤ 1.05` (parity) and `x ≥ 50` (log/microbench noise, e.g.
  the "134.84x Maximum concurrency" startup-log false positive), and `0%`.
- Expected result ≈ **443 PRs** (down from an un-tightened ~629 — the drop IS the
  noise removal; do not try to recover it).

### Step 3 — per-cluster agenda extraction (9 PARALLEL agents)
Launch **one agent per cluster** (9 total, in parallel — single message, 9 Agent
calls). Each agent reads ONLY its own `data/digests/<cluster>.md` (+ stats/report
for context) and writes `/tmp/agendas_new_<cluster>.md`.

**Agent task template** (fill `<CLUSTER>`):
> Read `data/digests/<CLUSTER>.md` (title+date per PR for the `<CLUSTER>` field)
> and `data/stats.md`/`report.md` for context. Identify the recurring improvement
> **agendas** in this cluster. Emit each agenda on the EXACT schema in
> `prompts/prompt_agendas.md` §3 — every field present, closed vocab only, titles
> in plain English (a bachelor's-level reader must parse them; no bare jargon).
> `pattern` must be a reusable WHEN→DO→GAIN recipe so we can later ask "does this
> apply to a future model/hardware/format?". `background` says why all these PRs
> are one agenda. `plain`+`example` are simple + one concrete fix with a real PR
> number from the digest. Leave the `- **gains:**` line as the literal placeholder
> `- **gains:** <<<FILL>>>` (assembly fills it). After the agendas, emit a
> `===TABLEROWS===` marker, then one matrix row per agenda:
> `| <CLUSTER> | <title> | <lever> | <trigger> | <wins> | <continuity> |`.
> Same structure for every agenda in the cluster.

Edge cases that bit us here:
- **The `memory` cluster digest had no `plain`/`example` seeds** (the earlier M2
  pass never reached it). Its agent MUST author grounded `plain`/`example` from
  real PRs — do not leave them blank. Known-good PRs: `#11743` (~70 GiB sleep
  mode), `#30515`, `#30407`, `#14469`. Avoid the thin bugfix PRs `#13315`,
  `#25721`, `#11810`.
- Titles that are jargon-only are a REJECT condition — the whole point of M3 is
  cross-cluster comparability. Re-run the offending cluster.

### Step 4 — assemble  →  `agendas.md`
Run `python3 pipeline/assemble_agendas.py`. It:
1. concatenates the 9 `/tmp/agendas_new_<cluster>.md` in the fixed `order` list;
2. splits off each file's `===TABLEROWS===` block and builds the header, the
   closed-vocab legend, and the **Cross-cluster comparison matrix**
   (`| cluster | agenda | lever | trigger | wins | continuity |`);
3. seeds each agenda's `- **gains:**` line from `perf_gains.jsonl` by **evidence
   PR number** (the strongest attribution signal), plus the first concrete number
   from the worked `example` if nothing already represents it;
4. edge case: the `[n/m sampled PRs carry a measured benchmark]` tail is emitted
   **only when `npw>0`** (never print "0/6" next to a listed number).

### Step 5 — attribute ALL quantitative PRs  →  rewrite gains in `agendas.md`
Run `python3 pipeline/attribute_gains.py` **after** assembly. It assigns every one
of the 443 `quant_gains.jsonl` PRs to a single agenda and rewrites each
`- **gains:**` line as: `N PRs carry a measured gain; top: … — mixed measures, not
additive.` Mechanics + edge cases:
- Attribution = **same field** + title/area keyword overlap (`STOP` word list
  removes generic tokens like kernel/support/optimization). Area-token overlap
  counts **double** (area is a strong signal).
- `FIELD_ALIAS = {"platform":"gpu_kernels"}` — platform/ROCm/TPU PRs compete for
  gpu_kernels agendas (they have no agendas of their own).
- **Evidence PRs are force-assigned**: `if pr["pr"] in a["evid"]: return a` — an
  evidence PR always lands in its own agenda regardless of keyword score.
- `gkey` orders gains by unit then magnitude (x-speedups first, then tok/s, %,
  memory, time); `disp` normalizes `×`→`x`, `gib`→` GiB`.
- Expected: **443 scanned, ~375 attributed, ~68 unattributed.** The ~68 are PRs
  whose field has no close agenda by keyword — acceptable; do not force them.
- Why TWO scans: `perf_gains` (strict, 376) gives trustworthy PR-number gains for
  evidence seeding; `quant_gains` (broad, 443) maximizes coverage for the roll-up.
  The user's requirement was "harvest ALL quantitative PRs, not just the ~6
  evidence PRs per agenda" — Step 5 is what satisfies it.

### Step 6 — author `agendas_research.md` (hand-written analysis)
Built from `agendas.md` + both gain views. Six sections:

**§1 Lifecycle (active / waning / dormant).** Bucket each agenda by its last
evidence-PR date vs the frontier:
- `active` (≥ 2026-01-01): ~78. `waning` (2025-07…2025-12): ~7. `dormant`
  (< 2025-07): 2.
- Headline: efficiency work almost never "finishes" — 78/87 run to the frontier
  because their triggers (`new-model`/`new-hardware`/`new-precision`) re-fire on
  every release.
- The **2 dormant** are V0-engine multi-step agendas (scheduling multi-step
  2025-02; prefill multi-step + CUDA-graph chunked prefill 2025-01) — **designed
  out** by V1 Async scheduling + Model Runner V2, not "solved and abandoned."
- The **7 waning** sit in the two smallest clusters (`kv_cache`, `prefill_ttft`);
  their work **migrated** into bigger clusters (prefix caching→scheduling, KV
  movement→memory/parallelism, prefill overhead→Model Runner V2).

**§2 Cross-CLUSTER meta-agendas.** The same idea recurs in different clusters.
Table sorted by #clusters spanned. **State explicitly: crossing is because the
agendas are CLOSE, not because PRs are shared** — single-label means each PR is in
one cluster; of 402 evidence PRs only 5 are reused, so meta-agendas are *distinct
PR sets doing sibling work*. Members (clusters spanned / member count): Disagg
prefill/KV transfer (5/6), Model Runner V2 (5/5), CUDA graph (5/5), FP8/low-prec
(3/14), MoE/experts (3/10), DeepSeek/MLA (3/4), FlashInfer (3/4),
torch.compile/fusion (3/3), Prefix caching (3/3), Sliding-window/sparse (3/3),
Mamba/SSM (2/2), Chunked prefill (2/3), RoPE (2/3), LoRA (2/2).

**§2b Cross-AREA agendas within ONE cluster.** The same closeness one level down:
sibling agendas inside a single cluster that differ only in which *area* (file
subtree) they touch. Same "close not shared" fact. Mechanism = **vLLM's file
layout** splits one lever into area-threads: kernel in `csrc/`, caller in
`model_executor/layers/`, selection in `platforms/`+`config`, comms in
`distributed/`. Examples: quantization FP8 (layers vs csrc/quantization vs cache);
gpu_kernels CUTLASS/MoE/AITER (csrc vs model_executor vs platform); attention MLA
kernels vs MLA backend-selection (backends vs platforms/config = build-vs-default);
parallelism MoE (layers vs device_communicators vs eplb); scheduling
(chunked-prefill vs prefix-cache vs P/D connector). Contrast with §2: cluster-cross
= subsystem boundaries, area-cross = file-tree boundaries.

**§3 Patterns that recur across agendas — each tagged with #agendas it summarizes.**
Lead with the two distribution tables (lever counts, trigger counts; each sums to
87). Then the structural shapes:
1. **Build it → then make it the default** (~12 agendas / 6 build-default pairs):
   FlashInfer integration vs MLA selection; CUDA graph support vs capture/dispatch;
   chunked prefill enablement vs default; FP8 support vs scaling/tuning; prefix
   cache support vs default-on; KV connector API vs P/D scheduling.
2. **One lever, squeezed in-place AND spread wide** (~29 = 11 reduce-precision + 11
   overlap + 7 reuse): continuous within a cluster + scattered across clusters.
3. **A model/format release fans out into every cluster** (32 = 20 new-model + 7
   new-precision + 5 new-hardware): DeepSeek→MLA + MLA kernels + block-FP8/DeepGEMM
   + MoE/EP.
4. **Shared-abstraction refactor** ("pull the special case into a base class", ~6):
   `FusedMoEMethodBase` (#5970), V1 KV Connector API (#15960), all2all dispatch
   iface, modular quant kernel framework. Signature = a `*Base`/connector/registry
   class landing AFTER a burst of near-duplicate PRs.
5. **Port the proven win to other hardware/backends** (~8): AITER/AMD (~60 measured
   PRs — biggest quant block), INT8 CPU/ARM, TPU KV, FlashInfer/TRTLLM as 2nd
   backend. Finished algorithmically, re-fires per platform.

**§4 Where measured gains concentrate.** From the broad scan: `gpu_kernels`
dominates (~150 quant PRs: AITER 60, Fused MoE 30, CUTLASS 21, LoRA 19, FP8 16).
Big ratios (2–6×) in kernels/attention; percent gains (3–50%) in
scheduling/parallelism; memory gains (GiB) in the memory cluster. ~25 agendas carry
no benchmark number — they are `correctness`/enabling, value = "it works / doesn't
OOM", not a speedup. Note: gains are different measures, NOT additive; attribution
is ±1 cluster.

**§5 Cross-cluster worked examples.** Copy 3 meta-agendas verbatim from `agendas.md`
with their real member agendas, and explain the MECHANISM of why one idea pays off
in several clusters at once:
- 5.1 **FP8 / `reduce-precision`** — "use fewer bits" is a representation change
  every byte-moving stage adopts independently (weights→quant, GEMM→gpu_kernels,
  KV-cache→memory). Memory-bandwidth-bound → half the bits = half the read time.
- 5.2 **MoE / experts** — one layer, three independent costs in three clusters:
  compute (kernel), footprint (quant experts), communication (all2all). Fixing one
  alone still stalls.
- 5.3 **Disaggregated prefill** — one decision ("prefill and decode on different
  GPUs") refracted through scheduler / transport (NIXL) / memory registration /
  TTFT. Disaggregation *relocates* a cost; the relocation is paid in every
  subsystem it passes through.
- General rule: an idea crosses clusters when it changes something **shared** — the
  byte-width of data (5.1), the structure of a layer (5.2), or the placement of a
  stage (5.3).

---

## 5. Run order (one shot)

```bash
cd /Users/michaelsoloveitchik/Repositories/Projects/vLLM/Knowledge
cp agendas.md /tmp/agendas_pre_gains.md            # restore point
python3 pipeline/quant_scan.py                     # → data/views/quant_gains.jsonl (~443)
# Step 3: launch 9 parallel extraction agents → /tmp/agendas_new_<cluster>.md
python3 pipeline/assemble_agendas.py               # → agendas.md (schema + matrix + seed gains)
python3 pipeline/attribute_gains.py                # rewrites gains lines (443 attributed→375)
# Step 6: hand-author agendas_research.md from agendas.md + the two gain views
grep -c "^### Agenda:" agendas.md                  # sanity: ~87
```

`perf_scan.py` (strict, → `perf_gains.jsonl`) is maintained by the clustering layer
(`prompt_clustering.md` v8) and is a prerequisite of assembly — it does not need
re-running here unless the clustering changed.

---

## 6. Edge-case checklist (verify before declaring done)

- [ ] Both distribution tables (lever, trigger) in `agendas_research.md` §3 sum to
      exactly **87**.
- [ ] Cross-cluster/cross-area sections state **"close agendas, not shared PRs"**
      and cite the 5/402 evidence-reuse fact. Single-label index is the proof.
- [ ] Gain scan masks hardware COUNT (`8xB200`) AND bare hardware/format tokens
      (`MI300X`,`fp8`,`e4m3`) before number parsing.
- [ ] `is_noise` drops parity (`≤1.05x`, `0%`) and log-noise (`≥50x`); #17494-style
      "134.84x Maximum concurrency" startup-log lines do NOT appear.
- [ ] `tok/s`/`req/s`/`ms` are EXCLUDED from `quant_gains` (raw output, not deltas).
- [ ] `attribute_gains.py` force-assigns evidence PRs and aliases
      `platform→gpu_kernels`; ~375/443 attributed, ~68 unattributed left alone.
- [ ] The `[n/m sampled PRs]` tail only prints when `npw>0`.
- [ ] `memory` cluster agendas have real `plain`/`example` (not blank — the digest
      lacked seeds).
- [ ] Every agenda carries all schema fields in order; no jargon-only titles.
- [ ] Gains are described as **different measures, NOT additive** everywhere they
      are rolled up.
- [ ] Lifecycle: 2 dormant are V0 multi-step **designed-out by V1**, not abandoned;
      7 waning **migrated** into bigger clusters.
```
