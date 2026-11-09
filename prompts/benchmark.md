# Benchmark harness prompt — Formula.vLLM directory + workload contract

Governs the HARNESS that produces the numbers. The NUMBERS themselves (units, row labels,
`reported` shapes, comparability flags) are governed by the sibling spec
`numeric_solution.md`. This file: directory layout, per-PR `workload/Dockerfile`,
weight-download scripts, and how `run.sh` drives the isolated per-task images.

Benchmark root: this repo (remote `github.com/Michael-Soloveitchik/Formula.vLLM`). Self-contained:
clone on the GPU box (docker + NVIDIA container toolkit), run. Nothing vLLM-related is needed on
the host: every per-PR image installs the PyPI wheel and fetches the PR's two commits itself.

---

## 1. Directory layout

The authoritative tree, with the owner of every file, is the first section of `README.md`
("Repository layout"); `gen/layout.py` is the single owner of paths in code. In short:

```
data/pr_<n>/            # EVERYTHING about one PR
  pr.md pr.diff solution.md     raw PR material (crawl)
  task.json                     the experiment definition (hand-written; `requirements` generated)
  bench.py tests/ notes.md      hand-written extras (optional)
  problem.md                    generated: solver objective (solution.md quoted + experiments)
  workload/default/             generated: Dockerfile, run_workload.sh, download.sh, workload.json, WORKLOAD.md
  workload/<hw-slug>/           generated: same, with task.json:hardware_overrides[<hw-slug>] applied
                                #   hw-slug `cpu` renders templates/Dockerfile.cpu-wheel (CUDA→CPU MIRROR of
                                #   the gpu-wheel image: same shas/trees/deps, torch CPU wheel, no CUDA)
  files/<arm>/                  solver OUTPUT: a vllm/ tree overlay the arm edited (one dir per arm).
                                #   cp -a files/<arm>/. over the base tree = that arm's applied solution.
  files/<arm>.patch             generated: unified diff (base→overlay) of ONLY the changed rows.
                                #   remote applies with one cmd: cd <vllm_site> && git apply files/<arm>.patch
  results.<hw-slug>.json        written by a run (record.py) — one per hardware the PR ran on
  results.ai.<arm>.json         written by apply_ai_solutions.py — files/<arm>/ re-measured vs gold on CPU
  logs/<hw-slug>/  work/        written by a run (git-ignored)
pipeline/   run.sh run_one.sh queue.sh health_check.sh run_workloads.sh record.py parse_workload.py
            measure.py aggregate.py derive.py machine.py requirements.py next_prs.py reproduce_report.py
gen/        layout.py layout.sh gen_index.py gen_requirements.py gen_workloads.py gen_tasks.py conf.py migrate_layout.py
tools/      check_terms.py serve.py serve.sh        stats/  build_stats.py stats.ipynb + generated csv/png
templates/  Dockerfile.gpu-wheel + Dockerfile.cpu-wheel (CPU mirror; both rendered per PR) + legacy *-source
web/index.html  index.json (generated catalog)  results/{all,manifest}.json (generated aggregate)
clusters.json config.yaml dict_of_harnesses.json (hand-written)   .work/ (global scratch)
run.sh run_one.sh queue.sh serve.sh   thin root wrappers into pipeline/ and tools/
```

---

## 2. The isolation model (FormulaCode-style)

One image per PR, built ONCE at `base_sha`. Every arm is a `.py`-tree SWAP inside that same
image, so the environment is byte-identical across arms — only the vllm source differs.

Arms per PR:
- `base`  = `base_sha` tree (pre-PR). Oracle expects tests FAIL.
- `gold`  = `merge_sha` tree (PR's solution). Oracle expects tests PASS.
- each enabled solver = a copy of the `base` tree that the harness edits, then re-verified.

`run_one.sh` pipeline: regenerate + build the per-PR image (`data/pr_<n>/workload/<hw-slug>/Dockerfile`,
rendered from `templates/Dockerfile.gpu-wheel`; /base and /gold are exported from it) →
isolate (`cp -a /arm/vllm → site-packages`, run hidden pytest oracle) → solve (each harness
edits `./vllm`) → measure (the task's workload cmd, `repeats` × per arm, ABBA) → record
(`data/pr_<n>/results.<hw-slug>.json`).

`measure.py`: paired A-B-B-A in one process, `repeats` independent runs → mean ± sd,
nop (baseline-vs-baseline) spread sets the MAD noise gate. Knobs in `config.yaml:measure`.

---

## 3. Solvers — `dict_of_harnesses.json` is the source of truth

`gen/conf.py solvers` reads `dict_of_harnesses.json` first, falls back to `config.yaml:solvers`
only if absent. Each arm: `key` (dashboard arm name, MUST match `web/index.html` HARNESSES),
`model`, `cmd`, `guided`, `enabled`, `env`, optional `compile`.

- `guided:false` = BLIND — objective has the `## PR's solution` section STRIPPED
  (FormulaCode: discover the optimization yourself).
- `guided:true`  = objective = full `problem.md` incl. merged patch, asked to BEAT it.
- `compile:"cpu"` = the arm must leave its edit under `data/pr_<n>/files/<arm>/` (a vllm/ tree
  overlay) and that overlay must COMPILE + run on the CPU mirror image. `apply_ai_solutions.py`
  overlays it on the base tree and re-measures with the SAME instrument (parse_workload.py).
  CPU proves correctness/applies-cleanly only; the real speedup is the GPU number.

The 8 arms = {claude, codex} × {blind, +PR's solution} × {GPU run, +cpu-compilation}.

Runner invokes `<cmd> "<objective>"` inside a copy of the baseline tree; the command edits
`./vllm` and exits. GPU box has NO IBM LiteLLM → codex arms disabled there; default solver
= Claude Agent SDK. Flip `enabled` to add/remove — no code edits.

---

## 4. Per-PR workload/ + Dockerfile (the per-task reproduction kit)

GOAL: each task carries `data/pr_<n>/workload/` that reproduces the PR AUTHOR's workload as
closely as possible — same model, same dataset, same bench cmd — so our measured ratio can
converge on the author's (the 1:1 goal from `numeric_solution.md §4`).

```
data/pr_<n>/workload/<hw-slug>/   # AUTO-GENERATED by gen/gen_workloads.py (idempotent); one dir per
                                  # hardware slug (`default` = generic, `<hw-slug>` = this machine with
                                  # task.json:hardware_overrides[<hw-slug>] applied: cmd, repeats, extra_pip, torch_fix, note)
  Dockerfile        # THE image run_one.sh builds: templates/Dockerfile.gpu-wheel rendered with the
                    #   task's wheel, shas, torch_fix, extra_pip (nvidia/cuda + pip install vllm==wheel;
                    #   /base + /gold cloned in-image). hw-slug `cpu` renders templates/Dockerfile.cpu-wheel
                    #   instead — a CUDA→CPU MIRROR: identical shas/trees/deps, torch CPU wheel, no CUDA.
                    #   gen_workloads.dockerfile_template(hw) picks the template; nothing else differs.
  run_workload.sh   # exact author bench cmd (task.json:workload.cmd), one arm per call (ARM env),
                    #   prints parseable lines, exit 0 even if an arm skips (GPU absent)
  download.sh       # fetch-on-demand weights for THIS pr's cluster into the SHARED
                    #   <repo>/.weights/<cluster> (gitignored); gated + idempotent
  workload.json     # CANONICAL experiment description: image + build cmd, exact cmd/env,
                    #   measurement policy (read from pipeline/run_one.sh at generation time), metrics
                    #   [{name,unit,higher_is_better,parse}], reported numbers + the author's
                    #   hardware/args quoted from pr.md, requirements, reproduced numbers
                    #   (results.<hw-slug>.json), BASELINE_GAP.md verdict, standalone docker lines
  WORKLOAD.md       # the same, rendered for humans: what is measured, how to re-run 1:1
                    #   without run_one.sh, reported vs reproduced, gap note
```

`gen/gen_workloads.py` generates these five files (for `default` + this machine's slug) and
`data/pr_<n>/problem.md` for every PR whose task.json has a workload cmd, from `task.json` +
`clusters.json` + `solution.md` + `pr.md` + `results.<hw>.json` + `BASELINE_GAP.md` +
`pipeline/run_one.sh`. Re-runnable and idempotent (no timestamps): overwrites the generated
files, never touches `task.json` / `bench.py` / `notes.md`. `run_one.sh` regenerates the PR's
dir before every build, so the image always matches task.json.

`./run.sh workloads` → `pipeline/run_workloads.sh`: iterates `clusters.json["order"]`, runs each
cluster's `download.sh` ONCE, then builds + runs every PR's workload Dockerfile for both
arms (`data/pr_<n>/work/{base,gold}_full` trees mounted at `/src`, shared weights at
`/weights`). GPU clusters self-skip on non-CUDA hosts; Mixtral is last and gated. One image per task.

Workload taxonomy (`task.json:workload.kind`, 40 mined): `microbench_in_pr`, `repo_benchmark`,
`serving`, `pytest_bench`. Every workload runs through the GPU wheel image (`needs: gpu`), even
when the bench itself has no CUDA kernels.

- Non-GPU hosts: SCAFFOLD ONLY — generate the workload dirs, do not execute.

### Applying saved solutions (`./run.sh --apply-ai-generated-solutions <pr>...`)

`pipeline/apply_ai_solutions.py` is the CPU re-measure path, independent of a GPU run and of any
solver CLI/API key. For each PR+arm it: builds the PR's CPU image (`workload/cpu/Dockerfile`),
exports /base + /gold, overlays `files/<arm>/` on base, runs the task's bench cmd ABBA-interleaved
`REPEATS` times, folds with `parse_workload.py` (SAME instrument as `record.py`), and writes
`results.ai.<arm>.json {scenario:{gold,<arm>,ratio_arm_over_gold}}`. The absolute number is not
the GPU speed — it proves the saved solution compiles, applies cleanly, and the instrument reads it.

It also writes `files/<arm>.patch` — a unified diff of ONLY the rows the arm changed (baseline →
overlay). The remote GPU box applies a solution without copying whole trees:
`cd <vllm_site> && git apply data/pr_<n>/files/<arm>.patch` (or `patch -p1 < ...`). The full
`files/<arm>/` overlay is kept too (fallback `cp -a files/<arm>/. <vllm_site>/`).

---

## 5. Weights — clustered download scripts, never bundle

Do NOT commit weights. `clusters.json` groups PRs by hardware target + shared model so each
model downloads ONCE and same-model tasks run back-to-back. `clusters.json["order"]` sizes
the pulls small→big so the 90 GB one is last + optional. Clusters:

| cluster | model / dataset | approx size | PRs |
|---|---|---|---|
| `none_cpu` | (none) | 0 | 10 pure-CPU micro/repo/pytest |
| `none_gpu` | (none, needs CUDA) | 0 | 6 GPU kernel/Triton microbench |
| `opt125m` | `facebook/opt-125m` | ~250 MB | 16436, 22940 |
| `qwen05b` | `Qwen/Qwen2-0.5B` | ~1 GB | 6779 (e2e arm), 28280 |
| `llama8b_fp8` | `neuralmagic/Meta-Llama-3-8B-Instruct-FP8` + ShareGPT | ~8.6 GB + ~650 MB | 9972, 28280 |
| `mixtral` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ~90 GB | 12583 (DEFERRED last, gated) |

Rules:
- Each PR's `download.sh` fetches its cluster's weights into the SHARED
  `.weights/<cluster>/` (gitignored), NOT a per-PR dir — cluster reuse, pull once.
- SAFE + idempotent: `cache HIT` skip if present, `resume_download=True`, snapshot_download.
- `pipeline/run_workloads.sh` runs `download.sh` ONCE per cluster (first PR in the cluster), then
  runs all that cluster's tasks off the shared cache before the next cluster.
- Mixtral refuses unless `ALLOW_MIXTRAL=1` (~90 GB, multi-GPU TP=2); ordered LAST.
- Download NOTHING at author time — scripts only; they pull + run when invoked on the GPU box.

---

## 6. DO / DON'T

**DO**
- Keep the dir self-contained + pushable; vendor everything except weights.
- One image per PR (wheel + both trees); arms = `.py` swaps in that same image.
- Keep the `cpu` workload image a byte-faithful MIRROR of the GPU one (only CUDA→CPU torch differs).
- Have `cpu-compilation` arms drop their edit in `files/<arm>/`; `--apply-ai-generated-solutions` re-measures it.
- Make per-PR `workload/` reproduce the author's model+dataset+cmd (1:1 goal).
- Scaffold (don't execute) GPU tasks on non-GPU hosts.
- Cluster weight downloads by shared model; idempotent, resumable scripts.
- Edit arms via `dict_of_harnesses.json`; keep keys in sync with `web/index.html` HARNESSES.

**DON'T**
- DON'T commit weights, datasets, or `.work/` scratch.
- DON'T download anything at author time — ship scripts that pull on the GPU box.
- DON'T pull a shared model once per PR — cluster and reuse.
- DON'T execute GPU workloads on the arm Mac — build/scaffold only.
- DON'T diverge arm keys from the dashboard HARNESSES list (blank rows otherwise).
