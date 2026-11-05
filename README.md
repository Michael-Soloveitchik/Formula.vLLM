# vLLM perf-benchmark — GPU-machine kit (202 PRs, idea A)

Run the 202 selected vLLM PRs as diagnose/optimize tasks on an Ubuntu +
NVIDIA RTX 3060 Ti (8 GB) box. Each PR = pre-merge snapshot (`base_sha`) +
hidden behavior contract (`test_patch`) + workload + reported gain; the merged
PR (`merge_sha`) is the expert comparator.

## Files

| file | role |
|---|---|
| `index.json` | 202 PRs: pr, title, field, reported_gain (+units), `base_sha`→`merge_sha`, PR link |
| `config.yaml` | machine spec, measurement (repeats/variance), `paths.vllm_git`, solver rows |
| `conf.py` | dependency-free reader for `config.yaml` (no PyYAML on the target box) |
| `interesting_prs.md` | small CPU/agnostic PRs to start with |
| `Dockerfile.gpu` | builds vLLM from source at a PR's base_sha (CUDA 12.1, sm_86) |
| `run.sh` | run all / a set / specific PRs |
| `run_one.sh` | build + verify (base FAIL, gold PASS) + run solvers + measure one PR → `results/<pr>.json` |
| `measure.py` | paired A-B-B-A timing, log-speedup, MAD noise gate |
| `tasks/pr_<n>/` | vendored per-task data: `metadata.json` (test_files), `problem.md` (objective), `results_measured.json` |
| `data/` | the 202-PR detail view (per-PR `.md`/`.diff`) for offline cards |
| `harnesses/` | optional solver runners (Claude Agent SDK; codex/qwen) |
| `tests/` | compact pytest suite: config parse, index integrity, measure math |
| `web/index.html` | live dashboard: top stats + results/variance table + per-PR cards |
| `results/*.json` | per-PR results; `all.json` aggregate the HTML reads live |

## Self-contained

This dir is meant to be pushed to GitHub and cloned on the GPU box — it carries
its own `index.json`, vendored `tasks/`, `data/` view, and `conf.py`, so it has
**no dependency on sibling `datasets/` or `data/` dirs**. The one thing it does
NOT vendor is the vLLM git clone (too large): set its path once.

```bash
export VLLM_GIT=/abs/path/to/vllm        # or edit paths.vllm_git in config.yaml
```

## Solvers (harness × model × cmd)

`config.yaml` lists each solver as one line — a `harness` name, a `model`, and a
`cmd`. The runner runs `cmd + ' ' + <task objective>` in the task work dir; the
command edits `./vllm` and exits. Example row:

```yaml
solvers:
  - harness: claude
    model: Fable-5.1
    cmd: "claude --model=Fable-5.1"
    enabled: true
```

`python3 conf.py solvers` prints the enabled rows (`harness\tmodel\tcmd`), which
`run_one.sh` loops over. Flip `enabled` to add/remove a solver — no code edits.

## Units (reported to your boss)

Results carry the PR's **own units**, not an abstract gain. E.g. pr_21222 is
µs per queue op (base 150→8 µs get_new_blocks, 33→9 µs free_blocks). Serving
PRs use tokens/s or ms TTFT. The table shows base, gold, Δ%, speedup×, ±sd,
and the PR's reported number side by side. Every row links to the PR.

## Setup (GPU box)

```bash
# host: NVIDIA driver + nvidia-container-toolkit, docker
sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker
nvidia-smi                       # confirm the 3060 Ti is visible

# this repo + a vllm git clone side by side; point VLLM_GIT if elsewhere
export VLLM_GIT=/path/to/vllm
python3 gen_index.py             # (re)build index.json with base_sha from your clone
```

## Run

```bash
./run.sh                 # default set (interesting) from config.yaml
./run.sh 21222 21627     # specific PRs
./run.sh all             # all 202 (long)
./serve.sh                           # then open http://localhost:8000/web/
```

`run_one.sh` builds `vllm-task:<pr>` once at `base_sha` (pure-python PRs then
swap `.py` trees for gold/solver with no recompile), runs the real `test_patch`
for each arm, and measures latency with `repeats` independent runs → mean ± sd
(the variance). Results dump to `results/<pr>.json`; the HTML live-reads
`results/all.json`, so leave it open and reload as runs land.

## 3060 Ti (8 GB) notes

- `TORCH_CUDA_ARCH_LIST=8.6` (Ampere) keeps the build small.
- 8 GB VRAM ⇒ tiny models only (`facebook/opt-125m`, float16). Keep workloads
  small (few prompts, short seqs). Set per task in `config.yaml`.
- Correctness tests run fine; serving-throughput PRs need their small workload
  to fit — otherwise they're correctness-only.

## Harnesses (no LiteLLM on the GPU box)

The GPU box has no IBM LiteLLM, so codex is disabled there. Default solver =
**Claude Agent SDK** (`harnesses/claude_agent_harness.py`, `pip install
claude-agent-sdk`, `ANTHROPIC_API_KEY`). A harness is any command run in the
task work dir that edits `./vllm` and exits; its patched tree is then verified
against the hidden oracle. Enable/disable in `config.yaml`.

## Fairness / isolation (how FormulaCode does it, and us)

FormulaCode isolates each task in **its own container** and times base vs patch
**inside the same container, interleaved (paired A-B-B-A)**, so both arms see
the identical CPU/GPU, libs, and thermal state — the comparison is a ratio
measured back-to-back, not across machines or runs. It repeats to get a noise
distribution and only counts a change that clears a MAD-based threshold.

We mirror this:
- one image per PR, built at `base_sha`; gold/solver are `.py` swaps **in that
  same image** so the environment is byte-identical across arms;
- paired A-B-B-A timing in one process (`measure.py`);
- `repeats` independent runs → mean ± sd (variance), plus a nop (base-vs-base)
  spread to set the noise gate.

The noise model (log-speedup, MAD threshold, pooled sigma) lives in `measure.py`;
`measure.repeats`/`pairs`/`inner`/`k_sigma` in `config.yaml` tune it.
