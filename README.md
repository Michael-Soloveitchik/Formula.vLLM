<p align="center"><img src="web/shots/logo_banner.png" alt="Formula.vLLM" width="820"></p>
<p align="center">
<a href="https://station-wsl-michael.taila4786c.ts.net/formula/web/"><img src="https://img.shields.io/badge/dashboard-live-0b8ce8?style=for-the-badge&labelColor=172a5c" alt="live dashboard"></a>
<img src="https://img.shields.io/badge/PRs-202-ffc426?style=for-the-badge&labelColor=172a5c" alt="202 PRs">
<img src="https://img.shields.io/badge/isolation-FormulaCode--style-1aa7ff?style=for-the-badge&labelColor=172a5c" alt="FormulaCode-style">
<img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti%20sm__120-0b8ce8?style=for-the-badge&labelColor=172a5c" alt="GPU">
</p>

<h3 align="center">202 real vLLM perf PRs, each a benchmark task. Did the PR's speedup reproduce?</h3>
<p align="center">Live dashboard: <a href="https://station-wsl-michael.taila4786c.ts.net/formula/web/">station-wsl-michael.taila4786c.ts.net:10000/formula/web/</a></p>

![dashboard](web/shots/dashboard_wide.png)

## <img src="web/shots/h/goal.svg" height="40" alt="Goal">

Turn merged vLLM perf PRs into reproducible tasks. Pre-merge code = baseline.
Merged code = author's solution. Measure both, side by side, in the PR's own unit.
Check the claimed gain is real.

## <img src="web/shots/h/example.svg" height="40" alt="Example">

PR #21222 (block-pool queue ops): author claimed 19×. Our isolated microbench shows
~2× — because the 19× was a whole-engine aggregate, not the one op. Both numbers are
genuine; different scope. The dashboard shows both and names the gap.

## <img src="web/shots/h/inspiration.svg" height="40" alt="Inspiration">

FormulaCode. One isolated image per task; baseline vs solution timed back-to-back in
the SAME container (paired A-B-B-A); repeat for a noise distribution; count only gains
that clear a MAD gate.

## <img src="web/shots/h/results.svg" height="40" alt="Partial results">

![table](web/shots/table_measured.png)

- 202 PRs catalogued. 35 standardized. 32 reproduce the author's direction.
- 9 CPU PRs fully measured in Docker (arm Mac). 26 GPU PRs carry the author's genuine
  reported numbers; measured arms pending on a GPU box.
- Key finding: the 202 PRs are **NOT hardware-agnostic** — each targets specific hardware
  (CUDA Triton kernels, FP8, multi-GPU PP/TP). ~10 reproduce on CPU; the rest need the
  author's GPU.
- Honest flags: author numbers are usually a **wider scope** (E2E serving) than our
  isolated op. Every mismatch carries a `NOT 1:1` banner + the exact fix to make it 1:1.
  7 PRs are feature/correctness-only with no real speedup — marked N/A, not faked.

### GPU box (RTX 5070 Ti, sm_120, wheel mode) — 2026-11-05

Measured in Docker, 5 repeats per arm (mean ± sd), author's own benchmark command.
`claude` = blind (objective without the PR's solution), `claude + PR's solution` = guided.

| PR | metric | baseline | PR's solution | ours | author | claude | claude + PR's solution |
|---|---|---|---|---|---|---|---|
| [#37225](https://github.com/vllm-project/vllm/pull/37225) top-k search | topk_whole b2048 v32k, ms | 1.259 ± 0.003 | 0.958 ± 0.003 | **1.31×** | 1.38× | **0.839 ± 0.166, PASS** (40 min) | **0.778 ± 0.037, PASS** |
| [#33538](https://github.com/vllm-project/vllm/pull/33538) Triton top-k/top-p | topk_whole b16 v32k, ms (10 repeats) | 1.222 ± 0.020 (PyTorch sort) | 0.093 ± 0.029 | **13×** | 4.2× | rerun pending | rerun pending |
| [#41163](https://github.com/vllm-project/vllm/pull/41163) AllPool.forward | median, µs | 120.2 ± 19.7 | 84.4 ± 17.3 | **1.42×** | 1.51× | **12.1 µs, PASS** | 12.1 µs, PASS |
| [#22940](https://github.com/vllm-project/vllm/pull/22940) slot mapping | opt-125m, output tok/s | 2270 ± 12 | 2280 ± 16 | +0.4 % | +2.07 % | 2280, PASS | 2282, PASS |

- #37225: with a 40-minute budget the blind claude arm rewrote the Triton search (611 added
  lines, kernel file only) and beats the PR's solution: 0.839 vs 0.958 ms on top-k, 0.795 vs
  1.590 ms on top-k+top-p; 89 hidden tests pass. With a 15-minute budget it produced nothing.
- #41163: both claude arms beat the PR's solution ~7× by also caching the first-token
  indices on the CPU (no device sync at all); hidden pooling test passes.
- #22940: GPU-bound workload on this box; the slot-mapping CPU gain is inside the noise.
- #33538 baseline = the script's PyTorch sort path (the PR introduces both kernel and script).

## <img src="web/shots/h/methodology.svg" height="40" alt="Methodology">

Two axes per number: **who** (measured by us / reported by author) × **which code**
(baseline / solution). Four rows per experiment, two colored metric pairs (vs reported,
vs measured), diagonal fill. Full spec: `prompts/numeric_solution.md` (numbers) +
`prompts/benchmark.md` (harness).

## <img src="web/shots/h/run.svg" height="40" alt="Run">

```bash
./run.sh                 # CPU PRs, this box
./run.sh gpu_triton      # GPU box: wheel-mode tasks (37225, 33538, ...) — no host clone needed
./run.sh 41163 22940     # specific PRs
./run.sh workloads       # GPU box: cluster weights, run per-PR workload images
./serve.sh               # dashboard at http://localhost:8000/web/ (gzip, serves from root)
```

GPU tasks self-skip without CUDA. Weights download on-demand, clustered by model, never
committed. Mixtral (~90 GB) is gated + last.

### <img src="web/shots/h/wheel.svg" height="40" alt="GPU box: wheel mode (no source build, no host clone)">

`Dockerfile.gpu` builds vLLM from source with CUDA 12.1 (sm_86). On newer GPUs that
cannot run (RTX 5070 Ti is sm_120), so pure-python / Triton PRs run in **wheel mode**:
`tasks/pr_<n>/metadata.json` carries `"wheel": "<vllm version>"` = the first PyPI
release after `merge_sha`, and the per-PR `tasks/pr_<n>/workload/Dockerfile`
(generated by `gen_workloads.py`, shas + wheel baked in) installs that wheel and
shallow-clones the PR's own `base_sha` / `merge_sha` trees **inside the image**
(`/base`, `/gold`). `run_one.sh` then swaps each arm's `.py` tree over the wheel's
site-packages (compiled `.so` stay), runs the oracle from a vllm-free copy of
`tests/`, and times the author's bench command `measure.repeats` times per arm
(`parse_workload.py` → `experiments[]` for the dashboard). Images also carry `gcc`
(Triton JIT) and `tblib` (vLLM `tests/conftest.py`). Every `workload/` also carries
`workload.json` + `WORKLOAD.md` (regenerated by `gen_workloads.py`): the canonical,
self-contained description of that PR's experiment — image, exact cmd and env, the
measurement policy as `run_one.sh` implements it, metrics, the author's reported
numbers and hardware, our reproduced numbers, the `BASELINE_GAP.md` verdict, and
`docker build` / `docker run` lines that re-run one arm without `run_one.sh`.

The host needs only docker + the NVIDIA container toolkit. `VLLM_GIT` is only for
legacy source mode.

### <img src="web/shots/h/dashboard.svg" height="40" alt="Public dashboard (this GPU box)">

`formula-dashboard.service` runs `serve.py` on :8000; Tailscale Funnel mounts it at
`https://station-wsl-michael.taila4786c.ts.net/formula/web/`
(`tailscale funnel --bg --https=10000 --set-path /formula http://127.0.0.1:8000`).
`data/area.json` (`gen_area.py`) keeps the first paint small over the slow Funnel link.
