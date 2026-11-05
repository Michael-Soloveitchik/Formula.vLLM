# Formula.vLLM

202 real vLLM perf PRs, each a benchmark task. Did the PR's speedup reproduce?

![dashboard](web/shots/dashboard_wide.png)

## Goal

Turn merged vLLM perf PRs into reproducible tasks. Pre-merge code = baseline.
Merged code = author's solution. Measure both, side by side, in the PR's own unit.
Check the claimed gain is real.

## Example

PR #21222 (block-pool queue ops): author claimed 19×. Our isolated microbench shows
~2× — because the 19× was a whole-engine aggregate, not the one op. Both numbers are
genuine; different scope. The dashboard shows both and names the gap.

## Inspiration

FormulaCode. One isolated image per task; baseline vs solution timed back-to-back in
the SAME container (paired A-B-B-A); repeat for a noise distribution; count only gains
that clear a MAD gate.

## Partial results

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

## Methodology

Two axes per number: **who** (measured by us / reported by author) × **which code**
(baseline / solution). Four rows per experiment, two colored metric pairs (vs reported,
vs measured), diagonal fill. Full spec: `prompts/numeric_solution.md` (numbers) +
`prompts/benchmark.md` (harness).

## Run

```bash
export VLLM_GIT=/path/to/vllm
./run.sh                 # CPU PRs, this box
./run.sh workloads       # GPU box: cluster weights, run per-PR workload images
./serve.sh               # dashboard at http://localhost:8000/web/
```

GPU tasks self-skip without CUDA. Weights download on-demand, clustered by model, never
committed. Mixtral (~90 GB) is gated + last.
