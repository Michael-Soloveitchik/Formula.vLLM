# Why our absolute baselines differ from the PR authors' — and what to change

Scope: the six PRs whose *absolute* reproduced numbers diverge from the author's table
even where the *ratio* reproduces. Evidence = `results/<pr>.json`, `.work/pr_<pr>/wl_*.log`,
`tasks/pr_<pr>/metadata.json`, the PR bodies under `data/prs/`, and a machine snapshot
taken 2026-11-07 (`python3 machine.py snapshot`). No workload was re-run for this analysis.

## The box (what every number below was measured on)

| | ours | authors (vLLM maintainers) |
|---|---|---|
| GPU | RTX 5070 Ti 16 GB, sm_120, 70 SMs, 896 GB/s | unstated; 22940 used `CUDA_VISIBLE_DEVICES=7`, 29499 used `-dp 8` → 8-GPU datacenter node, H100/H200-class (3.35-4.8 TB/s, 132 SMs) |
| OS / CUDA path | **WSL2** (kernel 6.18 microsoft-standard), driver 591.86, CUDA 13.1 via `/dev/dxg` | native Linux |
| GPU also drives | the Windows desktop (`Disp.A On`, Xwayland) + an idle ComfyUI server (pid 2903, 260 MiB) | nothing |
| CPU | i7-14700KF hybrid (8 P + 12 E cores), WSL2 sees 12 floating vCPUs | server Xeon/EPYC, all cores equal |
| GPU idle state | P8, SM 180 MHz, ramps to 3090 MHz under load; `SW Power Capping` counter 1.3 s | datacenter cards sit at base clock |
| Load at snapshot | load1 1.3 (claude/opencode/node), 0 % GPU util | dedicated |

Two consequences of WSL2 dominate everything launch-bound: (1) every kernel launch and every
host↔device sync (`.item()`, `.tolist()`, `synchronize`) goes through dxgkrnl and costs roughly
3-10× what it costs on native Linux; (2) WSL2's vCPUs are ordinary Windows threads that the
Windows scheduler may park on an E-core, so a single-threaded Python loop runs at one of two
speeds (~1.8× apart). Bandwidth-bound work is not affected by WSL2 — only by the 3.7× lower
memory bandwidth of the card.

## Per-PR verdicts

### 37225 — top-k ternary search (`benchmark_topk_topp.py`) — **hardware**
Reported 0.761→0.550 ms, ours 1.255→0.951 ms (b2048 v32k); all five reported rows are 1.65-2.0×
slower here, ratio-of-ratios 0.94-0.95 → **reproduced**. The Triton kernel at b2048 reads the
whole logits matrix per search iteration, i.e. bandwidth-bound; 896 GB/s vs ~3.35 TB/s predicts
≈3.7×, the observed 1.65-2× says the kernel is not saturating either card, but the direction and
size are hardware. Args match the author exactly (`--batch-sizes 128 2048 --vocab-sizes 32768
131072 --warmup-iters 5 --benchmark-iters 20`). The b128 rows have 10-30 % sd (0.0098/0.1018,
0.025/0.083): 5 warmup iters × ~0.1 ms is far too little for a P8-idling card to boost — see
recommendation 6. Cannot be made 1:1; the ratio is the comparable quantity.

### 41163 — `AllPool.forward` slicing — **contention / WSL2 host overhead** (not hardware)
Reported median 59.89→39.65 µs (author's runs 59.4-60.2, 39.5-39.7: sd < 1 %). Ours
106→78.5 µs with sd 23 / 16 µs (22 %). Per-run medians are **bimodal**: baseline
[89, 107, 91, 72, 132], PR's solution [71, 72, 71, 131, 88] — both arms touch ~71 and ~131 µs, a 1.85×
step. The benchmark is wall-clock around 4000 calls that each `.tolist()`/sync, so it measures
host launch+sync latency, not GPU work. Our *floor* (71 µs) is identical for baseline and PR's solution: the
PR's saving (fewer GPU bytes) is entirely hidden under WSL2 launch/sync latency, and the 1.35×
"gain" we record is an artefact of which runs landed on the slow mode. Verdict: box, not GPU —
P/E-core placement of the WSL2 vCPU and/or other tenants. Bench script is the author's verbatim
(64 × 128 tokens, subgroup 32, hidden 4096, bf16, 500 warmup, 5 × 4000). Fix: recommendations
1-5; also run baseline / PR-solution interleaved (ABBA) so both arms sample the same mode.

### 33538 — Triton top-k/top-p sampler vs PyTorch sort — **WSL2 launch latency (box), small-batch rows**
Baseline arm cannot run the script (it is introduced by the PR); baseline = the script's own
"PyTorch (ms)" column in the PR's-solution run (by design, `metadata.json:base_note`). Reported PyTorch
0.28 ms at b16 v32k; ours 1.21 ms. Our PyTorch column is **flat in batch**: 1.14 ms at b128,
1.23 ms at b16, 0.23-0.28 ms at b1 — i.e. ~1.1 ms of fixed per-call overhead for the ~20-30
kernels the sort/cumsum/mask path launches, vs the author's floor of ~0.10 ms (their b1 row).
That is a 10× per-launch gap, exactly the WSL2 signature. The Triton arm (3-4 launches) is only
1.6× off the author (0.067 → 0.109 ms), so our 11× ratio vs their 4.2× overstates the PR at small
batch. Large-batch rows (b2048: 20.4 / 81.5 ms PyTorch) are bandwidth-bound and are the honest
comparison. Args are a superset of the author's (`--batch-sizes 1 16 128 2048`; author also ran
4 24 32 48 56). Cannot be made 1:1 on WSL2; report b2048 rows as headline or note the floor.

### 22940 — `compute_slot_mapping` direct indexing — **hardware (CPU share hidden) + config**
Author: `--num-prompts 30000`, 268→274 req/s (+2.07 %, single run, no sd) on an 8-GPU node;
ours `--num-prompts 2000`, 30.1→30.2 req/s (+0.2 %, sd 0.7 %). The PR saves ~400 µs of host
work per step; it is only visible when the step is host-bound. On the 5070 Ti the same opt-125m
step is ~9× slower on the GPU side (26 K vs 235 K tok/s), so the host saving is <0.3 % of step
time — consistent with what we measured. Secondary config gap: vLLM 0.10.1
(`vllm/engine/arg_utils.py:1633`) defaults `max_num_seqs=1024, max_num_batched_tokens=16384`
when the GPU has ≥ 70 GiB (and is not an A100), else `256 / 8192`, so the author's slot-mapping
loop ran over 4× more requests and 2× more tokens per step. To match the author's args: add
`--max-num-seqs 1024 --max-num-batched-tokens 16384` and `--num-prompts 30000` (≈16 min/run;
1 repeat per arm is enough, the metric is already an average over 30000 requests). Even then
expect ≤1 % here. Verdict: hardware; the author's 2 %
is below any reasonable noise gate anyway.

### 13365 — Numba n-gram matcher — **workload mismatch (call count), per-call speed is 1:1**
Reported 54.3 s → 1.9 s *cumulative over a 5K-request ShareGPT run*: `propose()` runs once per
decode step per request (~1.2 M calls at ~250 output tokens each), so 54.3 s ≈ 45 µs/call.
Ours: `tasks/pr_13365/bench.py` calls `propose()` **once per sequence** (5000 calls), 0.165 s ≈
33 µs/call — the same order as the author's per-call cost. The 1.9 s on the author side also
includes the one-off Numba JIT (~0.5-1 s) which our bench deliberately warms up outside the
timed region, which is why our ratio (63×) exceeds theirs (28×). To match: time the cumulative
cost over decode-step-like calls, e.g. for each of the 5000 sequences call `propose(arr[:t], n, k)`
for `t` in `range(prompt_len, len(arr))` (≈1.2 M calls), and either include the JIT in the PR's-solution
total or report it separately. Hardware is irrelevant (CPU-only, CPU-scaled 1:1).

### 29499 — EPLB numpy rewrite — **workload mismatch (micro-bench regime) + unit label bug**
Author: 27 % on `rearrange` (1.37 s → 1.08 s) inside a live `-dp 8` Qwen3-Next-80B EPLB
rebalance (58 MoE layers × 512 physical experts, real P2P), plus "20-25 % per primitive"
unquantified. Ours: three inlined primitives at `NUM_LOCAL=16`: `get_ep_ranks_with_expert`
8.2× faster (consistent with the author), but `move_to_buffer` 0.74× and `move_from_buffer`
0.38× — at 16 local experts the pure-Python body is 16×16 = 256 trivial iterations (23 µs) while
the numpy body pays ~10 numpy-call overheads (31 µs); numpy only wins once `num_local_experts`
and the per-layer loop are large. The headline/median therefore reports a regression the author
never claimed. Also: `bench.py:_best_ms` returns **ms** (`*1e3`) but `metadata.json` declares
`unit: "us"` — every 29499 number on the dashboard is mislabelled by 1000×. To match the author:
bench the whole per-layer rebalance (loop over all `ep_rank`s × 58 layers, i.e. the
`rearrange_expert_weights_inplace` index-mapping path with `num_local_experts` ≥ 32 as in
Qwen3-Next with 32 redundant experts), make `get_ep_ranks_with_expert` or the E2E loop the
headline, and fix the unit to `ms`. Verdict: workload, not hardware (CPU-only).

## Summary table

| PR | reported baseline → PR's solution | ours baseline → PR's solution | ratio repro? | gap cause |
|---|---|---|---|---|
| 37225 | 0.761 → 0.550 ms | 1.255 → 0.951 ms | yes (0.95) | hardware (bandwidth); b128 rows under-warmed |
| 41163 | 59.9 → 39.7 µs | 106 → 78.5 µs ± 22 % | noise | box: WSL2 sync floor 71 µs + bimodal CPU mode |
| 33538 | 0.28 → 0.067 ms | 1.21 → 0.109 ms | overstated at b16 | box: WSL2 launch floor ~1.1 ms on PyTorch path |
| 22940 | +2.07 % (30000 prompts) | +0.2 % (2000) | no | hardware hides host saving; author's defaults were `--max-num-seqs 1024 --max-num-batched-tokens 16384` |
| 13365 | 54.3 → 1.9 s (1.2 M calls) | 0.165 → 0.0026 s (5 K calls) | per-call yes | workload: call count; JIT excluded |
| 29499 | ~27 % E2E | median 0.74× | no | workload: n=16 micro-bench regime; `us` label is really `ms` |

## Recommendations for `run_one.sh` (not applied — files are being edited concurrently)

1. **Pin the device explicitly**: `GPU_FLAG='--gpus "device=0"'` and pass
   `-e CUDA_VISIBLE_DEVICES=0 -e CUDA_DEVICE_ORDER=PCI_BUS_ID`. Harmless on a 1-GPU box, and
   it stops a second GPU ever being picked up silently when the box changes.
2. **Record a machine snapshot per arm**: call `python3 machine.py snapshot -o "$WORK/machine_${arm}_${i}.json"`
   right before each `run_in_img` in `run_workload`, and have `record.py` copy the first one
   into `results/<pr>.json:machine`. Then any 41163-style bimodality can be matched to GPU
   util / other tenants / load1 after the fact instead of guessed.
3. **Refuse (or flag) a contended run**: before `run_workload`, if `nvidia-smi
   --query-gpu=utilization.gpu` > 5 % or any compute process other than the idle ComfyUI server
   (pid 2903) holds > 1 GiB, or `load1` > `nproc/2`, either wait (`sleep 30` loop, bounded) or
   write `"contended": true` into `arms.json` so the dashboard greys the row.
4. **Interleave arms ABBA instead of all-baseline-then-all-PR-solution**: today `run_workload baseline` runs 5×
   then `run_workload PR-solution` 5× (37225: 12:55-12:58 baseline, 12:59-13:03 PR's solution). Drift across those
   8 minutes lands entirely on one arm. Loop `for i in 1..REPEATS: run A; run B; run B;
   run base` (one extra container start per pair) so every repeat sees the same clock/tenant
   state. This is the single biggest fix for 41163.
5. **Priority and CPU placement**: wrap the container in `nice -n -5` (needs `CAP_SYS_NICE`
   /sudo; without root use `nice -n 0` and at least `chrt --other 0`) and pass
   `--cpuset-cpus=0-7` to `docker run` so the bench thread does not share a vCPU with the host's
   claude/node processes. Note: inside WSL2 a cpuset pins to a *vCPU*, not a physical P-core —
   the P/E placement is decided by Windows. The only real fix is on the Windows side:
   set the `vmmem`/`vmmemWSL` process affinity to P-cores, or Windows power plan "High
   performance"; say so in the README rather than pretending `taskset` solves it.
6. **Warm the GPU out of P8 before timing**: add a 2-3 s busy matmul (`python3 -c "import torch;
   a=torch.randn(8192,8192,device='cuda'); [a@a for _ in range(50)]; torch.cuda.synchronize()"`)
   at the end of `PRELUDE`, and raise `--warmup-iters` from 5 to ≥ 50 for the topk benchmarks
   (both 33538 and 37225 accept it; the author's 5 is tuned for a card that never idles at
   180 MHz). This removes most of the 10-30 % sd on the b128 rows.
7. **Lock clocks — needs root, and not possible from WSL2**: `nvidia-smi -lgc 2500,2500` /
   `-lmc` require admin and on WSL2 must be run from the Windows host (`nvidia-smi.exe` as
   Administrator); inside the guest it is a no-op. Document it as a manual pre-step; do not
   script it in `run_one.sh`.
8. **More repeats for launch-bound benches**: `REPEATS=5` with 22 % sd (41163) gives a ±20 %
   CI; for `microbench_in_pr` tasks that time < 1 ms per call, set `REPEATS=9` and report the
   **median** of per-run medians (the author's own statistic) rather than mean ± sd.
9. **Docker overhead is not the cause**: container start (~1 s) is outside every timed region
   (all benches time inside the process); `--shm-size=2g` and the ro bind mounts do not touch
   the hot path. No change needed; mention it once in the README so the question stops coming up.
10. **Per-task arg fixes (metadata.json, not run_one.sh)**: 22940 add `--max-num-seqs 1024
    --max-num-batched-tokens 16384` (and optionally `--num-prompts 30000` with `REPEATS=1`); 13365 replace the per-sequence call
    with per-decode-step calls (≈1.2 M) and report JIT separately; 29499 fix `unit` to `ms`,
    bench at `num_local_experts ≥ 32` over all ranks × layers, and make `get_ep_ranks_with_expert`
    (or the E2E loop) the headline; 33538 make a b2048 row the headline (b16 is launch-floor-bound).

## What cannot be closed on this box

- Absolute latency of anything bandwidth-bound (37225 b2048, 33538 b2048): 896 GB/s vs 3.35 TB/s.
  Ratios reproduce; absolutes never will.
- Any benchmark whose per-call time is < ~100 µs on the author's card (41163, 33538 b1-b128): the
  WSL2 launch/sync floor (~70 µs per sync, ~30-50 µs per launch burst) is larger than the thing
  being measured. The fix is a native-Linux GPU host, not runner tweaks; until then, treat those
  rows as "ratio unreliable, absolute meaningless" and say so on the dashboard.
