# interesting_prs.md — start here

Small, CPU/agnostic, pure-python targets with a clear reported gain and an
isolable test oracle. Good first PRs to prove the pipeline on an 8 GB 3060 Ti.

| PR | field | reported gain | base_sha → merge_sha | source |
|---|---|---|---|---|
| **21222** | kv_cache | 95% (0.15ms→0.008ms) | `10904e6d75` → `ed25054577` | block_pool.py / kv_cache_utils.py |
| **21627** | scheduling | 80% (35us→7us) | `601f856d56` → `ca9e2be3ed` | v1/engine/core.py |
| 6865 | scheduling | 40% | `d7a299edaa` → `6ca8031e71` | core/scheduler.py |
| 7117 | scheduling | 1.5% | `9fadc7b7a0` → `83c644fe7e` | output_processor/single_step.py |
| 8868 | scheduling | 0% (noise fix) | `803f37eaaa` → `efa9084628` | engine/metrics.py |

## Why these

- **21222** — only true torch-free CPU unit: `FreeKVCacheBlockQueue.append_n /
  popleft_n` bulk ops replace loop-of-singles. Already measured on CPU
  (~+298% log-speedup, see `data/pr_21222/notes.md`).
- **21627** — moves request conversion off the engine-core busy loop; torch-free
  target logic, test boots EngineCore (needs the built image).
- **6865 / 7117** — scheduler/output-processing micro-opts, torch-free source.
- **8868** — a 0%-gain "no regression" task: metrics log noise fix. Good
  negative control (should NOT show a speedup; tests must still pass).

## Tiers

- `cpu` set = {21222, 21627} — verified runnable with the CPU image.
- Everything else in the 202 needs the full GPU serving workload; the harness
  builds the image at that PR's base_sha and runs its real `test_patch`.

Run: `./run.sh interesting` or `./run.sh 21222 21627`.
