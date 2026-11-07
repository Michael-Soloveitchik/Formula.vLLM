# Subclustering prompt — code area (dir tree)

Source-of-truth for `extract.py` area derivation. Edit `json`, rerun extract.
Subcluster = **which part of the code the PR contributed to**, from changed
file paths. One dominant area per PR (most-touched). Determines the
`data/pr_<n>/pr.md` (area kept in task.json; was the `data/prs/<field>/<area>/` tree).

## Rules
- `vllm/<sub>/...`: if `<sub>` ∈ `two_level` → area = `<sub>/<next>` (finer,
  e.g. `model_executor/layers`); else area = `vllm/<sub>`.
- `three_level_under` ({model_executor, v1}) MAY descend one more level, but a
  path segment containing a `.` is a FILE and stops descent (so area never
  points at a filename).
- `csrc/...` → `csrc/<sub>`; if the next segment is a file (has `.`) →
  `csrc/root` (CUDA/C++ kernels).
- Top-level infra dirs (`infra_dirs`: tests, benchmarks, docs, examples,
  .github, cmake, requirements, tools) kept as-is — they feed the clustering
  `exclude_areas` infra guard.
- No files → area `unknown`.
- Tie → most frequent path wins (`most_common(1)`).

## Edge cases
- The `platform` field OVERRIDES this area with the platform name (set in
  prompt_clustering), so platform PRs subcluster by hardware, not code dir.
- Area is slugged for the folder path: lowercased, non `[a-z0-9._-]`→`_`, max
  60 chars. A PR whose dominant file is a top-level `README.md` yields an area
  folder literally named `readme.md` — downstream counters must match only
  numeric `<pr>.md` files, never area dirs.

## Learnings
- v1: model_executor/layers dominates many fields (expected: layers = where
  attention/quant/moe kernels live). Keep two-level split so layers subdivides.

```json
{
  "two_level": ["model_executor","attention","distributed","v1","engine","worker","core","entrypoints","lora","multimodal","platforms","transformers_utils","compilation","spec_decode","device_allocator"],
  "three_level_under": ["model_executor","v1"],
  "infra_dirs": ["tests","benchmarks","docs","examples",".github","cmake","requirements","tools"]
}
```
