# Clustering prompt — field assignment

Source-of-truth for `extract.py`. Edit the `json` block below, rerun
`python3 pipeline/extract.py` to recluster. The script parses the FIRST
```json fence only.

## Intent
Cluster vLLM merged PRs by **efficiency contribution**. Exclude UI/frontend,
docs, CI/build, tests-only, pure refactors with no perf signal.

## Rules

### Scoring
- Match keywords (lowercased) against `title + changed file paths`.
- Weights encode intent vs incidence: **title hit = 2**, **path hit = 1**,
  **subject `[tag]` = 2**. (Touching a `.../quantization/` dir doesn't make a
  MoE refactor a quant PR — path is weight 1.)
- **Priority** breaks ties and stops the generic `gpu_kernels` bucket from
  swallowing specific work: a CUDA kernel for FP8 is `quantization`, not
  `gpu_kernels`. `gpu_kernels` is LAST so it wins only when nothing specific
  matches. Earlier in `priority` wins ties.
- Zero efficiency signal → `null` (excluded, non-contributing).

### Hard excludes (checked before scoring)
- Subject tag ∈ `exclude_tags` ([CI]/[Doc]/[Build]/[Frontend]/[Test]/[Chore]/
  [Deprecation]...) → `excluded_tag` (explicit non-efficiency intent).
- All changed files under `exclude_path_prefixes` (docs/ examples/ .github/) →
  `docs_examples_github_path`.

### Architecture → `platform` cluster (priority-FIRST, strong intent)
- Rationale: a hardware-port PR's perf is measured from an ARCHITECTURE POV,
  not a software-algorithm POV. A `[ROCm]` FP8 kernel makes AMD fast; it is not
  advancing the quant *algorithm*.
- **Trigger = a title ARCH-TAG only** ([ROCm]/[AMD]/[TPU]/[CPU]/[XPU]/[Intel]/
  [Gaudi]/[HPU]/[Neuron]/[CUDA]/[NVIDIA]/[s390x]...). Verb-intent ("add cuda
  graph support") is intentionally NOT a trigger — too noisy (cudagraph is
  software). HW merely *mentioned* in a body benchmark does NOT move a software
  PR here (that is the platform VIEW `data/views/`, not the cluster).
- Area is overridden to the platform name (`rocm_amd`/`tpu`/`cpu`/...) and
  subclustered BY platform, not by code dir.
- **Efficiency gate** (`arch_contrib`): a hardware tag alone is not a
  contribution. KEEP if the title has a `perf_indicator` OR an enablement verb
  (support/enable/implement/add/introduce/integrate/backend/kernel/optimize/
  accelerate/fuse). DROP as `arch_non_contrib` if the title is a version
  bump/pin/upgrade/downgrade/cleanup/rename/revert/dockerfile/typo/comment/
  disable-test/flaky/backport, OR a bugfix-tagged PR with no enablement/perf
  verb.

### Non-platform guards & recovery
- **Infra guard**: if the dominant code area ∈ `exclude_areas` (tests/docker/
  requirements/cmake/.github/tools/benchmarks/docs) → `infra_area`. Catches
  stray-keyword infra PRs ("[CI] stabilize ROCm", "bump flashinfer").
- **Perf recovery**: field `null` with reason `no_efficiency_keyword`, a
  `perf_indicator` in the title, and a non-infra area → routed to the field its
  code lives in via `area_field_fallback` (`recovered_perf_area`). Rescues real
  perf work phrased without a listed keyword.
- **Manual body-audit drop list** (`exclude_prs`): subagents READ conversation
  bodies; PRs that keyword-matched a field but whose body confirms NO efficiency
  contribution (enablement-only "perf later", pure refactor, metric-counting,
  version bump) → `manual_body_audit`, parked under `data/prs/_excluded/`.
  Editable to re-include.
- **Unresolvable PRs** (`unfetchable_prs`): a git commit tagged `(#N)` with no
  such GitHub PR (early-history artifact) → `unresolvable_pr`.

### Field disambiguation (the tricky overlaps)
- `prefill_ttft` = prefill MECHANISM (chunked prefill, prefill scheduling, P/D
  disaggregation, TTFT metrics). A kernel/quant PR merely quoting "X% TTFT"
  stays in its code field — its kernel/quant score outweighs a lone ttft
  mention. `prefill_ttft` sits above `scheduling` so prefill-scheduler ties
  route to prefill.
- `decoding_tpot` folds decode-phase latency metrics (TPOT/TBT/ITL) with the
  decode work (sampler/spec-decode/logits). Mirrors `prefill_ttft`.
- `quantization`/`attention`/`kv_cache` sit above `gpu_kernels` so FP8/MLA/paged
  kernels route to the algorithm, not the generic kernel bucket.
- `tag_map` routes subject tags to fields ([Quantization]→quantization,
  [Kernel]→gpu_kernels, [Spec Decode]→decoding_tpot...); `[perf]`/`[misc]` map
  to `null` (a perf tag is not itself a field).

## Learnings (update as run proceeds)
- v8: **performance vs enabling agenda split** (pipeline/agenda.py → data/views/
  agenda.{jsonl,md}, surfaced in report.md). Split line = a MEASURED benchmark
  number (perf_scan → perf_gains.jsonl). Set by a subagent body-read audit:
  the no-number side audited 12/12 truly non-performance (note: an accuracy /
  MTEB win is NOT performance → still enabling), but a bare prose "faster/
  optimize" keyword was only ~33% precise (dir names like .../optimization/,
  incidental "throughput" in a bugfix, re-enabling cudagraph). So prose alone
  does NOT promote to performance; it is kept as an advisory `perf_intent` flag
  on enabling PRs only. Result: performance=376, enabling=5599 (885 w/ unmeasured
  intent). Enabling subcats: correctness/enable_support/model_loading/refactor/
  metrics/oom_memory/numa_topology/other. agenda.py does NOT touch index.jsonl.
  Also improved perf_scan recall (355→376): (a) keep a round % (10/25/50/..%)
  when a WIN word sits in the same sentence ("~10% performance gain"); (b)
  relaxed arrow "7.87 -> 8.37 req/s" (unit only on 2nd operand) + normalize
  spaced units "req / sec"→req/s. Multi-block key:value benchmark dumps (e.g.
  #32770) still not parsed — acceptable miss.
- v1: gpu_kernels=3071 (35% of contributing) — too greedy. Generic kernel/cuda/
  rocm keywords absorbed attention+quant+moe PRs.
- v2: demote gpu_kernels to lowest priority; promote quantization/attention/
  kv_cache/decoding so specific intent wins. Expect gpu_kernels to shrink,
  specific fields to grow → better subcluster signal.
- v3: PRs whose DOMINANT changed area is infra (tests/docker/requirements/
  cmake/.github/tools/benchmarks) leaked in via stray keywords (e.g. "[CI]
  stabilize ROCm", "bump flashinfer"). These aren't engine-efficiency code.
  Added `exclude_areas`: if dominant area ∈ set → non-contributing.
- v5: rename `ttft`→`prefill_ttft` ("prefill(ttft)") and `decoding`→
  `decoding_tpot` ("decoding(TPOT)"). Folded decode-phase latency metrics
  TPOT/TBT/ITL into decoding_tpot (decode-side latency lives with the decode
  work: sampler/spec-decode/logits). Mirrors prefill_ttft on the prefill side.
- v7: `exclude_prs` — curated drop list from a BODY-READ audit (subagents read
  conversation bodies, not just titles). These are hardware/feature/bugfix/
  refactor PRs that keyword-matched a field but whose body confirms NO
  efficiency contribution (e.g. "enablement, will follow up with perf later",
  pure class-split refactor, metric-counting change, version bump). Honored in
  extract.py → reason="manual_body_audit", parked in data/prs/_excluded/. Edit
  the list to re-include a PR. Also tightened perf_scan HW regex to mask GPU
  COUNT notation ("8xB200", "4×GB200") that was misread as an 8x/4x speedup.
- v6: add `architecture` field. Rationale (user): perf of a hardware-port PR
  is measured from an ARCHITECTURE POV, not a software-algorithm POV. A
  `[ROCm]` FP8 kernel is not advancing the quant *algorithm*, it is making AMD
  fast. Trigger = title arch-TAG only (strong intent): [ROCm]/[AMD]/[TPU]/
  [CPU]/[XPU]/[Intel]/[Gaudi]/[HPU]/[Neuron]/[CUDA]/[NVIDIA]/[s390x]. HW merely
  *mentioned* in a body benchmark does NOT move a software PR here (that is the
  platform VIEW, data/views/platforms.md). Verb-intent ("add cuda graph
  support") intentionally NOT used — too noisy (cudagraph is software).
  architecture is priority-FIRST so an arch tag wins over code keywords.
  Subclustered BY platform (area override), not by code dir.
- v4: split `ttft` out of `scheduling` — prefill / first-token path is an
  important serving flow, deserves its own cluster. ttft field = prefill
  MECHANISM (chunked prefill, prefill scheduling, P/D disagg, ttft metrics).
  Kernel/quant PRs that merely quote "X% TTFT" stay in their code field
  (their kernel/quant keyword score outweighs a lone ttft mention). Removed
  prefill/chunked/ttft/disagg from scheduling keywords. Priority: ttft above
  scheduling so a tie on prefill-scheduler work routes to ttft.

```json
{
  "since": "2024-04-01",
  "fields": ["platform","kv_cache","memory","gpu_kernels","attention","decoding_tpot","quantization","parallelism","scheduling","prefill_ttft"],
  "priority": ["platform","quantization","attention","kv_cache","prefill_ttft","decoding_tpot","parallelism","scheduling","memory","gpu_kernels"],
  "architectures": {
    "rocm_amd":   ["rocm","\\bhip\\b","\\bamd\\b","aiter","mi\\d{2,}x?","gfx\\d+","navi"],
    "tpu":        ["\\btpu\\b","pallas","libtpu"],
    "xpu_intel":  ["\\bxpu\\b","intel gpu","oneapi","sycl","\\bintel\\b"],
    "gaudi_hpu":  ["gaudi","\\bhpu\\b","habana"],
    "neuron_aws": ["neuron","trainium","inferentia"],
    "cpu":        ["\\bcpu\\b","x86","aarch64","arm64","\\bppc\\b"],
    "cuda_nvidia":["\\bcuda\\b","nvidia","hopper","blackwell"],
    "s390x_ibm":  ["s390x","powerpc","ppc64","\\bpower\\b","\\bibm z\\b"]
  },
  "arch_tag_only": true,
  "keywords": {
    "kv_cache":   ["prefix cache","prefix-cache","kv cache","kv-cache","kvcache","block manager","block_table","paged","cache salt","cache reuse","prompt cache","automatic prefix","block pool","hybrid kv"],
    "memory":     ["memory","oom","vram","mem pool","allocator","offload","swap","cpu offload","footprint","fragmentation","sleep mode","weight loading","memory profiling"],
    "gpu_kernels":["kernel","cuda","triton","cutlass","rocm","hip","aiter","fused","custom op","csrc","flashinfer","gemm","elementwise","warp","ptx","torch.compile","torch compile","dynamo","inductor","piecewise cudagraph","compile mode","aot compil"],
    "attention":  ["attention","flash attn","flashattn","flash-attn","mla","gqa","sliding window","attention sink","rotary","rope","mamba","linear attn","chunked prefill attn","flex attention"],
    "decoding_tpot":["speculative","spec decode","mtp","sampling","sampler","logits","beam","eagle","medusa","ngram","draft model","guided decoding","structured output","structured decoding","logprob","tpot","tbt","itl","time per output token","inter-token","inter token","time between tokens","per output token","decode latency","decode throughput","output token throughput"],
    "quantization":["quant","fp8","int8","int4","w4a4","w8a8","w4a16","awq","gptq","gguf","marlin","bitsandbytes","mxfp","nvfp","compressed-tensors","smoothquant","kv cache fp8","per-token","per-channel"],
    "parallelism":["tensor parallel","pipeline parallel","expert parallel","data parallel","all-reduce","allreduce","all2all","all-to-all","nccl","distributed","collective","sharding","moe","context parallel","sequence parallel","ray "],
    "scheduling": ["scheduler","schedule","throughput","batching","continuous batch","preempt","priority queue","async scheduling","step overlap","cuda graph","piecewise","dbo"],
    "prefill_ttft":["ttft","time to first token","time-to-first-token","first token","prefill","chunked prefill","chunked-prefill","prompt latency","prefill schedul","long context prefill","disaggregat","p/d disagg","prefill decode","prefill/decode"]
  },
  "tag_map": {
    "quantization":"quantization","quant":"quantization","kernel":"gpu_kernels",
    "rocm":"gpu_kernels","cuda":"gpu_kernels","attention":"attention",
    "spec decode":"decoding_tpot","speculative decoding":"decoding_tpot","structured output":"decoding_tpot",
    "kv cache":"kv_cache","core":"scheduling","v1":"scheduling","perf":null,
    "misc":null,"hardware":"gpu_kernels"
  },
  "exclude_tags": ["doc","docs","documentation","frontend","ci","ci/build","build","test","tests","chore","typo","readme","example","examples","deprecation"],
  "exclude_path_prefixes": ["docs/","examples/",".github/"],
  "exclude_areas": ["tests","docker","requirements","cmake",".github","tools","benchmarks","docs"],
  "perf_indicators": ["[perf]","[performance]","optimiz","speedup","speed up","faster","overhead","regression","accelerat","reduce latency","reduce memory","fast path","efficien","less memory","lower latency","higher throughput"],
  "unfetchable_prs": [3974],
  "exclude_prs": [6447,10622,12038,12475,15889,27946,29960,38872,39291,7948,9639,40786,5524,10757,12711,12737,17699,21435,22949,30210,33994,11169,14667,15934,21848,34169,39478,41025,41932,42740,7735,34233,23736,32149,32772,4368,11463,17574,38799,18003,14189,39293,16562,21429,34876,29483,35326,3950,33535,41889,35189,32794,26130,27773,35478,42078,37324,15729,27424,5164],
  "area_field_fallback": {
    "csrc": "gpu_kernels",
    "compilation": "gpu_kernels",
    "model_executor/layers": "gpu_kernels",
    "attention": "attention",
    "v1/attention": "attention",
    "v1/sample": "decoding_tpot",
    "v1/spec_decode": "decoding_tpot",
    "spec_decode": "decoding_tpot",
    "distributed": "parallelism",
    "v1/kv_offload": "memory",
    "v1/core": "scheduling",
    "v1/engine": "scheduling",
    "v1/worker": "scheduling",
    "core": "scheduling",
    "worker": "scheduling"
  }
}
```
