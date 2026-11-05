# Task PR #39328 — [Core] Cache InductorPass.hash_source with functools.cache (#39328)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the expert's merged patch (gold)**.
You are given the expert's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
A compilation step computes a unique identifier for each pass by reading and hashing the pass's source code. It re-reads and re-hashes that source every single time the identifier is needed. The source never changes while the program runs, so this repeated work is wasted and slows down compilation.

### Issue — detailed
InductorPass.hash_source() in vllm/compilation/passes/inductor_pass.py calls inspect.getsource() on each function, type, or object instance passed to it, then feeds the result into a hashlib.sha256 hasher. inspect.getsource() reads and parses source files, which is expensive. This runs on every pass uuid() computation, so the same sources get inspected and hashed repeatedly. Because pass source code is fixed at runtime, this repeated inspection is pure overhead with nothing caching the results.

## Expert solution (reference — the merged PR)
**General:** The hashing work is pulled into a separate helper that remembers its results. The first call for a given input computes the hash; later calls with the same input return the stored result without re-reading source. Object instances are converted to their class first so the inputs form a stable lookup key.

**Detailed:** A new module-level function _hash_source_cached() is added and decorated with @functools.cache; it holds the sha256 loop that calls inspect.getsource() and returns the hex digest, running only on a cache miss. hash_source() no longer hashes inline; it builds a hashable cache_key tuple where each argument passes through if it is a str, type, or FunctionType, else resolves to src.__class__, then delegates to _hash_source_cached(*cache_key). Resolving instances to their class makes the key hashable and lets functools.cache dedupe, so repeated calls with the same resolved key skip inspect.getsource() entirely without changing the produced hash.

**Author's words:** InductorPass.hash_source() calls inspect.getsource() on every pass class each time it's invoked, which is expensive. Since source code doesn't change at runtime, extract the hashing into a module-level _hash_source_cached() helper decorated with functools.cache. Object instances are resolved to their __class__ for a hashable cache key, eliminating repeated inspect.getsource() calls across all pass uuid() computations.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (gold). Your
edit is timed on the SAME workload. Beat gold on these metrics:
- **cache_lookup** (ms): reported 32.5 -> 11.3 ms. workload: lookup ms
- **cold_compile** (s): reported 34.1 -> 29.7 s. workload: compile s

## Reported numbers (PR author)
- metric: cold compile time and cache lookup time (mean ± stddev) vs baseline
- value: cold compile 29.70 ± 0.50 vs baseline 34.10 ± 0.30 (1.15x); cache lookup 11.30 ± 0.30 vs baseline 32.50 ± 1.05 (2.88x)
- unit: seconds (compile); milliseconds (cache lookup); x speedup

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than gold on the measured experiments wins. Same speed = ties gold.
