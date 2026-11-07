# PR #39328: [Core] Cache InductorPass.hash_source with functools.cache

- field: scheduling  | area: compilation/passes
- author: frgossen
- state: MERGED  created: 2026-04-08T18:23:23Z  merged: 2026-04-20T18:06:15Z
- url: https://github.com/vllm-project/vllm/pull/39328
- labels: ready

## Body

## Purpose
InductorPass.hash_source() calls inspect.getsource() on every pass
class each time it's invoked, which is expensive. Since source code
doesn't change at runtime, extract the hashing into a module-level
_hash_source_cached() helper decorated with functools.cache. Object
instances are resolved to their __class__ for a hashable cache key,
eliminating repeated inspect.getsource() calls across all pass uuid()
computations.

## Test Plan
- tests/compile/passes/test_pass_manager.py (covers pass uuid computation)
- E2E: meta-llama/Meta-Llama-3-70B-Instruct, TP=4, cold compile

## Test Result
- 4/4 tests passed in test_pass_manager.py
- E2E test passed (cold compile 29.5s, correct output)
- Cold compile time: 29.70s ± 0.50s (baseline 34.10s ± 0.30s, 1.15x speedup)
- Cache lookup time: 11.30ms ± 0.30ms (baseline 32.50ms ± 1.05ms, 2.88x speedup)

## Comments

(no discussion comments)