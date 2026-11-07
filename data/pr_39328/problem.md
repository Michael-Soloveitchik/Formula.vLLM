# Task PR #39328 — [Core] Cache InductorPass.hash_source with functools.cache (#39328)

## Objective
Optimize the code in `./vllm` so the hidden correctness tests PASS **and** the
workload runs **as fast as or faster than the PR's solution**.
You are given the PR author's known solution below. You may reproduce it, but you
are explicitly asked to **try to beat it** — find a faster correct version.

## The problem (issue)
(from `solution.md`, quoted)

> InductorPass.hash_source() called inspect.getsource() on every source argument every time it ran. inspect.getsource() reads and parses the source file off disk for each class/function, which is costly. Because source code never changes at runtime, this work was repeated redundantly on every pass uuid() computation.

### Issue — detailed

> InductorPass.hash_source() looped over srcs, branching on type and calling inspect.getsource(src) or inspect.getsource(src.__class__) for functions/types/instances, then hashing with sha256. No memoization, so identical source was re-read and re-parsed on every call across all pass uuid() computations.

## PR's solution (reference — what was merged)
**Fix in words** (from `solution.md`):

> Diff extracts hashing into module-level _hash_source_cached(*srcs) decorated with @functools.cache, which does the sha256 over string-or-getsource inputs. hash_source() now builds a hashable cache_key by resolving object instances to src.__class__ (strings/types/functions pass through unchanged) then delegates to the cached helper. After first call per unique type/string tuple, inspect.getsource() is skipped entirely on cache hit. Result: cold compile dropped 34.10s→29.70s (1.15x) and cache lookup 32.50ms→11.30ms (2.88x).

**Author's words:**

> Author notes hash_source() repeatedly calls inspect.getsource() though source doesn't change at runtime. They extract hashing into a module-level _hash_source_cached() helper decorated with functools.cache, resolving object instances to their __class__ to form a hashable cache key, eliminating repeated inspect.getsource() calls across all pass uuid() computations.

## Workload + experiments (what will be measured)
Each experiment is timed before-merge (baseline) and after-merge (the PR's solution). Your
edit is timed on the SAME workload. Beat the PR's solution on these metrics:
- **cache_lookup** (ms): reported 32.5 -> 11.3 ms. workload: `python3 /task/bench.py`
- **cold_compile** (s): reported 34.1 -> 29.7 s.
- note: Real InductorPass.hash_source, 10000 lookups cycling 5 source objects after one untimed call each; total ms per 10000 lookups (author's scale), median of 5. cold_compile (Llama-3-70B TP4 torch.compile) is reported-only.

## Reported numbers (PR author)
- gain: 1.15x, 2.88x (pct)
- evidence: Test Result - 4/4 tests passed in test_pass_manager.py - E2E test passed (cold compile 29.5s, correct output) - Cold compile time: 29.70s ± 0.50s (baseline 3
- reported = numbers from the PR body (author's machine)

## Rules
- Edit only files under `./vllm`.
- Keep behavior correct (hidden pytest must pass).
- Faster than the PR's solution on the measured experiments wins. Same speed = ties the PR's solution.
