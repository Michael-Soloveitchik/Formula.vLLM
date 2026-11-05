# PR #39328: [Core] Cache InductorPass.hash_source with functools.cache

- field: gpu_kernels  | area: compilation_passes
- author: frgossen
- state: MERGED  created: 2026-04-08T18:23:23Z  merged: 2026-04-20T18:06:15Z
- url: https://github.com/vllm-project/vllm/pull/39328
- labels: ready

# Issue

## Problem

InductorPass.hash_source() called inspect.getsource() on every source argument every time it ran. inspect.getsource() reads and parses the source file off disk for each class/function, which is costly. Because source code never changes at runtime, this work was repeated redundantly on every pass uuid() computation.

## Example

On meta-llama/Meta-Llama-3-70B-Instruct with TP=4 cold compile, the repeated getsource() calls inflated cache-lookup time to ~32.5ms and total cold compile to ~34.1s.

## Use case

Happens during torch.compile graph-pass setup at model load / cold compile, when each custom inductor pass computes its UUID from hashed source.

## Resources

CPU and kernel-launch/compile time — inspect.getsource() does disk reads and source parsing, adding latency to the compilation path.

## Type of users

Users running vLLM with torch.compile enabled, especially large multi-GPU TP deployments (e.g. 70B, TP=4) where cold-compile latency matters.

## Kind of inputs

Any workload triggering graph compilation with multiple custom inductor passes; cost scales with number of passes and uuid() invocations, independent of request shape.

## Mechanism with the issue

InductorPass.hash_source() looped over srcs, branching on type and calling inspect.getsource(src) or inspect.getsource(src.__class__) for functions/types/instances, then hashing with sha256. No memoization, so identical source was re-read and re-parsed on every call across all pass uuid() computations.

# Solution

## Fix in words

Diff extracts hashing into module-level _hash_source_cached(*srcs) decorated with @functools.cache, which does the sha256 over string-or-getsource inputs. hash_source() now builds a hashable cache_key by resolving object instances to src.__class__ (strings/types/functions pass through unchanged) then delegates to the cached helper. After first call per unique type/string tuple, inspect.getsource() is skipped entirely on cache hit. Result: cold compile dropped 34.10s→29.70s (1.15x) and cache lookup 32.50ms→11.30ms (2.88x).

## Author's solution

Author notes hash_source() repeatedly calls inspect.getsource() though source doesn't change at runtime. They extract hashing into a module-level _hash_source_cached() helper decorated with functools.cache, resolving object instances to their __class__ to form a hashable cache key, eliminating repeated inspect.getsource() calls across all pass uuid() computations.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/compilation/passes/inductor_pass.py`

### Diff
```diff
diff --git a/vllm/compilation/passes/inductor_pass.py b/vllm/compilation/passes/inductor_pass.py
index 4610c62d1771..b54c7bfa14d0 100644
--- a/vllm/compilation/passes/inductor_pass.py
+++ b/vllm/compilation/passes/inductor_pass.py
@@ -51,6 +51,15 @@ def pass_context(compile_range: Range) -> Generator[None, None, None]:
         _pass_context = prev_context
 
 
+@functools.cache
+def _hash_source_cached(*srcs: str | type | types.FunctionType) -> str:
+    hasher = hashlib.sha256()
+    for src in srcs:
+        src_str = src if isinstance(src, str) else inspect.getsource(src)
+        hasher.update(src_str.encode("utf-8"))
+    return hasher.hexdigest()
+
+
 class InductorPass(CustomGraphPass):  # type: ignore[misc]
     """
     A custom graph pass that uses a hash of its source as the UUID.
@@ -72,19 +81,16 @@ def hash_source(*srcs: str | Any) -> str:
         Utility method to hash the sources of functions or objects.
         :param srcs: strings or objects to add to the hash.
         Objects and functions have their source inspected.
+        Results are cached by resolved types to avoid repeated
+        inspect.getsource() calls.
         :return:
         """
-        hasher = hashlib.sha256()
-        for src in srcs:
-            if isinstance(src, str):
-                src_str = src
-            elif isinstance(src, (types.FunctionType, type)):
-                src_str = inspect.getsource(src)
-            else:
-                # object instance
-                src_str = inspect.getsource(src.__class__)
-            hasher.update(src_str.encode("utf-8"))
-        return hasher.hexdigest()
+        # Resolve instances to their class for a hashable cache key.
+        cache_key = tuple(
+            src if isinstance(src, (str, type, types.FunctionType)) else src.__class__
+            for src in srcs
+        )
+        return _hash_source_cached(*cache_key)
 
     @staticmethod
     def hash_dict(dict_: dict[Any, Any]) -> str:

```
