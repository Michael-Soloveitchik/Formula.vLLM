# PR #13365: [V1][Spec Decode] Optimize N-gram matching with Numba

- field: decoding_tpot  | area: requirements-common.txt
- author: WoosukKwon
- state: MERGED  created: 2025-02-17T00:00:28Z  merged: 2025-02-18T21:19:58Z
- url: https://github.com/vllm-project/vllm/pull/13365
- labels: ready, ci/build, v1

# Issue

## Problem

The V1 N-gram speculative decode proposer implements KMP subarray matching in pure Python, executing per-token loops in the interpreter. At large batch sizes this per-request matching work dominates CPU overhead, adding tens of seconds of cumulative latency across many requests. The Python-level loop over context tokens is the bottleneck, not the asymptotic complexity.

## Example

On the ShareGPT benchmark with 5K requests, cumulative N-gram matching overhead was 54.3 sec (a measurable fraction of total runtime) before the change.

## Use case

Serving with N-gram-based speculative decoding enabled in vLLM V1, where each decode step runs the proposer to find a matching n-gram in the context and draft the following k tokens.

## Resources

CPU compute and Python interpreter overhead; the hot loop runs on the host, not the GPU, so it steals wall-clock time from the decode critical path (TPOT).

## Type of users

Users running V1 speculative decoding (specifically N-gram prompt-lookup drafting) at high concurrency / large batch sizes.

## Kind of inputs

Large numbers of concurrent requests with long context sequences, where the per-request KMP scan over context_token_ids is repeated every decode step.

## Mechanism with the issue

NgramProposer._find_subarray_kmp and _kmp_lps_array ran as interpreted Python methods using Python int lists and element-wise while-loops over context_token_ids. Every token comparison incurs Python interpreter and boxing overhead, so scanning long contexts across thousands of requests accumulates large CPU time in the decode path.

# Solution

## Fix in words

The diff JIT-compiles the matching code with Numba: _kmp_lps_array and _find_subarray_kmp move from static methods on NgramProposer to module-level functions decorated with @jit(nopython=True), operating on numpy arrays (np.int32) instead of Python lists. The lps array becomes np.zeros(..., dtype=np.int32) and the NgramProposer.propose call delegates to the free compiled function. gpu_model_runner now instantiates the drafter only on the last PP rank and warms up the JIT by calling propose on a np.zeros(1024) dummy input at init (compilation takes under 1 sec). numba==0.60.0 is pinned in requirements-common.txt (v0.61 drops Python 3.9). Net effect: the hot matching loop runs as compiled native code, yielding a 20-30x speedup and cutting ShareGPT 5K-request overhead from 54.3 sec to 1.9 sec.

## Author's solution

The author optimized the N-gram matching algorithm by JIT compiling it with Numba, observing a 20-30x speedup at large batch sizes. On ShareGPT with 5K requests, cumulative overhead dropped from 54.3 sec to 1.9 sec (~2.5% of total runtime).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `requirements-common.txt`
- `vllm/v1/spec_decode/ngram_proposer.py`
- `vllm/v1/worker/gpu_model_runner.py`

### Diff
```diff
diff --git a/requirements-common.txt b/requirements-common.txt
index b7c94cbdba8b..c52980bc7df7 100644
--- a/requirements-common.txt
+++ b/requirements-common.txt
@@ -1,6 +1,7 @@
 psutil
 sentencepiece  # Required for LLaMA tokenizer.
 numpy < 2.0.0
+numba == 0.60.0 # v0.61 doesn't support Python 3.9. Required for N-gram speculative decoding.
 requests >= 2.26.0
 tqdm
 blake3
diff --git a/vllm/v1/spec_decode/ngram_proposer.py b/vllm/v1/spec_decode/ngram_proposer.py
index 9b116e00af97..33289d05dabd 100644
--- a/vllm/v1/spec_decode/ngram_proposer.py
+++ b/vllm/v1/spec_decode/ngram_proposer.py
@@ -1,14 +1,12 @@
 # SPDX-License-Identifier: Apache-2.0
-from typing import List, Optional
+from typing import Optional
 
 import numpy as np
+from numba import jit
 
 
 class NgramProposer:
 
-    def __init__(self):
-        pass
-
     def propose(
         self,
         context_token_ids: np.ndarray,
@@ -21,7 +19,7 @@ def propose(
         that match.
         
         Args:
-            context_token_ids: List of token IDs representing the 
+            context_token_ids: Numpy array of token IDs representing the 
                                context sequence.
             n: Length of the n-gram to match.
             k: Number of tokens follow the match. If there are less 
@@ -41,66 +39,65 @@ def propose(
               followed that pattern. Here we will return [4,2,3] because 
               we only have three tokens after the match.
         """
-        # TODO: Use c++ to implement the _find_subarray_kmp to
-        # improve the efficiency
-        return self._find_subarray_kmp(context_token_ids, n, k)
+        return _find_subarray_kmp(context_token_ids, n, k)
 
-    @staticmethod
-    def _kmp_lps_array(pattern: List[int]) -> List[int]:
-        """
-        Build the lps (longest proper prefix which is also suffix) 
-        array for the pattern.
-        """
-        lps = [0] * len(pattern)
-        prev_lps = 0  # length of the previous longest prefix suffix
-        i = 1
 
-        while i < len(pattern):
-            if pattern[i] == pattern[prev_lps]:
-                prev_lps += 1
-                lps[i] = prev_lps
-                i += 1
+@jit(nopython=True)
+def _kmp_lps_array(pattern: np.ndarray) -> np.ndarray:
+    """
+    Build the lps (longest proper prefix which is also suffix) 
+    array for the pattern.
+    """
+    lps = np.zeros(len(pattern), dtype=np.int32)
+    prev_lps = 0  # length of the previous longest prefix suffix
+    i = 1
+
+    while i < len(pattern):
+        if pattern[i] == pattern[prev_lps]:
+            prev_lps += 1
+            lps[i] = prev_lps
+            i += 1
+        else:
+            if prev_lps != 0:
+                prev_lps = lps[prev_lps - 1]
             else:
-                if prev_lps != 0:
-                    prev_lps = lps[prev_lps - 1]
-                else:
-                    lps[i] = 0
-                    i += 1
+                lps[i] = 0
+                i += 1
+    return lps
 
-        return lps
 
-    @staticmethod
-    def _find_subarray_kmp(
-        context_token_ids: np.ndarray,
-        n: int,
-        k: int,
-    ) -> Optional[np.ndarray]:
-        context_len = context_token_ids.shape[0]
-        assert n > 0
+@jit(nopython=True)
+def _find_subarray_kmp(
+    context_token_ids: np.ndarray,
+    n: int,
+    k: int,
+) -> Optional[np.ndarray]:
+    context_len = context_token_ids.shape[0]
+    assert n > 0
 
-        pattern = context_token_ids[-n:]
-        # Precompute lps array for Y
-        lps = NgramProposer._kmp_lps_array(pattern)
+    pattern = context_token_ids[-n:]
+    # Precompute lps array for Y
+    lps = _kmp_lps_array(pattern)
 
-        i = 0
-        j = 0
-        # -n because the last n tokens are used as pattern
-        while i < context_len - n:
-            if context_token_ids[i] == pattern[j]:
-                i += 1
-                j += 1
+    i = 0
+    j = 0
+    # -n because the last n tokens are used as pattern
+    while i < context_len - n:
+        if context_token_ids[i] == pattern[j]:
+            i += 1
+            j += 1
 
-                # If we have matched the entire Y
-                if j == n:
-                    # Found pattern in context, gather the next K elements
-                    return context_token_ids[i:i + k]
+            # If we have matched the entire Y
+            if j == n:
+                # Found pattern in context, gather the next K elements
+                return context_token_ids[i:i + k]
+        else:
+            # Mismatch
+            if j != 0:
+                # Use the lps array to avoid re-checking elements
+                j = lps[j - 1]
             else:
-                # Mismatch
-                if j != 0:
-                    # Use the lps array to avoid re-checking elements
-                    j = lps[j - 1]
-                else:
-                    i += 1
+                i += 1
 
-        # Y not found
-        return None
+    # Y not found
+    return None
diff --git a/vllm/v1/worker/gpu_model_runner.py b/vllm/v1/worker/gpu_model_runner.py
index 5754422cb1f7..c38e22eb2b4e 100644
--- a/vllm/v1/worker/gpu_model_runner.py
+++ b/vllm/v1/worker/gpu_model_runner.py
@@ -121,11 +121,20 @@ def __init__(
         # Set up speculative decoding.
         self.use_spec_decode = False
         if self.speculative_config:
+            self.use_spec_decode = True
+
             # TODO: find a better way to check if we are using ngram.
             assert self.speculative_config.ngram_prompt_lookup_min, \
                     "Currently, only ngram spec decode is supported in V1."
-            self.drafter = NgramProposer()
-            self.use_spec_decode = True
+            if get_pp_group().is_last_rank:
+                self.drafter = NgramProposer()
+                # Trigger Numba JIT compilation for N-gram proposer.
+                # This usually takes less than 1 second.
+                self.drafter.propose(
+                    np.zeros(1024, dtype=np.int32),
+                    self.speculative_config.ngram_prompt_lookup_min,
+                    self.speculative_config.num_speculative_tokens,
+                )
 
         # Request states.
         self.requests: Dict[str, CachedRequestState] = {}

```
