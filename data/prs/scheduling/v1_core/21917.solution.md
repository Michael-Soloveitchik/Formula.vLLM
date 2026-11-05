# PR #21917: [Core] Optimize scheduler request removal for single completions

- field: scheduling  | area: v1_core
- author: chi2liu
- state: MERGED  created: 2025-07-30T08:24:15Z  merged: 2025-08-20T01:25:59Z
- url: https://github.com/vllm-project/vllm/pull/21917
- labels: ready, v1

# Issue

## Problem

The scheduler removed stopped/finished requests from the `self.running` list using O(n) list-comprehension rebuilds and per-item `list.remove()` calls. Even when only a single request completed in a step, the code rebuilt the entire running list with a comprehension that did a membership test against a set for every element. This made the common case (one completion per scheduler step) pay the full cost of scanning and reallocating the whole list.

## Example

At 200 req/s with a large running batch, each scheduler step typically stops just one request, yet `[req for req in self.running if req not in stopped_running_reqs]` scans all N running requests and allocates a new list every time.

## Use case

High-throughput online serving where the scheduler loop runs continuously and requests finish one at a time per step, so request-removal cost accumulates across millions of steps per hour.

## Resources

CPU time in the scheduler hot path — list scanning plus repeated list allocation/copying on the Python engine-core thread.

## Type of users

Anyone running vLLM v1 as an online server under sustained load; impact grows with large concurrent running batches.

## Kind of inputs

Workloads with many concurrent running requests but few completions per step (typically one), i.e. steady-state high-concurrency decode traffic.

## Mechanism with the issue

In `scheduler.py`, `update_from_output` rebuilt `self.running` via list comprehension, and `finish_requests` collected removals in a list then called `self.running.remove(request)` in a loop (each `remove` is O(n)). Both paths were O(n) or worse regardless of how few requests were actually being removed.

# Solution

## Fix in words

Adds a `remove_all(lst, items_to_remove)` helper in `utils.py`. For the common single-item case it takes a fast path: pop the one element from the set and call `lst.remove(item)` once (O(n) single scan, in-place, with `contextlib.suppress(ValueError)` guarding absence), returning the modified original list. For multiple items it keeps the original list-comprehension set-membership rebuild. `update_from_output` now calls `self.running = remove_all(self.running, stopped_running_reqs)` instead of the inline comprehension, and `finish_requests` changes `running_requests_to_remove` from a list to a set and replaces its per-item `self.running.remove()` loop with one `remove_all` call. Net effect: single-completion removals avoid full list rebuild, giving the reported 18.7% scheduler speedup and 2.6–3.4x faster single removal.

## Author's solution

Author added a fast path for single-request removal while keeping the original list-comprehension logic for multiple requests, aiming for minimal code change with maximum impact. They report an 18.7% scheduler performance improvement, 2.6–3.4x speedup on single removals, 64% improvement in removal operations, and ~560ms saved per hour at 200 req/s.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/sched/utils.py`

### Diff
```diff
diff --git a/vllm/v1/core/sched/scheduler.py b/vllm/v1/core/sched/scheduler.py
index 981023409045..ff02765fa625 100644
--- a/vllm/v1/core/sched/scheduler.py
+++ b/vllm/v1/core/sched/scheduler.py
@@ -25,7 +25,7 @@
                                        SchedulerOutput)
 from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                               create_request_queue)
-from vllm.v1.core.sched.utils import check_stop
+from vllm.v1.core.sched.utils import check_stop, remove_all
 from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                             EngineCoreOutputs)
 from vllm.v1.kv_cache_interface import KVCacheConfig
@@ -885,9 +885,7 @@ def update_from_output(
 
         # Remove the stopped requests from the running and waiting queues.
         if stopped_running_reqs:
-            self.running = [
-                req for req in self.running if req not in stopped_running_reqs
-            ]
+            self.running = remove_all(self.running, stopped_running_reqs)
         if stopped_preempted_reqs:
             # This is a rare case and unlikely to impact performance.
             self.waiting.remove_requests(stopped_preempted_reqs)
@@ -989,7 +987,7 @@ def finish_requests(
         else:
             request_ids = set(request_ids)
 
-        running_requests_to_remove = []
+        running_requests_to_remove = set()
         waiting_requests_to_remove = []
         valid_requests = []
 
@@ -1002,13 +1000,13 @@ def finish_requests(
 
             valid_requests.append(request)
             if request.status == RequestStatus.RUNNING:
-                running_requests_to_remove.append(request)
+                running_requests_to_remove.add(request)
             else:
                 waiting_requests_to_remove.append(request)
 
         # Remove all requests from queues at once for better efficiency
-        for request in running_requests_to_remove:
-            self.running.remove(request)
+        if running_requests_to_remove:
+            self.running = remove_all(self.running, running_requests_to_remove)
         if waiting_requests_to_remove:
             self.waiting.remove_requests(waiting_requests_to_remove)
 
diff --git a/vllm/v1/core/sched/utils.py b/vllm/v1/core/sched/utils.py
index 42ec95091f96..42d3e5c68b4c 100644
--- a/vllm/v1/core/sched/utils.py
+++ b/vllm/v1/core/sched/utils.py
@@ -1,5 +1,6 @@
 # SPDX-License-Identifier: Apache-2.0
 # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
+import contextlib
 from typing import Optional
 
 import torch
@@ -7,6 +8,38 @@
 from vllm.v1.request import Request, RequestStatus
 
 
+def remove_all(lst: list, items_to_remove: set) -> list:
+    """Remove all items from a list that are in the items_to_remove set.
+    
+    This method optimizes for the common case of removing a single item,
+    falling back to list comprehension for multiple items.
+    
+    Args:
+        lst: The list to remove items from
+        items_to_remove: Set of items to remove
+    
+    Returns:
+        Either the modified original list (for single item removal) or
+        a new list (for multiple item removal). Callers should use the
+        returned value.
+    
+    Note:
+        For single item removal, this modifies the original list in-place
+        and returns it. For multiple items, it creates and returns a new list.
+    """
+    if not items_to_remove:
+        return lst
+
+    if len(items_to_remove) == 1:
+        # Fast path for single item removal (most common case)
+        item = next(iter(items_to_remove))
+        with contextlib.suppress(ValueError):
+            lst.remove(item)
+        return lst
+    # For multiple items, use list comprehension
+    return [item for item in lst if item not in items_to_remove]
+
+
 def check_stop(request: Request,
                max_model_len: int,
                pooler_output: Optional[torch.Tensor] = None) -> bool:

```
