# PR #6865: [core][misc] improve free_finished_seq_groups

- field: scheduling  | area: core_scheduler.py
- author: youkaichao
- state: MERGED  created: 2024-07-27T17:15:08Z  merged: 2024-07-30T21:32:12Z
- url: https://github.com/vllm-project/vllm/pull/6865
- labels: ready

# Issue

## Problem

free_finished_seq_groups is called by the engine after every model step, yet it scanned all three scheduler queues (running, swapped, waiting) to find finished sequence groups. Newly finished requests can only appear in the running queue, so scanning swapped and waiting is wasted work. Cost grows linearly with total requests held by the engine, making it a per-step bottleneck.

## Example

facebook/opt-125m, input 16, output-len 16: at 100k prompts throughput on main is 211 req/s vs 577 req/s at 10k prompts (~3x slowdown), largely from this O(total requests) per-step scan.

## Use case

Large offline batch inference where a user submits a very large dataset (e.g. 100k or 500k prompts) to the LLM engine in one call.

## Resources

CPU time in the scheduler critical path; per-step overhead dominated by Python iteration over queues rather than GPU compute.

## Type of users

Users running offline batch inference over large datasets, especially with small/fast models where scheduler overhead is a large fraction of per-step time.

## Kind of inputs

Tens to hundreds of thousands of queued prompts with short sequences, so most requests sit in waiting/swapped while only a few are running per step.

## Mechanism with the issue

free_finished_seq_groups iterated over [self.running, self.swapped, self.waiting] calling seq_group.is_finished() on every group each step, then rebuilt self.running with a second is_finished() pass. Since finished groups only arise in running, scanning swapped and waiting scaled per-step cost with the full backlog, turning an O(running) step into O(total requests).

# Solution

## Fix in words

Before: looped over all three queues collecting finished request ids, then rebuilt self.running via a comprehension re-checking is_finished(). After: loops over self.running once, appending finished request ids to self._finished_requests_ids and collecting non-finished groups into a new remaining deque that replaces self.running. Drops the swapped/waiting scans and the second is_finished() pass, so per-step cost is proportional to running-queue size, not total request count. Also added comments noting _finished_requests_ids feeds Mamba cache eviction.

## Author's solution

Author observed new finished requests only appear in the running queue, so searching all queues every step is unnecessary. Fix restricts free_finished_seq_groups to scan only the running queue, cutting dominant scheduler overhead for large-dataset batch inference (100k-prompt slowdown improves from ~3x to ~40%).

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/core/scheduler.py`

### Diff
```diff
diff --git a/vllm/core/scheduler.py b/vllm/core/scheduler.py
index 6e59c5e0f74f..83db52c94687 100644
--- a/vllm/core/scheduler.py
+++ b/vllm/core/scheduler.py
@@ -313,6 +313,7 @@ def __init__(
         # Sequence groups finished requests ids since last step iteration.
         # It lets the model know that any state associated with these requests
         # can and must be released after the current step.
+        # This is used to evict the finished requests from the Mamba cache.
         self._finished_requests_ids: List[str] = list()
         # Time at previous scheduling step
         self.prev_time = 0.0
@@ -374,6 +375,7 @@ def abort_seq_group(self, request_id: Union[str, Iterable[str]]) -> None:
             for aborted_group in aborted_groups:
                 # Remove the sequence group from the state queue.
                 state_queue.remove(aborted_group)
+                # Remove the aborted request from the Mamba cache.
                 self._finished_requests_ids.append(aborted_group.request_id)
                 for seq in aborted_group.get_seqs():
                     if seq.is_finished():
@@ -1058,13 +1060,16 @@ def free_seq(self, seq: Sequence) -> None:
         self.block_manager.free(seq)
 
     def free_finished_seq_groups(self) -> None:
-        for queue in [self.running, self.swapped, self.waiting]:
-            self._finished_requests_ids += [
-                seq_group.request_id for seq_group in queue
-                if seq_group.is_finished()
-            ]
-        self.running = deque(seq_group for seq_group in self.running
-                             if not seq_group.is_finished())
+        remaining: Deque[SequenceGroup] = deque()
+        for seq_group in self.running:
+            if seq_group.is_finished():
+                # Add the finished requests to the finished requests list.
+                # This list will be used to update the Mamba cache in the
+                # next step.
+                self._finished_requests_ids.append(seq_group.request_id)
+            else:
+                remaining.append(seq_group)
+        self.running = remaining
 
     def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
         self.block_manager.allocate(seq_group)

```
