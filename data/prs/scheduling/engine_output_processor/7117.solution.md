# PR #7117: [core][misc] simply output processing with shortcut for non-parallel sampling and non-beam search usecase

- field: scheduling  | area: engine_output_processor
- author: youkaichao
- state: MERGED  created: 2024-08-04T01:49:58Z  merged: 2024-08-04T07:22:19Z
- url: https://github.com/vllm-project/vllm/pull/7117
- labels: 

# Issue

## Problem

The single-step output processor's `_process_sequence_group_outputs` always ran the full general-purpose path built for parallel sampling (n>1) and beam search. That path builds parent/child sequence lists, maps samples to parents, forks sequences in the block manager, and runs beam-search selection logic. For the overwhelmingly common case of n==1 and no beam search, this is pure overhead on every decode step.

## Example

A standard greedy/top-p request with sampling_params.n==1 still allocated child_seqs lists, called get_seqs(status=RUNNING), iterated parent-sample pairs, and evaluated beam-search branches it would never use. benchmark_throughput at input-len 256/output-len 256 on Meta-Llama-3-8B showed ~1.5% throughput gain once bypassed.

## Use case

Normal online/offline LLM serving where nearly all requests use single-sequence sampling (n=1, no beam search) and output processing runs once per generated token per request.

## Resources

CPU time and per-step Python overhead in the engine's output-processing loop (kernel-launch/scheduling critical path between decode steps), not GPU compute.

## Type of users

Anyone serving high-throughput single-sequence generation, i.e. the default config without parallel sampling or beam search.

## Kind of inputs

Requests with sampling_params.n == 1 and use_beam_search == False — the dominant request shape at any batch size.

## Mechanism with the issue

`_process_sequence_group_outputs` in vllm/engine/output_processor/single_step.py unconditionally constructed parent_seqs/child_seqs, looped sample-to-parent assignment, and branched into beam-search selection. For n==1 there is exactly one sample and one sequence, so all that bookkeeping is wasted Python work on the hot per-token path.

# Solution

## Fix in words

Added early-exit shortcut at top of `_process_sequence_group_outputs`: when `sampling_params.n == 1 and not use_beam_search`, grab the single sample (`outputs.samples[0]`) and single sequence (`seq_group.seqs[0]`) directly, append the token id + logprobs, optionally detokenize in place, run `stop_checker.maybe_stop_sequence`, free the seq via each scheduler if finished, then `return`. This skips all parent/child list building, block-manager forking, and beam-search selection. Remaining diff just caches `seq_group.sampling_params` into a local `sampling_params` to avoid repeated attribute lookups in the general path. Yields ~1.5% throughput improvement.

## Author's solution

Author added a fast-path shortcut in output processing for the non-parallel-sampling (n=1) and non-beam-search case, bypassing the general multi-sequence logic. Described as a tiny step toward issue #7116. Benchmarked on H100 with benchmark_throughput (Meta-Llama-3-8B, 256/256) showing ~1.5% throughput gain vs main.

## Ground-truth diff

The authoritative resolution — the actual merged diff.

### Files changed
- `vllm/engine/output_processor/single_step.py`

### Diff
```diff
diff --git a/vllm/engine/output_processor/single_step.py b/vllm/engine/output_processor/single_step.py
index 59eb4bc439d1..4a46c93f8425 100644
--- a/vllm/engine/output_processor/single_step.py
+++ b/vllm/engine/output_processor/single_step.py
@@ -81,6 +81,29 @@ def process_prompt_logprob(self, seq_group: SequenceGroup,
 
     def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                         outputs: SequenceGroupOutput) -> None:
+        sampling_params = seq_group.sampling_params
+        if sampling_params.n == 1 and not sampling_params.use_beam_search:
+            # only have one output sample
+            sample = outputs.samples[0]
+            # only have one sequence
+            seq = seq_group.seqs[0]
+            seq.append_token_id(sample.output_token, sample.logprobs)
+            if sampling_params.detokenize and self.detokenizer:
+                new_char_count = self.detokenizer.decode_sequence_inplace(
+                    seq, sampling_params)
+            else:
+                new_char_count = 0
+            self.stop_checker.maybe_stop_sequence(
+                seq,
+                new_char_count,
+                sampling_params,
+                lora_req=seq_group.lora_request,
+            )
+            if seq.is_finished():
+                for scheduler in self.scheduler:
+                    scheduler.free_seq(seq)
+            return
+
         # Process samples
         samples = outputs.samples
         parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
@@ -127,20 +150,20 @@ def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
             child_seqs.append((parent, parent))
 
         for seq, _ in child_seqs:
-            if seq_group.sampling_params.detokenize and self.detokenizer:
+            if sampling_params.detokenize and self.detokenizer:
                 new_char_count = self.detokenizer.decode_sequence_inplace(
-                    seq, seq_group.sampling_params)
+                    seq, sampling_params)
             else:
                 new_char_count = 0
             self.stop_checker.maybe_stop_sequence(
                 seq,
                 new_char_count,
-                seq_group.sampling_params,
+                sampling_params,
                 lora_req=seq_group.lora_request,
             )
 
         # Non-beam search case
-        if not seq_group.sampling_params.use_beam_search:
+        if not sampling_params.use_beam_search:
             # For newly created child sequences, add them to the sequence group
             # and fork them in block manager if they are not finished.
             for seq, parent in child_seqs:
@@ -164,8 +187,8 @@ def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
         # Select the child sequences to keep in the sequence group.
         selected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
         unselected_child_seqs: List[Tuple[Sequence, Optional[Sequence]]] = []
-        beam_width = seq_group.sampling_params.best_of
-        length_penalty = seq_group.sampling_params.length_penalty
+        beam_width = sampling_params.best_of
+        length_penalty = sampling_params.length_penalty
 
         # Select the newly finished sequences with the highest scores
         # to replace existing finished sequences.
@@ -219,8 +242,8 @@ def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
             best_running_seq = running_child_seqs[0][0]
             current_worst_seq = all_finished_seqs[beam_width - 1][0]
             stop_beam_search = self._check_beam_search_early_stopping(
-                seq_group.sampling_params.early_stopping,
-                seq_group.sampling_params, best_running_seq, current_worst_seq)
+                sampling_params.early_stopping, sampling_params,
+                best_running_seq, current_worst_seq)
 
         if stop_beam_search:
             # Stop the beam search and remove all the running sequences from

```
