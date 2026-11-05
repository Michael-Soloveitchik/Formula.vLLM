#!/usr/bin/env python3
"""Microbench for vLLM PR #33593: membership test over a slice, [:] vs islice.

The PR's real hot path (reasoning/basic_parsers.py BaseThinkingReasoningParser
.extract_content_ids and step3 parser) is a *membership* test:

    base:  self.end_token_id not in input_ids[:-1]
    gold:  self.end_token_id not in islice(input_ids, 0, max(0, len-1))

The win is NOT materializing a new list. `input_ids[:-1]` copies the whole
sequence (O(n) alloc) *before* the `in` scan. `islice(...)` is lazy, so the
`in` membership both (a) allocates nothing and (b) short-circuits at the first
match. A second hot path slices a suffix `all_token_ids[delta_from:]` that is
then membership-scanned — same copy-vs-lazy story.

Two experiments, each matched to a real call site:
  membership_present : end token sits early in a long sequence (typical: the
                       </think> marker ends reasoning long before generation
                       finishes). base copies all N then scans; gold scans
                       lazily and stops at the marker. Big win.
  membership_absent  : end token absent (reasoning still open). base copies N
                       then scans N; gold scans N but copies nothing. Win =
                       the elided allocation.

Emits per-repeat pairs for ratio stats + best-of-R summary.
"""

import time
from itertools import islice

SIZE = 100_000     # long generation sequence (tokens)
EARLY = SIZE // 10 # reasoning-end marker lands ~10% in
R = 9              # repeats (paired)
INNER = 200        # calls per timed block (membership is cheap; amortize)

# present: a real token id sitting early; absent: an id that never occurs.
PRESENT_TOKEN = -7
ABSENT_TOKEN = -999

seq_present = list(range(SIZE))
seq_present[EARLY] = PRESENT_TOKEN

seq_absent = list(range(SIZE))


def timed(fn):
    start = time.perf_counter()
    for _ in range(INNER):
        fn()
    return (time.perf_counter() - start) / INNER


def base_present():
    return PRESENT_TOKEN not in seq_present[:-1]


def gold_present():
    return PRESENT_TOKEN not in islice(seq_present, 0, max(0, len(seq_present) - 1))


def base_absent():
    return ABSENT_TOKEN not in seq_absent[:-1]


def gold_absent():
    return ABSENT_TOKEN not in islice(seq_absent, 0, max(0, len(seq_absent) - 1))


# warmup
for _ in range(2):
    base_present(); gold_present(); base_absent(); gold_absent()

present_base, present_gold = [], []
absent_base, absent_gold = [], []
for _ in range(R):
    # A-B-B-A paired ordering to cancel drift
    present_base.append(timed(base_present))
    present_gold.append(timed(gold_present))
    present_gold.append(timed(gold_present))
    present_base.append(timed(base_present))
    absent_base.append(timed(base_absent))
    absent_gold.append(timed(gold_absent))
    absent_gold.append(timed(gold_absent))
    absent_base.append(timed(base_absent))

print(f"membership_present base {min(present_base):.9f}")
print(f"membership_present gold {min(present_gold):.9f}")
print(f"membership_absent base {min(absent_base):.9f}")
print(f"membership_absent gold {min(absent_gold):.9f}")

for b, g in zip(present_base, present_gold):
    print(f"run membership_present {b:.9f} {g:.9f}")
for b, g in zip(absent_base, absent_gold):
    print(f"run membership_absent {b:.9f} {g:.9f}")
