# Semantic solution prompt — vLLM PR enrichment (v2)

Builds the per-PR `<n>.sem.json` the viewer's **Solution — semantic** block renders.
The pattern below IS the prompt: the model fills exactly these fields and returns
STRICT JSON only. Model: `claude-opus-4-8` via LiteLLM proxy.

This prompt is applied **3 times independently** per PR (temperature-varied samples),
then a separate **judge** (see bottom) merges the 3 into one packed 5-component result.
Keep the structure identical across sampler and judge so the merge is a like-for-like
reconciliation.

## Background (what this dataset is for)

Each entry teaches a reader what a real vLLM optimization PR did and why. The audience
is a competent CS **bachelor** with NO vLLM internals knowledge. The split into
`*_general` (one-glance, plain) and `*_detailed` (file/function/mechanism) lets the
viewer show a plain summary first and the technical depth on demand. `workload` records
the concrete serving scenario that exercises the code so PRs can be grouped/benchmarked;
`performance` records the measured gain in the PR's own numbers so nothing is invented.

## Inputs the model sees (ORIGINAL PR, not a summary)

- `title` — PR title.
- `body` — the author's FULL original PR description.
- `comments` — ALL PR comments verbatim (benchmark tables and reviewer-surfaced
  motivation usually live here). Read them; never rely on a pre-made summary.
- `diff` — the merged unified diff (authoritative: what actually changed).

## Decision-making (how to derive each field)

- **issue** + **workload** ← body + comments + diff together. The diff reveals what was
  really wrong even when the body is vague.
- **solution** ← the diff (what changed) cross-checked with the author's description.
- **authors_words** ← the author's OWN body + their OWN comments only.
- **performance** ← ONLY numbers the PR states. Absent ⇒ `null`. Never invent, round up,
  or infer from general knowledge.
- **Conflicts / retractions**: if a later comment corrects or retracts an earlier number
  (common: an author posts a speedup, then says it was a benchmarking artifact), use the
  CORRECTED number and say so in `evidence`. The retracted number must not appear in
  `value`. If body and diff disagree on what changed, trust the diff.
- **General-fix duty**: while reading, if you notice a concrete detail the PR clearly
  states but which is missing or wrong in any derived text, capture the correct detail
  from the original PR — do not propagate a sloppy earlier summary.

## The pattern (fields the model must return)

Every field: short, clear sentences. State the algorithm, the inefficiency, the change,
the effect. No poetry, no marketing, no hedging. Technical terms exact.

```json
{
  "issue_general": "2-3 short sentences. Plain-language: what was wrong and why it mattered. A bachelor with no vLLM knowledge must understand it. Name the operation in plain words, not just a symbol.",
  "issue_detailed": "3-5 short sentences. The exact code path / function / data structure that was slow or wrong, and the mechanism WHY (extra kernel launches, redundant memory traffic, O(n^2) scan, lock contention, recompilation, blocking sync, etc.). Cite file/function names from the diff.",
  "solution_general": "2-3 short sentences. Plain-language: what the fix does and why it removes the problem. Bachelor-level.",
  "solution_detailed": "3-5 short sentences. The concrete code change (before vs after): the new function/kernel/data structure/config, where it is wired in, and the mechanism that makes it faster or correct. From the diff.",
  "authors_words": "2-4 short sentences. ONLY the PR author's own stated approach and reasoning, condensed faithfully from their body/their own comments. Do NOT fold in reviewer remarks here (reviewer-sourced numbers belong in performance.evidence / workload.evidence). If the author gave no approach, exactly: (author gave no solution description).",
  "workload": {
    "description": "2-3 short sentences. The concrete serving/inference workload that exercises this code: model(s), parallelism, dtype/quantization, batch size, sequence length, request pattern — whatever the PR specifies. Bachelor-level but exact.",
    "evidence": "1-2 short sentences quoting or tightly paraphrasing the PR's OWN words that establish this workload (command line, config, benchmark setup, model name). Traceable to the PR text, never assumed. If the PR gives no workload detail, exactly: (PR states no explicit workload)."
  },
  "performance": {
    "metric": "What was measured, in the PR's own terms (e.g. 'end-to-end throughput', 'decode latency', 'peak GPU memory', 'first-request latency', 'TTFT'). null if the PR states no measurement.",
    "value": "The headline numbers ONLY, no unit text. Keep the before->after pair and/or ratio as the PR states it: '87.5 -> 92', '31.343 -> 2.890', '40.48 -> 14.98', '~33', '1.8x'. Do NOT repeat the unit here. null if unstated.",
    "unit": "The dimension of `value`, stated once (e.g. 'tokens/s', 'seconds', 'GiB', '%', 'x speedup'). null if unstated.",
    "evidence": "1-2 short sentences: where in the PR this number comes from (author body, a named benchmark, a comment table) and the headline takeaway in words. Note any retraction/correction. null if no number stated."
  }
}
```

## DO

- Read every comment before writing; benchmark numbers hide there.
- Put file/function names ONLY in `*_detailed`; keep `*_general` jargon-light.
- Numbers only in `performance.value`; the unit once in `performance.unit`.
- Prefer corrected numbers over retracted ones, and state the correction in evidence.
- Use the diff as the authority for what actually changed.
- Keep sentences short enough for a bachelor to follow.

## DON'T

- Don't invent, round, or infer any performance number. Absent ⇒ null.
- Don't duplicate the unit inside `value`.
- Don't put reviewer remarks in `authors_words`.
- Don't leak file/function names into `*_general`.
- Don't emit anything outside the single JSON object. No prose, no markdown fences.

## Cases (how to handle the common shapes)

- **Pure addition / new model or feature** (no removed code): `issue_general` = the
  capability that was missing; `issue_detailed` = why the existing machinery could not
  do it (from the diff). Solution describes what the addition provides.
- **Numbers retracted in comments**: use corrected; name the retraction in evidence.
- **No measurement stated**: `performance` is all four keys `null`. Fine and common.
- **No explicit workload**: `workload.description` gives the natural serving situation
  from the diff; `workload.evidence` is exactly `(PR states no explicit workload)`.
- **Config-only PR** (e.g. adds a tuning JSON): solution_detailed names the files/shapes;
  the "inefficiency" is the missing-config fallback the PR removes.

## Worked example of the performance split (do this)

PR says "87.5 tok/s guided-unoptimized vs 92 tok/s guided-optimized, ~5%":
`"metric":"decode throughput, single request"`, `"value":"87.5 -> 92 (~5%)"`,
`"unit":"tokens/s"`, `"evidence":"PR comment: 87.5 vs 92 tok/s on Llama 3.1 8B / H100, ~5% faster."`
The unit (`tokens/s`) appears once, in `unit` — never inside `value`.

---

# Judge prompt — merge 3 samples into one packed result

After the sampler prompt runs 3 times on a PR you get `sample_1`, `sample_2`,
`sample_3`, each the full JSON above. The judge produces ONE final JSON of the SAME
structure. The judge ALSO re-reads the original PR (title + body + all comments + diff)
and is the final authority.

## Judge decision rules (super precise)

1. **Same structure out**: exactly the 5 text components + `workload{description,
   evidence}` + `performance{metric,value,unit,evidence}`. STRICT JSON only.
2. **Agreement**: where ≥2 samples agree on a fact, keep it. Prefer the clearest, most
   concrete wording among the agreeing samples; do not average prose.
3. **Hard conflict**: if samples disagree on a substantive fact (different mechanism,
   different number, different workload), RE-CHECK the original PR and pick the version
   the PR supports. If none is supported, write only what the PR supports and drop the
   rest. Never keep two contradictory claims.
4. **Numbers**: `performance` must match the PR exactly. If samples differ on a number,
   the PR text decides. Honor retractions (corrected number wins). Absent ⇒ null.
   Never introduce a number no sample and no PR states.
5. **authors_words**: author-only, reconciled to what the author actually wrote.
6. **General-fix duty**: if, on re-reading the PR, you find a concrete detail that the
   samples got wrong or missed (a file, a model name, a config, a direction of change),
   correct it from the original PR. The final entry must faithfully reflect the PR.
7. **Granularity discipline**: file/function names only in `*_detailed`; `*_general`
   stays plain; unit only in `performance.unit`.

## Judge inputs

- `original` — title, full body, ALL comments, diff.
- `sample_1`, `sample_2`, `sample_3` — the three sampler JSONs.

## Judge output

ONE JSON object, same schema as the sampler, no commentary outside it.
