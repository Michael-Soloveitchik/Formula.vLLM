# PR #37347: [Perf] Optimize token_embed for pooling models, 1.0% token throughput improvement

- field: scheduling  | area: v1/pool
- author: yewentao256
- state: MERGED  created: 2026-03-17T21:50:59Z  merged: 2026-03-19T02:27:51Z
- url: https://github.com/vllm-project/vllm/pull/37347
- labels: ready, v1

## Body

## Purpose

~1. Identical code replacement, but much less calculation.~

```py
        hidden_states_lst = [
            hidden_states[first : last + 1]
            for first, last in zip(
                pooling_cursor.first_token_indices_gpu.tolist(),
                pooling_cursor.last_token_indices_gpu.tolist(),
            )
        ]
```
to
```py
        split_sizes = pooling_cursor.num_scheduled_tokens_cpu.tolist()
        hidden_states_lst = list(hidden_states.split(split_sizes, dim=0))
```

2. reuse cumsum that has been computed (query_start_loc_gpu)

## Test

### Acc

Covered in pooling model unit test

```bash
tests/models/language/pooling/test_multi_vector_retrieval.py
tests/models/language/pooling/test_all_pooling_plus_chunked_prefill.py
tests/models/language/pooling/test_token_classification.py
tests/entrypoints/pooling/embed/test_online.py
```

### Perf

`vllm serve intfloat/multilingual-e5-small   --runner pooling   --dtype bfloat16   --host 127.0.0.1   --port 9257   --max-num-seqs 128   --enforce-eager`

Using this bench script:

```py
import json
import time
import statistics as st
import requests

BASE = "http://127.0.0.1:9257"
MODEL = "intfloat/multilingual-e5-small"
# Start with a moderate batch/length so token-embed responses stay manageable.
# Once this runs reliably, you can scale BATCH up to 32 or 64.
BATCH = 16
ROUNDS = 20
WARMUP_ROUNDS = 5
MAX_INPUT_TOKENS = 256
text = ("The best thing about vLLM is token pooling performance. " * 80).strip()

session = requests.Session()
tokens = session.post(
    f"{BASE}/tokenize", json={"model": MODEL, "prompt": text}, timeout=30
).json()["tokens"]
tokens = tokens[:MAX_INPUT_TOKENS]

payload = {
    "model": MODEL,
    "input": [tokens[:] for _ in range(BATCH)],
    "task": "token_embed",
    # Returning token-level embeddings as JSON floats is extremely large.
    # Use raw bytes so the benchmark primarily measures backend latency.
    "encoding_format": "bytes_only",
    "embed_dtype": "float16",
}

print(f"tokens_per_prompt={len(tokens)} batch={BATCH}")
for i in range(WARMUP_ROUNDS):
    t0 = time.perf_counter()
    response = session.post(f"{BASE}/pooling", json=payload, timeout=300)
    response.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(
        f"warmup {i + 1}/{WARMUP_ROUNDS}: "
        f"{elapsed_ms:.2f} ms, {len(response.content)} bytes"
    )

lat_ms = []
for i in range(ROUNDS):
    t0 = time.perf_counter()
    r = session.post(f"{BASE}/pooling", json=payload, timeout=300)
    r.raise_for_status()
    lat_ms.append((time.perf_counter() - t0) * 1000)
    if (i + 1) % 5 == 0 or i + 1 == ROUNDS:
        print(f"round {i + 1}/{ROUNDS}: {lat_ms[-1]:.2f} ms")

lat_sorted = sorted(lat_ms)
mean_ms = st.mean(lat_ms)
p50_ms = lat_sorted[len(lat_sorted) // 2]
p95_ms = lat_sorted[int(len(lat_sorted) * 0.95) - 1]
print(json.dumps({
    "batch": BATCH,
    "rounds": ROUNDS,
    "mean_ms": round(mean_ms, 2),
    "p50_ms": round(p50_ms, 2),
    "p95_ms": round(p95_ms, 2),
    "seq_per_s": round(BATCH * 1000.0 / mean_ms, 2),
    "input_tok_per_s": round(BATCH * len(tokens) * 1000.0 / mean_ms, 2),
}, indent=2))
```

Got

```bash
# now
{
  "batch": 16,
  "rounds": 20,
  "mean_ms": 52.17,
  "p50_ms": 51.41,
  "p95_ms": 56.52,
  "seq_per_s": 306.7,
  "input_tok_per_s": 78514.6
}

# main
{
  "batch": 16,
  "rounds": 20,
  "mean_ms": 52.69,
  "p50_ms": 52.55,
  "p95_ms": 54.05,
  "seq_per_s": 303.64,
  "input_tok_per_s": 77730.59
}
```

## Comments

### yewentao256 @ 2026-03-18T18:35:06Z

@noooop Thanks for the review! I benched again using only option 2, we can still get around 1% perf improvement, would you take a look again?
---