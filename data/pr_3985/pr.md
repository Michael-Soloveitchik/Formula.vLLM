# PR #3985: [Core][Model] Use torch.compile to accelerate layernorm in commandr

- field: scheduling  | area: model_executor/models
- author: youkaichao
- state: MERGED  created: 2024-04-11T00:35:43Z  merged: 2024-04-11T01:48:26Z
- url: https://github.com/vllm-project/vllm/pull/3985
- labels: 

## Body

If we just add `@torch.compile` to `forward` function, pytorch will try to trace `self`, and everytime `self` changes, it tries to re-compile.

However, `self` actually does not matter in our computation. We just need a standalone layernorm function. So we can extract the computation in a standalone way, so that `torch` only compile once.

benchmark script:

```shell
python benchmark_latency.py --model CohereForAI/c4ai-command-r-v01 -tp=4 --batch-size=1
```

without compile:
Avg latency: 2.43594957968841 seconds
10% percentile latency: 2.42827770607546 seconds
25% percentile latency: 2.429899246431887 seconds
50% percentile latency: 2.433337311260402 seconds
75% percentile latency: 2.436360448366031 seconds
90% percentile latency: 2.4394099053926768 seconds

with compile on `forward`:
Avg latency: 2.434882906700174 seconds
10% percentile latency: 2.4297383648343382 seconds
25% percentile latency: 2.4304619752801955 seconds
50% percentile latency: 2.433090591803193 seconds
75% percentile latency: 2.436410292983055 seconds
90% percentile latency: 2.441519308835268 seconds

with compile on standalone function:

Avg latency: 2.3143214230115214 seconds
10% percentile latency: 2.3101052045822144 seconds
25% percentile latency: 2.312119716545567 seconds
50% percentile latency: 2.3148383810184896 seconds
75% percentile latency: 2.316791471093893 seconds
90% percentile latency: 2.3175419165752826 seconds

More results:

# latency benchmark with default batchsize=8

```shell
python benchmark_latency.py --model CohereForAI/c4ai-command-r-v01 -tp=4
```

without this PR:
Avg latency: 2.900428555874775 seconds
10% percentile latency: 2.8955503132194282 seconds
25% percentile latency: 2.897518435958773 seconds
50% percentile latency: 2.9004721106030047 seconds
75% percentile latency: 2.9027304998598993 seconds
90% percentile latency: 2.9058421747758985 seconds

with this PR:
Avg latency: 2.73428927740703 seconds
10% percentile latency: 2.7297521037049592 seconds
25% percentile latency: 2.7316771240439266 seconds
50% percentile latency: 2.732553725130856 seconds
75% percentile latency: 2.734609372681007 seconds
90% percentile latency: 2.744487049616873 seconds

# throughput benchmark with default arg
```shell
python benchmark_throughput.py --model CohereForAI/c4ai-command-r-v01 --dataset ShareGPT_V3_unfiltered_cleaned_split.json -tp=4
```

Without this PR:
Throughput: 7.52 requests/s, 3290.22 tokens/s

With this PR:
Throughput: 7.91 requests/s, 3463.78 tokens/s

# Summary

5% latency & throughput improvement

## Comments

(no discussion comments)