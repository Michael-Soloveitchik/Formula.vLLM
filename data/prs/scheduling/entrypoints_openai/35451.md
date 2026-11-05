# PR #35451: [Core] Add optional flags to check for repetitive token patterns in engine output

- field: scheduling  | area: entrypoints/openai
- author: aykoppol
- state: MERGED  created: 2026-02-26T22:52:29Z  merged: 2026-03-03T04:23:26Z
- url: https://github.com/vllm-project/vllm/pull/35451
- labels: frontend, ready, v1

## Body

## Purpose
This PR allows users to end generation early if they see repetitive token patterns (for example, nonsensically repeated sentences) in their output. This is a real failure case for models prone to hallucination. Testing with Qwen3 VL on multimodal input shows a ~33% increase in throughput and near-total reduction in models hitting the configured max_output_len.

This feature is off by default. We include it in the core so that this can be leveraged for non-streaming generation too.

## Test Plan
New test file, online tests

## Test Result
Huge increase in throughput for error-prone multimodal input without sacrificing quality/latency.


## Comments

### aykoppol @ 2026-02-28T00:02:18Z

Pushed new version addressing comments, thanks for the review! cc @DarkLight1337 
---