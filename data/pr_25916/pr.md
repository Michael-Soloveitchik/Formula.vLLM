# PR #25916: EAGLE 3: Fix preamble so that measured speedup over Eagle 1 becomes 32% instead of 5% on MTBench

- field: decoding  | area: vllm/benchmarks
- author: ekagra-ranjan
- state: MERGED  created: 2025-09-30T01:19:03Z  merged: 2025-10-02T18:29:35Z
- url: https://github.com/vllm-project/vllm/pull/25916
- labels: performance, ready

## Body

Fixes https://github.com/vllm-project/vllm/issues/20780

E3 has 20% better AL than E1 but the e2e TOPS was just 4%. The expectation was to see atleast 20% better e2e gains.
The issue was traced to data in benchmark having a very small difference which lead to this huge difference. Offline inference gave the AL of 2.79 for E3 on MTBench. However, the AL reported in Online (after hacking it to give the overall AL and not snapshot of AL by bypassing the reset of the prometheus metric) was ~2.2. This happened because both offline and online inference share the same dataset but offline sets `add_special_tokens` to False whereas online was setting it to True for llama 3.1 model based on model config. 

This meant the prompt being used in online serving had `<|begin_of_text|>` twice in the beginning, one from the chat template and one from the tokenizer.encode with `add_special_tokens` as True. This very small difference was enough to throw the E3 off balance and we see such sharp drop in AL. This sudden drop is not seen in E1 which is why this discrepancy in data bw online and offline inference was never discovered during E1 ablations. 

The right way is to skip chat template in the dataset builder and use `/v1/chat/completitions` endpoint as done below.

cmd:
server
`
VLLM_USE_V1=1 vllm serve meta-llama/Llama-3.1-8B-Instruct  --disable-log-requests --port 9001  --speculative_config '{"method": "eagle3","model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B", "num_speculative_tokens": 3}'
`

client
`
vllm bench serve --port 9001 --save-result --save-detailed --model meta-llama/Llama-3.1-8B-Instruct --temperature=0.0 --top-p=1.0 --dataset-name hf  --dataset-path philschmid/mt-bench  --num-prompts 80  --max-concurrency 4 --result-dir "./throwaway" --endpoint "/v1/chat/completions" --backend openai-chat --skip-chat-template
`

TPOT on MTBench BS4 on H100
* E1: 4.32ms
* E3 before this PR: 4.09ms (5.6% faster than E1)
* E3 with this PR: 3.25ms (32.9% faster than E1)

## Comments

### ekagra-ranjan @ 2025-09-30T03:10:51Z

vllm benchmark only supports /v1/completitions as of now. I can take a look if it can be extended to /v1/chat/completitions to avoid these changes.
---
### ekagra-ranjan @ 2025-09-30T16:01:36Z

@benchislett - updated to use /v1/chat/completitions API correctly.
---
### ekagra-ranjan @ 2025-09-30T23:13:55Z

@benchislett - can you add the ready tag and auto-merge on the PR so that CI can run?
---
### ekagra-ranjan @ 2025-10-01T15:01:26Z

`v1/spec_decode/test_eagle.py:373:` is failing [here](https://buildkite.com/vllm/ci/builds/33019/summary?jid=01999d0c-1593-4ce5-9f0d-6744f963e0ac) with 
`FAILED v1/spec_decode/test_eagle.py::test_load_model[True-1-FLASH_ATTN-eagle] - RuntimeError: generator raised StopIteration`. This PR doesnt touch that code so probably not related to this PR.
---