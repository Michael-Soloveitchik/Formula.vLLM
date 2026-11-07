# PR #24989: [Core] Get num_encoder_tokens from scheduler config

- field: scheduling  | area: v1/core
- author: russellb
- state: MERGED  created: 2025-09-16T19:00:19Z  merged: 2025-09-17T03:59:07Z
- url: https://github.com/vllm-project/vllm/pull/24989
- labels: ready, v1

## Body

This is the same change that was made in #24866. In that PR, it was
pointed out that this code:

    MULTIMODAL_REGISTRY.get_encdec_max_encoder_len(...)

is much slower than getting the same value that's cached on the
scheduler config:

    scheduler_config.max_num_encoder_input_tokens

This PR makes the change to more spots: the scheduler, kv cache
manager, and gpu model runner.

Related to issue #24946.

Signed-off-by: Russell Bryant <rbryant@redhat.com>


## Comments

### russellb @ 2025-09-16T19:09:01Z

This isn't a huge improvement (about 3% throughput increase), but it's a good step. There's still something more significant limiting throughput that I'm tracking down.
---