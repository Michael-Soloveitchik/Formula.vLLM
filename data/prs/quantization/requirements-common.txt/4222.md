# PR #4222: Updating lm-format-enforcer version and adding links to decoding libraries in docs

- field: quantization  | area: requirements-common.txt
- author: noamgat
- state: MERGED  created: 2024-04-20T07:38:43Z  merged: 2024-04-20T08:33:16Z
- url: https://github.com/vllm-project/vllm/pull/4222
- labels: 

## Body

- Updated lm-format-enforcer version from 0.9.3 to 0.9.8. These versions have no API changes, solve a few bugs and improve JSONSchemaParser performance by 2x in many cases.
- Added links to the libraries in the guided-decoding-backend docstring, so the vLLM OpenAI Compatible Server docs page now looks like this:
![image](https://github.com/vllm-project/vllm/assets/1331304/547d791a-ea7f-450b-bc31-273e2943e15f)
I think it will help users understand what this parameter means.


## Comments

### noamgat @ 2024-04-20T07:47:06Z

I also made a change not directly for this PR, because I saw that the doc generation CI runs on python 3.8 and it crashed due to a tuple type definition being done with `tuple` instead of `Tuple`
---