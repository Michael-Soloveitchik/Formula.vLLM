# PR #29322: [LoRA] Continue optimizing MoE LoRA weight loading

- field: decoding  | area: lora/layers
- author: jeejeelee
- state: MERGED  created: 2025-11-24T16:19:28Z  merged: 2025-11-27T13:56:28Z
- url: https://github.com/vllm-project/vllm/pull/29322
- labels: ready, qwen

## Body

<!-- markdownlint-disable -->

## Purpose
Currently, for MoE LoRA, we need to handle 2 formats of weights:
### First format
Similar to GPT-OSS's MoE styple, where the MoE is a 3D weight—meaning all expert weights are fused into one large weight. When adding LoRA to this type of MoE, each MoE layer will only have 2 LoRAs: W13(gate_up_proj) and W2(down_proj), see:[gpt-oss-lora](https://huggingface.co/jeeejeee/gpt-oss-20b-lora-adapter-text2sql/blob/main/adapter_model.safetensors). We have optimized the  loading  of this format at https://github.com/vllm-project/vllm/pull/29222
### Second format
Each expert's W1(gate_proj), W2(down_proj), and W3(up_proj) are separate linear layers, and each linear layer will have a LoRA added to it. Each expert has 3 LoRAs. If a model has M experts and N decoder layers, then for all MoE modules, there will be a total of 3MN LoRAs (not counting the `router`). For example, [OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) has 64 experts and 16 layers, there will be 3,072 LoRAs. Since each LoRA contains lora_a and lora_b, there will be a total of 6,144 lora weights, see: [olmoe-lora](https://huggingface.co/jeeejeee/olmoe-instruct-text2sql-spider/blob/main/adapter_model.safetensors)

This PR focus on optimizing the second format. Main Optimization Points:

- In the `check_unexpected_modules` function, use `set` instead of `list`, which reduces query complexity. Using [olmoe-lora](https://huggingface.co/jeeejeee/olmoe-instruct-text2sql-spider/blob/main/adapter_model.safetensors) testing, the time consumption decreased from 200ms to 20ms.
- Fuse small LoRA weights into one large 3D weight to reduce the number of CPU->GPU copy operations. Time consumption decreased from 196ms to 30ms.


- main
<img width="2933" height="683" alt="image" src="https://github.com/user-attachments/assets/49fc440f-fefe-4a8b-935e-5ab7a1e48b63" />

- this PR
<img width="3037" height="721" alt="image" src="https://github.com/user-attachments/assets/3b4ea01e-53f7-40f8-8677-47aef3c066f9" />


### TODO
### pin memory optimzation
Still, due to the large number of LoRA weights, the `pin_memory `operation overhead is quite significant and will need further optimization in the future.

### lora.optimze() optimization

The current PR moved lora.optimize() to execute in `_create_merged_loras_inplace`, which also increased `_create_merged_loras_inplace` time. In the future, maybe we can execute `lora.optimize()` function within the LoRA's kernel function.

## Test Plan

## Test Result

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>



## Comments

(no discussion comments)