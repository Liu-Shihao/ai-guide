import transformers
import torch
"""
# TypeError: BFloat16 is not supported on MPS

The search results indicate that the BFloat16 data type is not currently supported on the MPS (Metal Performance Shaders) backend on macOS devices with Apple Silicon chips like the M1 and M2.

The key points are:

- The M1 chip uses the ARMv8.5-A instruction set, which does not support BFloat16. BFloat16 support was added in the later ARMv8.6-A instruction set, which is used in the M2 chip.[1]

- PyTorch does not support using the BFloat16 data type with the MPS backend, as it is an intended behavior due to the lack of BFloat16 support in the underlying hardware.[1]

- While the M2 chip does support BFloat16, PyTorch has not yet added support for using BFloat16 with the MPS backend.[2][3][4]

- As a workaround, you can try running the model on the CPU instead of the MPS device, but this will likely result in slower performance.[5]

In summary, the error "TypeError: BFloat16 is not supported on MPS" is due to the current limitations of the MPS backend on macOS devices with Apple Silicon chips. This issue is expected to be resolved in the future as PyTorch adds support for BFloat16 on the MPS backend, but for now, the only solution is to use the CPU instead of the MPS device.[1][2][3][4][5]

Citations:
[1] https://github.com/pytorch/pytorch/issues/78168
[2] https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/discussions/13
[3] https://github.com/Lightning-AI/litgpt/issues/498
[4] https://stackoverflow.com/questions/77359161/bfloat16-is-not-supported-on-mps-macos
[5] https://huggingface.co/stabilityai/stablecode-instruct-alpha-3b/discussions/6
"""
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cpu",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])

