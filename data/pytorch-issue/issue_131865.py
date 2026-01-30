_TORCH_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using torch device: {_TORCH_DEVICE}")


@functools.cache
def _build_llama_pipeline():
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        device=_TORCH_DEVICE,
    )
    return pipeline

...
messages = [
        {"role": "system", "content": prompt},
    ]
pipe = _build_llama_pipeline()
result = pipe(messages, max_new_tokens=200)

import torch
import transformers

if not torch.backends.mps.is_available():
    raise RuntimeError(
        "Please enable MPS on your machine. See https://pytorch.org/docs/stable/backends.html#torch-multiprocessing-mp"
    )
_TORCH_DEVICE = "mps"

pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    device=_TORCH_DEVICE,
)

needle = "needle"
hay = "hay"  # 3 + 1 chars per element

target_num_tokens_approximate = 20_000
haystack = [hay] * target_num_tokens_approximate
test_index = 42
haystack[test_index] = needle
haystack_list = list(haystack)
prompt = f"""
Please return the index of the needle in the haystack as an integer.
Return only the integer number index.
<BEGIN HAYSTACK>
{" ".join(haystack_list)}
<END HAYSTACK>
"""

messages = [
    {"role": "system", "content": prompt},
]
result = pipeline(
    messages,
    max_new_tokens=200,
)
messages = result[0]["generated_text"]
completion = messages[-1]["content"]

print("successful inference")
print(f"completion: {completion}")
print(f"test_index: {test_index}")