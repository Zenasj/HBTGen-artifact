import torch
import torch.nn.functional as F

with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(query,key,value,
                #attn_mask=get_attn_mask(query, key),
                dropout_p=0.0,   
                scale=scale, # does not matter
                is_causal=True)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "How to start GPU programming on ROCm",
]
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="/app/model")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")