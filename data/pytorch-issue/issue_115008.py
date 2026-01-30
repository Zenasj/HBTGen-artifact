from transformers import AutoModelForCausalLM
import torch
from torch.sparse import to_sparse_semi_structured
import torch.nn as nn

from huggingface_hub import login

login("<insert_access_token_here>")

device = "cuda"
model_path = "meta-llama/Llama-2-7b-hf"


@torch.compile
def to_sparse_semi_structured_compiled(x):
    return to_sparse_semi_structured(x)


model = AutoModelForCausalLM.from_pretrained(model_path)
model = model.to(device).half()

# Enforcing N:M sparsity
for fqn, module in model.named_modules():
    if isinstance(module, nn.Linear):
        mask = (
            torch.Tensor([0, 0, 1, 1])
            .tile((module.weight.shape[0], module.weight.shape[1] // 4))
            .half()
            .to("cuda")
            .bool()
        )
        module.weight = torch.nn.Parameter(mask * module.weight)

mem = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"Mem: {mem:.3f}MB")
        
for fqn, module in model.named_modules():
    if isinstance(module, nn.Linear):
        module.weight = nn.Parameter(to_sparse_semi_structured_compiled(module.weight))

mem = torch.cuda.memory_allocated() / (1024 ** 2)
print(f"Mem: {mem:.3f}MB")