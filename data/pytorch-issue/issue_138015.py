import torch
torch.set_default_dtype(torch.bfloat16)

@torch.compile
def f():
    return torch.randn(1, dtype=torch.bfloat16)

print("Expected: torch.bfloat16\nOutput:", f().dtype)

Expected: torch.bfloat16
Output: torch.float32