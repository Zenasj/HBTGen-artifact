import torch

def foo(x):
    return x > .2

inp = torch.tensor([.2002], device="cuda", dtype=torch.bfloat16)
print(foo(inp))

print(torch.compile(foo)(inp))