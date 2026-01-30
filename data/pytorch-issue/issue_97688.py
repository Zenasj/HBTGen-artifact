import torch

@torch.compile()
def f(x, inf):
    return x + inf

print(f(torch.randn(2), 3))
print(f(torch.randn(2), 3))