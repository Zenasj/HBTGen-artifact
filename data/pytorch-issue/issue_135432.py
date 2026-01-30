import torch

@torch.compile
def foo(x):
    return torch.Tensor(x)

print(foo([1, 2]))
print(foo([3, 4]))