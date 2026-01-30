import torch

def fn(x):
    return x.cos()

nt = torch.nested.nested_tensor_from_jagged(
    torch.randn(10, 10),
    torch.tensor([0, 1, 3, 6, 10]),
)

torch.compile(fn)(nt)