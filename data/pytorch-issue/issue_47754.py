import torch

py
inputs = torch.randn(N, 3)
f = lambda x: x.clamp(min=0)
expected = torch.stack([f(inputs[i]) for i in range(N)])
result = torch.vmap(f)(inputs)
assert torch.allclose(result, expected)