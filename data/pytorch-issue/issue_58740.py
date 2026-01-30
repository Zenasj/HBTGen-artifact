import torch

a = torch.tensor(True)
b = True

assert a & b
assert torch.logical_and(a, b)