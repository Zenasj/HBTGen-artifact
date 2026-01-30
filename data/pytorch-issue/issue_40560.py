import torch

a = torch.rand(10)
w = torch.ones(10)
assert (a * w).sum() / w.sum() == a.mean()