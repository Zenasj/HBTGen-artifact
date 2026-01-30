import torch
x = torch.randn(2, device="cuda")
assert x == x.clone()