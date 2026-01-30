import torch
x = torch.tensor([float("nan")])
assert torch.equal(x, x)  # ❌ fails