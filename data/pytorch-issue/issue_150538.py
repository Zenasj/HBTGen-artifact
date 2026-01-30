import torch

torch.manual_seed(42)
print("torch.rand(5):", torch.rand(5))
print("torch.empty(5).exponential_(1):", torch.empty(5).exponential_(1))