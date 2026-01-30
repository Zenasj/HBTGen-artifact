import torch

print(torch.hardshrink(torch.tensor([float("nan")] * 17, dtype=torch.float32)))