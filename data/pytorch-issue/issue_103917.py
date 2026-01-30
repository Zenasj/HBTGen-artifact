import torch

x1 = torch.as_tensor([0], dtype=torch.int32)
x2 = torch.as_tensor([0], dtype=torch.int32)
torch.inner(x1, x2)