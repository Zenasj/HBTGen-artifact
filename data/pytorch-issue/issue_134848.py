import torch

x1 = torch.tensor([[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0, 2.0]], [[3.0, 3.0, 3.0, 3.0]]], dtype=torch.float16)
y1 = torch.empty((1, 1, 4), dtype=torch.float16)
torch.mean(x1, dim=0, keepdim=True, out=y1)

print(y1)