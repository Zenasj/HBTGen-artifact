import torch

a = torch.tensor([[[0, 1]]])
print(a.unique())
print(a.to("mps").unique())
print(a.to("mps").cpu().unique())