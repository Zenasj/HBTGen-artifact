import torch

x = torch.tensor([1, 2])
index=torch.tensor([0, 1])

x.scatter_(dim=0, index=index, value=0)
print(x)  # tensor([0, 0])

x.scatter_(dim=dim, index=index, src=0.0)