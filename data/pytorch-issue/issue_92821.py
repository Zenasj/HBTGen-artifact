import torch

a=torch.rand(3,3,3,device="mps")
print(a)
print(torch.unique(a))