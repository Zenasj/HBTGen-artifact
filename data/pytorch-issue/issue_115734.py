import torch

a = torch.tensor([1,2,3,4],dtype=float,device="cuda:0")
b = torch.sum(a)
print(b)