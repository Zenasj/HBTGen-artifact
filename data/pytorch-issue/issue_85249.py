import torch

input = torch.tensor(1)
other = torch.tensor([1]) 
out = torch.empty(3, 4, 5)

torch.lcm(input,other,out=out)