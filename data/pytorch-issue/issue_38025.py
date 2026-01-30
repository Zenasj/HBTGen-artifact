py
import torch

ones = torch.ones(4).cuda()
zeros = torch.zeros(4).cuda()
indices = torch.tensor([0, 1], dtype=torch.int32).cuda()

torch.scatter(ones, 0, indices, zeros)
print(ones.cpu())