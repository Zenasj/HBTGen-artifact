import torch

a = torch.tensor([[1,2]]).expand(3, -1)
b = torch.tensor([[10], [20], [30]])

tensor([[ -9,  -8],
        [-19, -18],
        [-29, -28]])

tensor([[-59, -58],
        [-59, -58],
        [-59, -58]])