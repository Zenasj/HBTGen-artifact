import torch

a = torch.zeros((3,))
b = torch.zeros((5,))
a @ b

a = torch.zeros((3,), device='meta')
b = torch.zeros((5,), device='meta')
a @ b   # size = ()