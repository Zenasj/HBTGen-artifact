import torch
x = torch.tensor([[1., 2, 3], [4., 5, 6]], device='cuda:0')
c = torch.as_strided(x, size=[2, 2, 2], stride=[3, 1, 1])
torch.einsum('...ab,...bc->...ac', c, c)

import torch
x = torch.tensor([[1., 2, 3], [4., 5, 6]], device='cuda:0')
c = torch.as_strided(x, size=[2, 2, 2], stride=[3, 1, 1])
torch.mm(c[0], c[0]) # Fails