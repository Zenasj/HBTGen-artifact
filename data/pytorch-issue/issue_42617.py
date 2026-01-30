import torch

tensor = torch.zeros(B0, 1)
vmap(lambda t: t.normal_())(tensor)