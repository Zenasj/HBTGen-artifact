import torch

x = torch.tensor([10.], device=device)
y = vmap(lambda x: x.sum(0))(x)