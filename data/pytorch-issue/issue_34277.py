import torch

z = torch.distributions.Poisson(torch.tensor([4]))
z.sample()