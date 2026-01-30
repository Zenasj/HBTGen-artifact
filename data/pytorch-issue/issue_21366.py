import torch
torch.manual_seed(0)
x = torch.randn(10)
P = torch.exp(-(x - x.unsqueeze(-1)) ** 2)
torch.distributions.MultivariateNormal(loc=torch.ones(10), precision_matrix=P)