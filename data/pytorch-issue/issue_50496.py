import torch.distributions as dist

dist.Independent(dist.Normal(torch.zeros(1), torch.ones(1)), 1).support == dist.Normal(torch.zeros(1), torch.ones(1)).support