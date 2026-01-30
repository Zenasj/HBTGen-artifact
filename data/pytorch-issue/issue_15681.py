import torch
g = torch.distributions.Gumbel(loc=0.0, scale=1.0)
print(g.log_prob(-5.0))