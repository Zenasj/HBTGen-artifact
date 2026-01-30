import torch
from torch.distributions import *
b1 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
b2 = MultivariateNormal(torch.tensor([0.0]), torch.eye(1))
t = StickBreakingTransform()
d1 = TransformedDistribution(b1, t)
d2 = TransformedDistribution(b2, t)
print(d1.base_dist)  # Independent with 1 dimension
print(d2.base_dist)  # MultivariateNormal