import torch
from torch.distributions import LowRankMultivariateNormal

DEVICE = "cuda"

torch.manual_seed(23)
for i in range(10):
    print(i)
    distrib = LowRankMultivariateNormal(
        torch.randn(1, 512, 512, 2).to(DEVICE),
        torch.randn(1, 512, 512, 2, 10).to(DEVICE),
        torch.randn(1, 512, 512, 2).to(DEVICE).exp()
    )