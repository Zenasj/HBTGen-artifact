# torch.rand(1, 512, 512, 2, dtype=torch.float32)
import torch
from torch.distributions import LowRankMultivariateNormal
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Replicate the problematic distribution initialization each forward pass
        device = x.device
        loc = torch.randn(1, 512, 512, 2, device=device)
        cov_factor = torch.randn(1, 512, 512, 2, 10, device=device)
        cov_diag = torch.randn(1, 512, 512, 2, device=device).exp()
        distrib = LowRankMultivariateNormal(loc, cov_factor, cov_diag)
        return distrib.mean  # Dummy output to satisfy forward requirements

def my_model_function():
    return MyModel()

def GetInput():
    # Return input matching the expected shape (loc dimensions)
    return torch.rand(1, 512, 512, 2, dtype=torch.float32)

