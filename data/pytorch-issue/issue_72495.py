# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn
from torch.distributions import Transform, Normal, Weibull

class CumulativeDistributionTransform(Transform):
    def __init__(self, dist):
        super().__init__()
        self.dist = dist

    def _call(self, x):
        return self.dist.cdf(x)

    def _inverse(self, y):
        return self.dist.icdf(y)

    def log_abs_det_jacobian(self, x, y):
        return self.dist.log_prob(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal_cdf = CumulativeDistributionTransform(Normal(0, 1))
        self.weibull_icdf = CumulativeDistributionTransform(Weibull(4, 2)).inv

    def forward(self, x):
        x = self.normal_cdf(x)
        x = self.weibull_icdf(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.float32)  # B=2, 2D input as per base_dist examples

