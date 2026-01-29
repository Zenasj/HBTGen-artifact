# torch.rand(B, 4, dtype=torch.float, requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        concentration = torch.sigmoid(x)
        gamma = torch.distributions.Gamma(concentration, torch.tensor(1.0, device=x.device))
        samples = gamma.rsample()
        return samples / samples.sum(dim=-1, keepdim=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4, requires_grad=True)

