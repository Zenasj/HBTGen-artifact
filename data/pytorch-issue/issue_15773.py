# torch.rand(1, device='cuda')  # dummy input (sample_shape is fixed to 10)
import torch
from torch import nn
from torch.distributions import Gamma

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(3, 4, 1, 5, device='cuda'))  # Matches original shape (3,4,1,5)
        self.b = nn.Parameter(torch.ones(3, 4, 1, 5, device='cuda'))
    
    def forward(self, dummy_input):
        # Gamma-based workaround for missing CUDA Beta distribution gradient
        s1 = Gamma(self.a, 1.0).rsample(torch.Size([10]))
        s2 = Gamma(self.b, 1.0).rsample(torch.Size([10]))
        return s1 / (s1 + s2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, device='cuda')  # Dummy input matching forward() requirement

