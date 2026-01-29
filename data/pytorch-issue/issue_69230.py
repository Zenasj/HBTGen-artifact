# torch.rand((), dtype=torch.float32), target=torch.tensor(-1., dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_desired_kl(self, input, target):
        output = F.kl_div(input, target)
        mask = target < 0
        output[mask] = float('nan')
        return output

    def forward(self, inputs):
        input, target = inputs
        current = F.kl_div(input, target)
        desired = self.compute_desired_kl(input, target)
        # Return True where outputs differ (NaN vs 0)
        return torch.isnan(desired) != torch.isnan(current)

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.rand((), dtype=torch.float32).log()
    target = torch.tensor(-1., dtype=torch.float32)
    return (input, target)

