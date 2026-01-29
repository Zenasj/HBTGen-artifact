# torch.rand(2, 3, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, other=1, alpha=2):
        super().__init__()
        self.other = other
        self.alpha = alpha
    
    def forward(self, inputs):
        out = torch.empty_like(inputs)
        return torch.add(inputs, self.other, alpha=self.alpha, out=out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, 4)

