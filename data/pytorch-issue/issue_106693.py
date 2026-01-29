# torch.rand(3, 3, 2, 2, dtype=torch.float32)  # Inferred input shape from issue's sample
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return x.unsqueeze_(1)  # In-place unsqueeze operation causing FX graph issue

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, 2, 2, dtype=torch.float32)

