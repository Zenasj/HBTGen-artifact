# torch.rand(1024, dtype=torch.float32, device="cuda")  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            *[nn.Linear(1024, 1024, bias=False, device="cuda") for _ in range(10)]
        )
    
    def forward(self, x):
        return self.layers(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1024, dtype=torch.float32, device="cuda")

