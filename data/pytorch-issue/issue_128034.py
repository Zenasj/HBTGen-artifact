# torch.rand(B, 1024, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(1024, 1024)
        self.layer1 = nn.Linear(1024, 1024)
    
    def forward(self, x):
        # Replicate inlined behavior by explicitly passing parameters
        x = torch._C._nn.linear(x, self.layer0.weight, self.layer0.bias)
        x = torch._C._nn.linear(x, self.layer1.weight, self.layer1.bias)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the 1024x1024 example input shape from the issue
    return torch.rand(1024, 1024, dtype=torch.float32)

