# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)
        # Disable gradients for all parameters to replicate the issue scenario
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, dtype=torch.float32)

