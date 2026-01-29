# torch.rand(1, dtype=torch.float32, requires_grad=False)
import torch
from torch import nn

class ModelA(nn.Module):
    def forward(self, x):
        return torch.gather(x, 0, torch.tensor(0, device=x.device))

class FuncModel(nn.Module):
    def forward(self, x):
        return torch.gather(x, 0, torch.tensor(0, device=x.device))

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = ModelA()  # Original Model from the issue
        self.model_b = FuncModel()  # Function wrapped as a module
    
    def forward(self, x):
        # Execute both models and return their outputs
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        return (out_a, out_b)  # Return tuple for comparison purposes

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32, requires_grad=False)

