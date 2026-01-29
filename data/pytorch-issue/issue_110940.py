# torch.rand(B, 1, dtype=torch.float64, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, device: str = "cuda"):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1, device=device, dtype=torch.float64)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 1, device='cuda', dtype=torch.float64)

