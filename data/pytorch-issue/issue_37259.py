# torch.rand(64, 8, 2, 128, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Simulate the tensor creation causing threading issues
        z = torch.zeros(64, 8, 2, 128, dtype=x.dtype, device=x.device)
        return x + z  # Dummy computation to ensure tensor is used

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(64, 8, 2, 128, dtype=torch.float32)

