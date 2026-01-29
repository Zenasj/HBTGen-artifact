# torch.rand(10000, 40000, dtype=torch.float32)  # Input shape (B, C)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40000, 10000)
    
    def forward(self, out):
        out = self.fc1(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel().cuda()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(10000, 40000, dtype=torch.float32).cuda()

