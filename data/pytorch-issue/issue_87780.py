# torch.randn(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        return self.softplus(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, dtype=torch.float32)

