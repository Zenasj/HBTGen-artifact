# torch.rand(3, 5, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        return self.log_softmax(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 5, requires_grad=True, dtype=torch.float32)

