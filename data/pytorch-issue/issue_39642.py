# torch.rand(B, F, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.logsigmoid = nn.LogSigmoid()  # Core module from PyTorch's nn.LogSigmoid

    def forward(self, x):
        return self.logsigmoid(x)

def my_model_function():
    return MyModel()  # Returns the wrapped LogSigmoid model

def GetInput():
    return torch.randn(128, 10, requires_grad=True)  # Matches benchmark input shape (B=128, features=10)

