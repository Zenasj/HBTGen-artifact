# torch.rand((), dtype=torch.float32)  # Scalar input as in the issue example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.prelu = nn.PReLU()  # Core PyTorch PReLU module causing the ONNX export issue

    def forward(self, x):
        return self.prelu(x)

def my_model_function():
    return MyModel()  # Returns the problematic PReLU model instance

def GetInput():
    return torch.rand((), dtype=torch.float32)  # Matches scalar input shape from the issue's test case

