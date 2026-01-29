# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)  # Inferred from common usage patterns

    def forward(self, x):
        x = self.linear(x)
        return torch.floor(x)  # Problematic operator causing ONNX export failure

def my_model_function():
    # Returns a model instance with random initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.randn(1, 10)  # Batch size 1, 10 features

