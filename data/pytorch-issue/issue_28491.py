# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class X:
    pass

class Y(X):
    pass

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = Y()  # Problematic attribute causing TorchScript type error
        self.linear = nn.Linear(10, 10)  # Valid layer for forward compatibility

    def forward(self, x):
        # Returns valid tensor output while retaining problematic attribute
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor compatible with the model's forward()
    return torch.rand(1, 10, dtype=torch.float32)

