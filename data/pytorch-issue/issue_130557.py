# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a = set(['a', 'b'])
        b = set(['a', 'c'])
        # The following line triggers Dynamo's unsupported 'intersection' method
        _ = a.intersection(b)  # Deliberately included to test Dynamo compatibility
        return x  # Return input tensor as a valid model output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

