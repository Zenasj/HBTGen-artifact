# torch.rand(B, 1, dtype=torch.uint8), torch.rand(B, 1)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        condition = (b < 2.0)
        combined = a & condition  # Requires type promotion between Byte and Bool
        return combined.float().mean()

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randint(0, 2, (1,), dtype=torch.uint8)
    b = torch.rand(1)
    return (a, b)  # Returns tuple of two tensors as required by MyModel's forward

