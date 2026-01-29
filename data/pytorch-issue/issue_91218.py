# torch.rand(B, C, H, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        r = x.mT  # Transpose last two dimensions (problematic operation)
        return torch.nn.functional.relu(r)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 4)  # Matches the example input shape (2,3,4)

