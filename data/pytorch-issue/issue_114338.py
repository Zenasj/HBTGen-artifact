# torch.rand(10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        with torch.enable_grad():
            out = x + 1
        out2 = out + 1
        return out, out2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(10, requires_grad=True)

