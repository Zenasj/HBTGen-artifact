import torch
import torch.nn as nn

# torch.rand(B, 3, dtype=torch.float32)  # Matches input shape from the example
class MyModel(nn.Module):
    def forward(self, x):
        if hasattr(x, "to"):
            return x.to("cpu")
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)  # Matches the input type and shape in the minified repro

