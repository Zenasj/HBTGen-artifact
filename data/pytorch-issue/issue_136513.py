# torch.rand(3, 4, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.to(device="cuda", non_blocking=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(3, 4)

