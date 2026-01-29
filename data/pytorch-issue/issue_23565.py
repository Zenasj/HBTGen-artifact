# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Emulates the exception-handling scenario from the PR's example
        if x[0, 0, 0, 0] == 1:
            raise Exception("Invalid input")
        return x * 2  # Example computation continuing after valid path

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a valid input tensor (B=1, C=1, H=1, W=1) with values <1 (to avoid exception)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)

