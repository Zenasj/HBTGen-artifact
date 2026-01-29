# torch.rand(2, dtype=torch.float32, device=torch.device("cuda"))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Split input tensor into a list of individual tensors (as in the numpy error example)
        return [x[0], x[1]]

def my_model_function():
    return MyModel()

def GetInput():
    # Create a CUDA tensor with 2 elements to match the issue's scenario
    return torch.rand(2, dtype=torch.float32, device=torch.device("cuda"))

