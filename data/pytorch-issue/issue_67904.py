# torch.rand(3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Attempt to create sparse zero tensor matching input's shape/dtype
        return torch.zeros_like(x, layout=torch.sparse_coo)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the example input from the issue (3x3 dense tensor)
    return torch.rand(3, 3)

