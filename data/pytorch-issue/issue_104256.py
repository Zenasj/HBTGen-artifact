# torch.rand(1, 1, 2, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Create a sparse tensor matching the example's 2x2 structure
    dense = torch.rand(1, 1, 2, 2)
    sparse = dense.to_sparse()
    return sparse

