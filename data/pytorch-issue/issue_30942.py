# torch.rand(30, 18, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Perform SVD with full matrices (some=False)
        u, s, v = torch.svd(x, some=False)
        return u, s, v

def my_model_function():
    return MyModel()

def GetInput():
    # Return random input matching the 2D matrix shape from the issue
    return torch.rand(30, 18, dtype=torch.float32)

