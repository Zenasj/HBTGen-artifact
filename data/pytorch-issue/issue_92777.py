# torch.rand(B, C, D, H, W, dtype=torch.float64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # This layer triggers the reported floating point exception
        return torch.nn.functional.adaptive_max_pool3d(x, [0, 0, 0], False)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from the first example in the issue
    return torch.rand(2, 3, 6, 4, 10, dtype=torch.float64)

