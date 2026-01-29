# torch.rand(5, 68, 64, 64, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compute PyTorch's default sum (float32)
        sum_pt = x.sum()
        # Compute higher-precision sum (float64 converted back to float32)
        sum_f64 = x.double().sum().float()
        # Return the difference between the two methods
        return sum_f64 - sum_pt

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces the input from the original issue (float32 tensor of 0.1s)
    return torch.ones(5, 68, 64, 64, dtype=torch.float32) * 0.1

