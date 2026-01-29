# torch.rand(B, C, H, W, dtype=torch.float32, device="cuda")
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x  # Identity model to trigger stride profiling

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor matching the input shape expected by MyModel
    # Example shape (B=4, C=1, H=8, W=8) as per first input in the issue
    return torch.randn(4, 1, 8, 8, dtype=torch.float32, device="cuda")

