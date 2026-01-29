# torch.rand(2, dtype=torch.float32)  # Inferred input shape is a 1D tensor of size 2
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        sigmoid = torch.sigmoid(x)
        ones = torch.ones(2, dtype=x.dtype, device=x.device)
        mul = torch.mul(sigmoid, ones)
        relu = torch.relu(mul)
        zeros = torch.zeros(2, dtype=x.dtype, device=x.device)
        add = torch.add(relu, zeros)
        round_1 = torch.round(add)  # Using torch.round instead of torch.ops.aten.round for compatibility
        return round_1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)  # Matches the input shape and dtype used in the test case

