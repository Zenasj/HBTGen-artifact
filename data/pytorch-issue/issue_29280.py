# torch.rand(B, C, H, W, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Performs reduction with BF16 input and FP32 accumulation (as per PR's intent)
        return x.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Matches shape and dtype from benchmark examples
    return torch.rand(1, 500, 500, 4, dtype=torch.bfloat16)

