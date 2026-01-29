# torch.rand(3, 4, dtype=torch.float32).cuda()
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        mask = x.ge(0.5)  # Create mask based on input tensor
        return x.masked_select(mask)  # Operation causing CUDA graph capture failure

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, dtype=torch.float32).cuda()  # Matches input shape and device

