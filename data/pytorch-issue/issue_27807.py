# torch.rand(1, 300, 25000, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # This tensor movement to CUDA is slow in affected configurations
        return x.to("cuda")

def my_model_function():
    # Returns model that moves input to CUDA (problematic operation)
    return MyModel()

def GetInput():
    # Returns a tensor matching the input shape expected by MyModel
    return torch.rand(1, 300, 25000, 1)  # Matches B=1, C=300, H=25000, W=1

