# torch.rand(B, 2, 3, dtype=torch.complex32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Complex32 parameter to trigger the serialization issue
        self.param = nn.Parameter(torch.rand(1, 2, 3, dtype=torch.complex32))
    
    def forward(self, x):
        return x + self.param  # Example operation using the problematic dtype

def my_model_function():
    return MyModel()

def GetInput():
    # Returns input matching the model's expected dimensions and dtype
    return torch.rand(1, 2, 3, dtype=torch.complex32)

