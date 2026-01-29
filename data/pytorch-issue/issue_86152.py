# torch.rand(B, 2800, dtype=torch.float)  # Input is float32, converted to half for GELU
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = nn.GELU()  # GELU layer causing issues in half precision
        
    def forward(self, x):
        # Convert to half to trigger GELU in half precision (as in the error example)
        x_half = x.half()
        x_gelu = self.gelu(x_half)
        
        # Reproduce repeat_interleave operation (from error examples)
        x_repeated = x_gelu.repeat_interleave(2, dim=0)  # Triggers aten::repeat_interleave
        
        # Reproduce slicing operation leading to buffer issues
        sz = x_repeated.size(1)
        sliced = x_repeated[:, sz-128:]  # Triggers buffer allocation checks
        
        return sliced

def my_model_function():
    return MyModel()

def GetInput():
    # Generates input matching the model's expected shape and dtype
    return torch.rand(1, 2800)  # (B=1, features=2800)

