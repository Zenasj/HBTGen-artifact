# torch.rand(1, 1, 1, dtype=torch.float32)  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Generate tensor using torch.randint with signed int64
        a = torch.randint(-1, 0, (1, 1, 1), device=x.device, dtype=torch.int64)
        # Perform bitwise right shift (problematic when compiled)
        res = torch.bitwise_right_shift(a, 31)
        return res

def my_model_function():
    return MyModel()

def GetInput():
    # Return dummy input tensor matching expected shape (shape doesn't affect computation)
    return torch.rand(1, 1, 1)

