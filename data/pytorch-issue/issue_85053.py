# torch.rand(1, dtype=torch.float32)  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        out_shape = (5, 1, 1, 1)
        out_div = torch.ones(out_shape, dtype=x.dtype, device=x.device)
        out_mul = torch.ones(out_shape, dtype=x.dtype, device=x.device)
        out_remainder = torch.ones(out_shape, dtype=x.dtype, device=x.device)
        
        torch.div(x, 1, out=out_div)
        torch.mul(x, 1, out=out_mul)
        torch.remainder(x, 1, out=out_remainder)
        
        return out_div  # Returns one output for compilation, others trigger the error

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones(1)

