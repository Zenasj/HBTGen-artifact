# torch.rand(B, C, H, W, dtype=torch.float32)  # Example input shape (3, 6, 4, 2)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, mul=2, dim=-1):
        super().__init__()
        self.mul = mul
        self.dim = dim

    def forward(self, x):
        dim_size = x.shape[self.dim]
        divisor = self.mul
        quotient = dim_size // divisor
        remainder = dim_size % divisor
        is_divisible = (remainder == 0)
        
        # Convert to tensors for symbolic tracing compatibility
        is_divisible_tensor = torch.as_tensor(is_divisible)
        quotient_tensor = torch.as_tensor(quotient, dtype=torch.float)
        dim_size_tensor = torch.as_tensor(dim_size, dtype=torch.float)
        
        result = torch.where(is_divisible_tensor, quotient_tensor, dim_size_tensor)
        return result

def my_model_function():
    return MyModel(mul=2, dim=-1)

def GetInput():
    return torch.rand((3, 6, 4, 2), dtype=torch.float32)

