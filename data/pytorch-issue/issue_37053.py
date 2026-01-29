# torch.rand(10000000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert input to different dtypes and compute sums with specified output dtypes
        x_float = x
        x_int32 = x.to(torch.int32)
        x_bool = x.to(torch.bool)
        x_uint8 = x.to(torch.uint8)
        
        # Sums for each dtype, using torch.int32 for non-float types per benchmark setup
        sum_float = x_float.sum()  # Float sum remains in float
        sum_int32 = x_int32.sum(dtype=torch.int32)
        sum_bool = x_bool.sum(dtype=torch.int32)
        sum_uint8 = x_uint8.sum(dtype=torch.int32)
        
        return sum_float, sum_int32, sum_bool, sum_uint8

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates a 1D tensor matching the benchmark's input requirements
    return torch.randint(2, (10000000,), dtype=torch.float32)

