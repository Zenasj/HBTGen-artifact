# torch.rand(1, 1, 1, 1, dtype=torch.float32)  # Input shape inferred as scalar in 4D tensor format
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Convert input to float16 and bfloat16, then compute the difference between them
        x_fp16 = x.to(torch.float16).to(torch.float32)  # Convert back to float32 for accurate difference calculation
        x_bf16 = x.to(torch.bfloat16).to(torch.float32)
        return x_fp16 - x_bf16  # Return the difference to highlight precision discrepancy

def my_model_function():
    return MyModel()

def GetInput():
    # Create a 4D tensor with value 3811.0 to trigger the precision issue
    return torch.tensor([3811.0], dtype=torch.float32).view(1, 1, 1, 1)

