# torch.rand(B=2, C=10, H=3, W=4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight1 = nn.Parameter(torch.randn(10, 20))
        self.weight2 = nn.Parameter(torch.randn(20, 30))
        
    def forward(self, x):
        # Process 4D input into 2D for matrix operations
        batch, channels, height, width = x.shape
        x = x.view(batch * height * width, channels)  # Flattens spatial dimensions
        x = torch.mm(x, self.weight1)  # First matrix multiplication
        x = torch.mm(x, self.weight2)  # Second matrix multiplication
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Create non-contiguous input tensor
    input = torch.rand(2, 10, 3, 4, dtype=torch.float32)
    input = input[:, :, :, :3]  # Slice to make non-contiguous
    return input

