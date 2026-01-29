# torch.rand(3, dtype=torch.float32, device='cuda')  # Inferred input shape based on test case
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = torch.tensor([2., 2., 2.], dtype=torch.float32, device='cuda')  # Constant tensor causing fusion issue

    def forward(self, x):
        # Operations involving constant tensor and fusion-prone ops
        out = x + self.const  # Constant tensor used in computation
        out = out + 3.0       # Additional operation to create fusion pattern
        return out

def my_model_function():
    return MyModel()  # Returns model instance with problematic constant tensor

def GetInput():
    # Returns random tensor matching input shape (3 elements float32)
    return torch.rand(3, dtype=torch.float32, device='cuda')

