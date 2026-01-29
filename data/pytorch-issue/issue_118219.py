# torch.rand(4, 32, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Linear layer with in_features=32 (from input's permuted dim) and out_features=4
        self.linear = nn.Linear(32, 4, bias=True)

    def forward(self, x):
        tmp = x.permute(0, 2, 1).contiguous()  # Permute and ensure contiguous
        out = self.linear(tmp)                 # Apply linear layer
        return out.permute(0, 2, 1)            # Reverse permutation

def my_model_function():
    # Initialize the model with default parameters (random weights/bias)
    return MyModel()

def GetInput():
    # Generate input matching the model's expected shape with gradients enabled
    return torch.randn(4, 32, 4, requires_grad=True)

