# torch.rand(B, C, H, W, dtype=torch.complex128)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(10, dtype=torch.complex128))  # Matches input's last dimension

    def forward(self, x):
        # Example operation causing historical Inductor compilation issue with complex tensors
        return x + self.bias  # Complex addition

def my_model_function():
    return MyModel()

def GetInput():
    # 4D tensor with shape (B, C, H, W) where W=10 to match parameter's dimension
    return torch.rand(1, 1, 1, 10, dtype=torch.complex128)

