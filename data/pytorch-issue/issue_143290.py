import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention  # Import from user's provided path

# torch.rand(1, 8, 256, 128, dtype=torch.bfloat16, requires_grad=True)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flex_attention = torch.compile(flex_attention)  # Compile the function as in the issue

    def forward(self, x):
        return self.flex_attention(x, x, x)  # Uses same input for Q/K/V as in minimal repro

def my_model_function():
    return MyModel()  # Returns compiled model instance

def GetInput():
    return torch.randn(
        (1, 8, 256, 128),
        device='cuda',
        dtype=torch.bfloat16,  # dtype set to trigger the error
        requires_grad=True
    )

