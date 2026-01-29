# torch.rand(B, C, H, W, dtype=torch.float32).to('cuda')  # Input shape: (1, 2, 4, 8)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(8, 16, bias=False)
    
    def forward(self, x):
        return torch.nn.functional.scaled_dot_product_attention(x, x, self.l(x), is_causal=True)

def my_model_function():
    # Returns model instance initialized on CUDA
    return MyModel().to('cuda')

def GetInput():
    # Returns a random tensor matching the required input shape and device
    return torch.rand(1, 2, 4, 8, dtype=torch.float32).to('cuda')

