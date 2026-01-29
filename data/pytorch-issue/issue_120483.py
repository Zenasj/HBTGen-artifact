# torch.rand(1, 10, dtype=torch.float32, device='cuda')
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LayerNorm(10),
            nn.Linear(10, 20),
            nn.LayerNorm(20)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def my_model_function():
    # Initialize model with CUDA and required layers
    return MyModel().cuda()

def GetInput():
    # Generate input matching (B=1, C=10) with float32 and CUDA
    return torch.rand(1, 10, dtype=torch.float32, device='cuda')

