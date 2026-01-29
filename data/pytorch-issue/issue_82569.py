# torch.rand(60, 1000, 1000, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Perform topk along the last dimension (dim=-1) with k=10
        values, _ = torch.topk(x, k=10, dim=-1)
        return values

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor that triggers the CUDA error in vulnerable PyTorch versions
    return torch.rand(60, 1000, 1000, dtype=torch.float32, device='cuda')

